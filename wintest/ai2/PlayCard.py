from CreateActionList import CreateActionList
from CountValue import CountValue
from config import CompareRank
import config
import json
import time
from strategy import Strategy


class PlayCard():

    def _eval_cards_from_full_action(self, act):
        """Return cards list for evaluation (参谋按 virtual_ranks 变成虚拟点数牌).

        - 真实扣牌/剩余手牌计算必须使用 act[2] 原始牌面。
        - 评估/算分可使用虚拟点数（不依赖花色），以便理解参谋补牌的消歧义信息。
        """
        try:
            cards = act[2]
        except Exception:
            return []
        if not isinstance(cards, list):
            return cards

        try:
            if isinstance(act, list) and len(act) >= 4 and isinstance(act[3], dict):
                vr = act[3].get("virtual_ranks")
                if isinstance(vr, list) and len(vr) == len(cards):
                    out = []
                    for c, r in zip(cards, vr):
                        # 只有参谋才会出现 r != c[-1]；用统一黑桃承载点数即可。
                        if isinstance(c, str) and len(c) >= 2 and isinstance(r, str) and r != c[-1]:
                            out.append("S" + r)
                        else:
                            out.append(c)
                    return out
        except Exception:
            pass
        return cards

    def _adviser_opportunity_cost(self, cards, curRank, hand_size=None):
        """Opportunity-cost penalty for consuming adviser cards ('H'+curRank).

        设计目标：早期惩罚重、晚期惩罚轻。

        - ai2 的剩余手牌估值(restValue)不会把参谋当癞子，因此容易过早打掉参谋。
        - 这里用一个“阶段缩放”的机会成本惩罚，引导早期保留参谋以获得未来灵活性。
        """
        if not isinstance(cards, list):
            return 0.0

        adviser = 'H' + str(curRank)
        n = 0
        for c in cards:
            if c == adviser:
                n += 1
        if n <= 0:
            return 0.0

        stage = None
        try:
            stage = getattr(Strategy, 'roundStage', None)
        except Exception:
            stage = None

        # 每张参谋的基础机会成本：阶段越早越重（可按体验再调）
        if stage == 'ending':
            per = 0.15
        elif stage == 'middle':
            per = 0.30
        elif stage == 'beginning':
            per = 0.45
        else:
            # 兜底：按手牌张数粗略判断阶段
            try:
                hs = int(hand_size) if hand_size is not None else None
            except Exception:
                hs = None
            if hs is None:
                per = 0.35
            elif hs >= 15:
                per = 0.45
            elif hs >= 9:
                per = 0.30
            else:
                per = 0.15

        return per * float(n)

    def _best_from_full_action_list(self, handCards, curRank, fullActionList, mode="free", formerAction=None):
        """Pick the best action directly from server-provided actionList.

        This is necessary for 参谋多解：同一组真实牌面可能对应多种合法补法，
        仅靠 (type, rank, cards) 可能无法稳定消歧义。
        """
        # fullActionList is a list like: [[type, rank, cards, (optional detail)], ...]
        if not isinstance(fullActionList, list) or len(fullActionList) == 0:
            return None

        # Prefer non-PASS when leading.
        allow_pass = True
        if formerAction is None:
            allow_pass = False

        best = None
        best_idx = 0
        best_value = -10**9

        for idx, act in enumerate(fullActionList):
            try:
                typ = act[0]
                rank = act[1]
                cards = act[2]
            except Exception:
                continue

            if typ == 'PASS' and not allow_pass:
                continue

            # 扣牌必须用真实牌面
            restCards = CreateActionList().GetRestCards(cards if typ != 'PASS' else [], handCards)
            restValue, restActions = CountValue().HandCardsValue(restCards, 0, curRank)
            # 算分可用虚拟点数（参谋消歧义）
            eval_cards = self._eval_cards_from_full_action(act) if typ != 'PASS' else []
            thisHandValue = CountValue().ActionValue(eval_cards, typ, rank, curRank)

            if mode == "free":
                thisHandValue += Strategy.freeActionRV.get(typ, 0)
                if (typ, rank) in Strategy.freeActionRV:
                    thisHandValue += Strategy.freeActionRV[(typ, rank)]
            else:
                thisHandValue += Strategy.restrictedActionRV.get(typ, 0)
                if (typ, rank) in Strategy.restrictedActionRV:
                    thisHandValue += Strategy.restrictedActionRV[(typ, rank)]

            if thisHandValue < 0:
                thisHandValue = 0
            total = thisHandValue + restValue
            # 机会成本：用掉参谋要扣分，避免过于激进。
            if typ != 'PASS':
                total -= self._adviser_opportunity_cost(cards, curRank, hand_size=len(handCards) if isinstance(handCards, list) else None)

            tie_smaller = False
            if best is not None and total == best_value:
                try:
                    card_cmp = rank
                    if typ in ('Bomb', 'StraightFlush'):
                        card_cmp = len(cards)
                    best_card_cmp = best["rank"]
                    if best["type"] in ('Bomb', 'StraightFlush'):
                        best_card_cmp = len(best["action"]) if isinstance(best.get("action"), list) else best["rank"]
                    tie_smaller = CompareRank().Smaller(typ, rank, card_cmp, best, curRank)
                except Exception:
                    tie_smaller = False

            if best is None or total > best_value or (total == best_value and tie_smaller):
                best_value = total
                best_idx = idx
                best = {"action": cards if typ != 'PASS' else 'PASS', "type": typ, "rank": rank, "actIndex": idx}

        return best

    def actBack(self, handCards, curRank):
        bestPlay = []
        maxValue = -100
        for rank in config.cardRanks:
            if (rank!=curRank and rank<='9' and rank>='2'):
                for card in handCards:
                    if (card[1]==rank):
                        action = [card]
                        restCards = CreateActionList().GetRestCards(action, handCards)
                        restValue, restActions = CountValue().HandCardsValue(restCards, 0, curRank)
                        if (restValue>maxValue):
                            maxValue = restValue
                            bestPlay = {"action": action, "type": "back", "rank": rank}
                        #print(card, restValue)
                        break
        return bestPlay

    def GetAdditionalActionList(self, typeList, curRank, fullActionList):
        additionalActionList=[]
        dict = {}
        for action in fullActionList:
            if (action[0] in typeList and ((action[0], action[1]) not in dict.keys())):
                for card in action[2]:
                    if card == 'H'+curRank:
                        additionalActionList.append(action)
                        dict[(action[0], action[1])] = 1
                        break
        return additionalActionList

    def FreePlay(self, handCards, curRank, fullActionList = None):
        # If server provided a concrete actionList, pick directly from it (covers 参谋补牌的所有合法动作)。
        if isinstance(fullActionList, list) and len(fullActionList) > 0 and isinstance(fullActionList[0], list):
            try:
                handValue, handActions = CountValue().HandCardsValue(handCards, 0, curRank)
                Strategy.SetRole(handValue, handActions, curRank)
                Strategy.makeReviseValues()
            except Exception:
                # Strategy 状态可能尚未完全初始化；不影响直接从 actionList 选动作。
                pass
            pick = self._best_from_full_action_list(handCards, curRank, fullActionList, mode="free", formerAction=None)
            if pick is not None:
                return pick

        handValue, handActions = CountValue().HandCardsValue(handCards, 0, curRank)
        #print(handActions)
        #Strategy.SetBeginning(0)
        Strategy.SetRole(handValue, handActions, curRank)
        Strategy.makeReviseValues()
        #print(Strategy.recordPlayerActions)
        additionalActionList = self.GetAdditionalActionList(["ThreePair", "Straight"], curRank, fullActionList)
        #print("additionalActionList", additionalActionList)
        #beginning
        bestPlay = {}
        if (len(handCards)>=15 or Strategy.roundStage != 'ending'):
            minValue = 100
            for action in handActions:
                actionValue = CountValue().ActionValue(action, action['type'], action['rank'], curRank) - Strategy.freeActionRV[action['type']] \
                        - Strategy.freeActionRV[(action['type'],action['rank'])]
                #print(action, actionValue)
                if actionValue < minValue:
                    minValue = actionValue
                    bestPlay = action
        #print(Strategy.freeActionRV[('Pair','Q')])
        else:
            maxValue = -100
            actionList = CreateActionList().CreateList(handCards)
            for i in range(0, len(config.cardTypes)):
                type = config.cardTypes[i]
                #if (type == 'StraightFlush'): continue
                for rank1 in actionList[type]:
                    for card in actionList[type][rank1]:
                        color = None
                        rank = rank1  # to distinguish StraightFlush from others
                        if (type == 'StraightFlush'):
                            rank = rank1[1]
                            color = rank1[0]
                        #print("Free play trying type, rank, card:", type, rank, card)
                        action = CreateActionList().GetAction(type, rank, card, handCards, color)
                        restCards = CreateActionList().GetRestCards(action, handCards)
                        restValue, restActions = CountValue().HandCardsValue(restCards, 0, curRank)
                        thisHandValue = CountValue().ActionValue(action, type, rank, curRank)
                        thisHandValue += Strategy.freeActionRV[type]
                        if (type, rank) in Strategy.freeActionRV.keys():
                            thisHandValue += Strategy.freeActionRV[(type, rank)]
                        # print(Strategy.actionValueRevise)
                        # print(rank, card, thisHandValue, restValue)
                        if (thisHandValue < 0): thisHandValue = 0
                        total = thisHandValue + restValue - self._adviser_opportunity_cost(action, curRank, hand_size=len(handCards) if isinstance(handCards, list) else None)
                        if (total > maxValue or (total == maxValue and \
                            (bestPlay == [] or CompareRank().Smaller(type, rank, card, bestPlay, curRank)))):
                            maxValue = total
                            bestPlay = {"action": action, "type": type, "rank": rank}

            #try additional list
            for action in additionalActionList:
                type = action[0]
                rank = action[1]
                card = rank
                if type == 'Bomb':
                    card = len(action[2])

                restCards = CreateActionList().GetRestCards(action[2], handCards)
                restValue, restActions = CountValue().HandCardsValue(restCards, 0, curRank)
                restValue += Strategy.handRV[type]
                thisHandValue = CountValue().ActionValue(self._eval_cards_from_full_action(action), type, rank, curRank)
                thisHandValue += Strategy.freeActionRV[type]
                if (type, rank) in Strategy.freeActionRV.keys():
                    thisHandValue += Strategy.freeActionRV[(type, rank)]
                # print(Strategy.actionValueRevise)
                # print(rank, card, thisHandValue, restValue)
                if (thisHandValue < 0): thisHandValue = 0
                total = thisHandValue + restValue - self._adviser_opportunity_cost(action[2], curRank, hand_size=len(handCards) if isinstance(handCards, list) else None)
                if (total > maxValue or (total == maxValue and
                                (bestPlay == [] or CompareRank().Smaller(type, rank, card, bestPlay, curRank)))):
                    maxValue = total
                    bestPlay = {"action": action[2], "type": type, "rank": rank}

        #print("bestplay:",bestPlay, "handValue", handValue)
        return bestPlay

    def RestrictedPlay(self, handCards, formerAction, curRank, fullActionList = None):
        # If server provided a concrete actionList (already legal & includes 参谋补牌 variants), pick directly.
        if isinstance(fullActionList, list) and len(fullActionList) > 0 and isinstance(fullActionList[0], list):
            try:
                maxValue, restActions = CountValue().HandCardsValue(handCards, 0, curRank)
                Strategy.SetRole(maxValue, restActions, curRank)
                Strategy.makeReviseValues()
            except Exception:
                pass
            pick = self._best_from_full_action_list(handCards, curRank, fullActionList, mode="restricted", formerAction=formerAction)
            if pick is not None:
                return pick

        actionList = CreateActionList().CreateList(handCards)

        additionalActionList = self.GetAdditionalActionList(["Bomb", "StraightFlush", "ThreePair", "Straight"], curRank,
                                                            fullActionList)
        #print("additionalActionList:", additionalActionList)

        bestPlay = []
        maxValue, restActions = CountValue().HandCardsValue(handCards, 0, curRank)
        Strategy.SetRole(maxValue, restActions, curRank)
        Strategy.makeReviseValues()
        maxValue += Strategy.restrictedActionRV["PASS"]

        #print(maxValue)
        toc = time.time()
        #print(toc - tic)

        for i in range(0, len(config.cardTypes)):
            type = config.cardTypes[i]
            #print(type, formerAction["type"])
            #if (type == 'StraightFlush'): continue
            if (type != 'Bomb' and type != 'StraightFlush' and type != formerAction["type"]): continue
            for rank1 in actionList[type]:
                for card in actionList[type][rank1]:
                    color = None
                    rank = rank1  # to distinguish StraightFlush from others
                    if (type == 'StraightFlush'):
                        rank = rank1[1]
                        color = rank1[0]
                    #print("Restricted play trying rank, card:", type, rank, card)
                    if (CompareRank().Larger(type, rank, card, formerAction, curRank)):
                        action = CreateActionList().GetAction(type, rank, card, handCards, color)
                        restCards = CreateActionList().GetRestCards(action, handCards)
                        restValue, restActions = CountValue().HandCardsValue(restCards, 0, curRank)
                        #restValue += Strategy.handRV[type]
                        thisHandValue = CountValue().ActionValue(action, type, rank, curRank)
                        thisHandValue += Strategy.restrictedActionRV[type]
                        if (type, rank) in Strategy.restrictedActionRV.keys():
                            thisHandValue += Strategy.restrictedActionRV[(type, rank)]
                        #print(Strategy.actionValueRevise)
                        #print(rank, card, thisHandValue, restValue)
                        if (thisHandValue < 0): thisHandValue = 0
                        total = thisHandValue + restValue - self._adviser_opportunity_cost(action, curRank, hand_size=len(handCards) if isinstance(handCards, list) else None)
                        if (total > maxValue or (total == maxValue and \
                        (bestPlay==[] or CompareRank().Smaller(type, rank, card, bestPlay, curRank)))):
                            maxValue = total
                            bestPlay = {"action": action, "type": type, "rank": rank}

        #try additional list
        for action in additionalActionList:
            type = action[0]
            rank = action[1]
            card = rank
            if type == 'Bomb':
                card = len(action[2])
            if (CompareRank().Larger(type, rank, card, formerAction, curRank)):
                restCards = CreateActionList().GetRestCards(action[2], handCards)
                restValue, restActions = CountValue().HandCardsValue(restCards, 0, curRank)
                #restValue += Strategy.handRV[type]
                thisHandValue = CountValue().ActionValue(self._eval_cards_from_full_action(action), type, rank, curRank)
                thisHandValue += Strategy.restrictedActionRV[type]
                if (type, rank) in Strategy.restrictedActionRV.keys():
                    thisHandValue += Strategy.restrictedActionRV[(type, rank)]
                # print(Strategy.actionValueRevise)
                # print(rank, card, thisHandValue, restValue)
                if (thisHandValue < 0): thisHandValue = 0
                total = thisHandValue + restValue - self._adviser_opportunity_cost(action[2], curRank, hand_size=len(handCards) if isinstance(handCards, list) else None)
                if (total > maxValue or (total == maxValue and
                                                (bestPlay == [] or CompareRank().Smaller(type, rank, card, bestPlay, curRank)))):
                    maxValue = total
                    bestPlay = {"action": action[2], "type": type, "rank": rank}

        if (bestPlay==[]):
            bestPlay = {'action': 'PASS', 'type': 'PASS', 'rank': 'PASS'}
        #print("bestplay:", bestPlay, "maxvalue", maxValue)
        return bestPlay

    def Play(self, handCards, curRank):
        self.FreePlay(handCards, curRank)


#hand_cards = [[1, '3'], [3, '3'], [2, '5'], [3, '5'], [0, '6'], [2, '6'], [0, '7'], [2, '7'], [2, '7'], [0, '7'], [1, '8'], [2, '8'], [3, '9'], [1, '10'], [2, 'J'], [3, 'J'], [1, 'Q'], [2, 'Q'], [3, 'K'], [0, 'K'], [0, 'K'], [2, 'A'], [0, 'A'], [1, '2'], [0, '2'], [0, 'JOKER'], [0, 'JOKER']]
#cards = ['H2', 'C2', 'D3', 'S4', 'H4', 'C5', 'C5', 'S6', 'D6', 'H8', 'D8', 'S9', 'H9', 'ST', 'CT', 'SJ', 'CJ', 'DJ', 'HQ', 'DQ', 'SK', 'SA', 'HA', 'H7', 'D7', 'SB', 'HR']
#formerAction = {"action": ['H5', 'C5'], "type": 'Pair', 'rank': '5'}
#print(PlayCard().FreePlay(cards,'2'))
#print(PlayCard().RestrictedPlay(hand_cards, formerAction))

'''tic = time.time()

cards =  ['H3', 'C3', 'S4', 'C4', 'C5', 'S2', 'S2', 'H2', 'D2', 'D2']
Strategy.SetBeginning(0, cards)
Strategy.curRank = '2'
Strategy.restHandsCount=[10, 27, 15, 15]
Strategy.roundStage = 'ending'

#Strategy.UpdatePlay(1, ['Straight', '3', ['S3', 'C4', 'D5','D6','D7']], 1, ['Straight', '3', ['S3', 'C4', 'D5','D6','D7']])
fullActionList = [['ThreePair', '2', ['S2', 'D2', 'H3', 'H2', 'S4', 'C4']], ['ThreePair', '3', ['H3', 'C3', 'S4', 'C4', 'C5', 'H2']], ['Straight', 'A', ['H2', 'S2', 'H3', 'S4', 'C5']], ['Straight', '2', ['S2', 'H3', 'S4', 'C5', 'H2']]]
#print(PlayCard().RestrictedPlay(cards,{'action':['S3', 'C4', 'D5','D6','D7'], 'type': 'Striaight', 'rank': '3'}, 'K', fullActionList))
#print("RV:", Strategy.restrictedActionRV[('Single','R')])
#Strategy.UpdatePlay(-1, None, -1, None)
print(PlayCard().FreePlay(cards, '2', fullActionList))

toc = time.time()
print(toc-tic)'''

#cards = ['H2', 'C2', 'D3', 'S4', 'H4', 'C5', 'C5', 'S6', 'D6', 'H8', 'D8', 'S9', 'H9', 'ST', 'CT', 'SJ', 'CJ', 'DJ', 'HQ', 'DQ', 'SK', 'SA', 'HA', 'H7', 'D7', 'SB', 'HR']
#print(PlayCard().actBack(cards, '2'))

#print(not CompareRank().Larger('Pair', 'A', 'A', {'action': ['S6', 'H6', 'C6', 'C6'], 'type': 'Bomb', 'rank': '6'}, '3'))