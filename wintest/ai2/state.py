# -*- coding: utf-8 -*-
# @Time       : 2020/10/1 19:21
# @Author     : Duofeng Wu
# @File       : gd_handle.py
# @Description: 自动解析掼蛋所发送来的JSON数据

from PlayCard import PlayCard
from strategy import Strategy


def _fmt_card(card: str) -> str:
    suit_symbol = {"C": "♣", "D": "♦", "S": "♠", "H": "♥"}
    if card in {"SB", "HR"}:
        return "大JOKER" if card == "HR" else "小JOKER"
    if not card:
        return ""
    s = card[0]
    r = card[-1]
    if r == "T":
        r = "10"
    return f"{suit_symbol.get(s, s)}{r}"


def _fmt_action(action) -> str:
    # action is expected like ['Pair','B',['SB','SB']] or ['PASS','PASS',[]]
    if not action:
        return ""
    try:
        typ = action[0]
    except Exception:
        return str(action)

    type_ch = {
        "Single": "单张",
        "Pair": "对子",
        "Trips": "三张",
        "ThreePair": "三连对",
        "ThreeWithTwo": "三带二",
        "TwoTrips": "钢板",
        "Straight": "顺子",
        "StraightFlush": "同花顺",
        "Bomb": "炸弹",
        "PASS": "过",
    }.get(typ, str(typ))

    if typ == "PASS":
        return type_ch

    try:
        cards = action[2]
    except Exception:
        cards = []
    if cards == "PASS":
        return type_ch
    # Compress duplicates as ×n to reduce noise.
    try:
        from collections import Counter

        cnt = Counter(cards) if isinstance(cards, list) else Counter()
        ordered = []
        seen = set()
        if isinstance(cards, list):
            for c in cards:
                if c in seen:
                    continue
                seen.add(c)
                ordered.append(c)
        parts = []
        for c in ordered:
            n = int(cnt.get(c, 0))
            if n <= 1:
                parts.append(_fmt_card(c))
            else:
                parts.append(f"{_fmt_card(c)}×{n}")
        cards_s = " ".join(parts)
    except Exception:
        cards_s = str(cards)
    return f"{type_ch}: {cards_s}".strip()

class State(object):

    def __init__(self):

        """
        每个实例的保护属性对应JSON中的字段值，私有属性表示根据不同type和stage进行不同解析。
        type:          表示消息类型。可取值包括notify和act。notify表示通知类型，act表示动作类型（即收到该类型的消息时需要做出动作）
        stage:         表示游戏阶段。可取值包括beginning, play, tribute, anti-tribute, back, episodeOver, gameOver
                        分别对应开始阶段、出牌阶段、进贡阶段、抗贡阶段、还贡阶段、结束阶段
        myPos:         表示自己的座位号
        publicInfo:    表示游戏中玩家公共信息
        actionList:    表示可行的动作列表
        curAction:     表示某玩家做出的动作
        curPos:        表示做出当前动作的玩家的座位号
        greaterPos:    表示最大动作的玩家的座位号
        greaterAction: 表示最大到你工作
        handCards:     表示手牌
        oppoRank:      表示对手等级
        curRank:       表示当前游戏在使用的等级None
        selfRank:      表示我方等级
        antiNum:       表示抗贡人数
        antiPos:       表示抗贡玩家（们）的 座位号
        result:        表示进贡或者还贡的结果
        order:         表示完牌的次序
        curTimes:      当前的对局次数
        settingTimes   设定的对局次数
        victoryNum     表示达到设定场次时的最终结果（哪一方赢得多）
        parse_func:    表示用于解析的函数
        """

        self._type = None
        self._stage = None
        self._myPos = None
        self._publicInfo = None
        self._actionList = None
        self._curAction = None
        self._curPos = None
        self._greaterPos = None
        self._greaterAction = None
        self._handCards = None
        self._oppoRank = None
        self._curRank = None
        self._selfRank = None
        self._antiNum = None
        self._antiPos = None
        self._result = None
        self._order = None
        self._curTimes = None
        self._settingTimes = None
        self._victoryNum = None
        self._draws = None
        self._restCards = None

        self.retValue = None

        # TODO: 选手可根据(stage, type)自行定义处理的函数
        self.__parse_func = {
            ("beginning", "notify"): self.notify_begin,
            ("play", "notify"): self.notify_play,
            ("tribute", "notify"): self.notify_tribute,
            ("anti-tribute", "notify"): self.notify_anti,
            ("back", "notify"): self.notify_back,
            ("gameOver", "notify"): self.notify_game_over,
            ("episodeOver", "notify"): self.notify_episode_over,
            ("gameResult", "notify"): self.notify_game_result,

            ("play", "act"): self.act_play,
            ("tribute", "act"): self.act_tribute,
            ("back", "act"): self.act_back,
        }

    def parse(self, msg):
        assert type(msg) == dict
        for key, value in msg.items():
            setattr(self, "_{}".format(key), value)
        try:
            self.__parse_func[(self._stage, self._type)]()
            self._stage = None
            self._type = None
        except KeyError:
            raise KeyError

    def notify_begin(self):
        """
        游戏开始阶段，告知每位玩家的手牌情况
        形如下所表示的JSON:
        {
            "type": "notify",
            "stage": "beginning",
            "handCard": ['S2', 'S2'],
            "myPos": 1,
        }
        请仅在对应的JSON格式下访问对应的实例属性，若此时访问其他属性则很有可能是之前处理时未更新的实例属性，不具有准确性。
        """
        # TODO: 选手可自行做出其他处理
        Strategy.SetBeginning(self._myPos, self._handCards)
        self._greaterPos = -1
        self._greaterAction = None
        try:
            n = len(self._handCards) if self._handCards is not None else 0
        except Exception:
            n = 0
        print("游戏开始, 我是{}号位（手牌 {} 张）".format(self._myPos, n))

    def notify_play(self):
        """
        出牌阶段，用于通知其他玩家做出了什么动作
        形如下所表示的JSON格式:
        {
            "type": "notify",
            "stage": "play",
            "curPos": 1,
            "curAction": {"rank": '2', "type": Single, "actions": ['S2']},
            "greaterPos": 1,
            "greaterAction": {"rank": '2', "type": Single, "actions": ['S2']}
        }
        请仅在对应的JSON格式下访问对应的实例属性，若此时访问其他属性则很有可能是之前处理时未更新的实例属性，不具有准确性。
        """
        # TODO: 选手可自行做出其他处理
        Strategy.UpdatePlay(self._curPos, self._curAction, self._greaterPos, self._greaterAction)
        cur_s = _fmt_action(self._curAction)
        greater_s = _fmt_action(self._greaterAction)
        # If the current action is also the current maximum, avoid repeating it.
        if self._curPos == self._greaterPos and self._curAction == self._greaterAction:
            print("{}号位打出{}（当前最大）".format(self._curPos, cur_s))
        else:
            print("{}号位打出{}，当前最大 {}号位 {}".format(self._curPos, cur_s, self._greaterPos, greater_s))

    def notify_tribute(self):
        """
        进贡阶段，用于通知所有玩家进贡者（们）都进贡了什么牌
        形如下所表示的JSON格式:
        {
            "type": "notify",
            "stage": "tribute",
            "result": [[0, 3, 'S2']] 或 [[0, 3, 'S2'], [2, 1, 'S2']]
        }
        请仅在对应的JSON格式下访问对应的实例属性，若此时访问其他属性则很有可能是之前处理时未更新的实例属性，不具有准确性。
        """
        # TODO: 选手可自行做出其他处理
        for tribute_result in self._result:
            tribute_pos, receive_tribute_pos, card = tribute_result
            print("{}号位向{}号位进贡{}".format(tribute_pos, receive_tribute_pos, card))

    def notify_anti(self):
        """
        抗贡阶段，用于通知所有玩家，有人抗贡。其中antiNums的取值与antiPos数组的长度所对应
        形如下所表示的JSON格式:
        {
            "type": "notify",
            "stage": "anti-tribute",
            "antiNums": 2,
            "antiPos": [0, 2]
        }
        请仅在对应的JSON格式下访问对应的实例属性，若此时访问其他属性则很有可能是之前处理时未更新的实例属性，不具有准确性。
        """
        # TODO: 选手可自行做出其他处理
        for pos in self._antiPos:
            print("{}号位玩家抗贡".format(pos))

    def notify_back(self):
        """
        还贡阶段，用于通知所有玩家还贡者（们）都还贡了什么牌
        形如下所表示的JSON格式:
        {
            "type": "notify",
            "stage": "back",
            "result": [[3, 0, 'S2']] 或 [[3, 0, 'S2'], [1, 2, 'S2']]
        }
        请仅在对应的JSON格式下访问对应的实例属性，若此时访问其他属性则很有可能是之前处理时未更新的实例属性，不具有准确性。
        """
        # TODO: 选手可自行做出其他处理

        for back_result in self._result:
            back_pos, receive_back_pos, card = back_result
            print("{}号位向{}号位还贡{}".format(back_pos, receive_back_pos, card))

    def notify_episode_over(self):
        """
        小局结束阶段，用于通知所有玩家小局结束
        形如下所表示的JSON格式:
        {
            "type": "notify",
            "stage": "episodeOver",
            "order": [0, 1, 2, 3]
            “curRank": 1,
            "restCards": [[pos, handcards], ...]
        }
        请仅在对应的JSON格式下访问对应的实例属性，若此时访问其他属性则很有可能是之前处理时未更新的实例属性，不具有准确性。
        """
        # TODO: 选手可自行做出其他处理
        print("对局结束，完牌次序为{}，结束时所打的等级为{}".format(self._order, self._curRank))
        Strategy.Clear()
        for rest in self._restCards:
            rest_pos, rest_cards = rest
            print("{}号位剩余卡牌{}".format(rest_pos, rest_cards))

    def notify_game_over(self):
        """
        到达指定游戏次数游戏结束，用于通知所有玩家游戏结束
        形如下所表示的JSON格式:
        {
            "type": "notify",
            "stage": "gameOver",
            "curTimes": 1
            “settingTimes": 1,
        }
        请仅在对应的JSON格式下访问对应的实例属性，若此时访问其他属性则很有可能是之前处理时未更新的实例属性，不具有准确性。
        """
        # TODO: 选手可自行做出其他处理
        print("当前训练次数为{}, 设定的游戏次数为{}".format(self._curTimes, self._settingTimes))

    def notify_game_result(self):
        """
        到达指定游戏次数游戏结束，用于通知所有玩家游戏结束
        形如下所表示的JSON格式。该JSON表示经过2场对局后游戏结束，其中0号位玩家和2号位玩家胜利次数位2。
        {
            "type": "notify",
            "stage": "gameResult",
            "final": [2, 0, 2, 0]
        }
        请仅在对应的JSON格式下访问对应的实例属性，若此时访问其他属性则很有可能是之前处理时未更新的实例属性，不具有准确性。
        """
        # TODO: 选手可自行做出其他处理
        print("达到设定场次, 其中0号位胜利{}次，1号位胜利{}次，2号位胜利{}次，3号位胜利{}次".format(*self._victoryNum))
        print("其中平局次数：0号位平局{}次，1号位平局{}次，2号位平局{}次，3号位平局{}次".format(*self._draws))

    def act_play(self):
        """
        出牌阶段，用于通知该玩家做出动作
        形如下所表示的JSON格式:
        {
            "type": "act",
            "handCards": [C3, D3, D3, H5, C5, D5, S6, D6 ... ] ,
            "publicInfo": [
                {'rest': 27, 'playArea': None},
                {'rest': 27, 'playArea': None},
                {'rest': 27, 'playArea': None},
                {'rest': 27, 'playArea': None}
            ],
            "selfRank": ‘2’,
            "oppoRank": ‘2’,
            "curRank": ‘2’,
            "stage": "play",
            "curPos": -1,
            "curAction": None,
            "greaterAction": -1,
            "greaterPos": None,
            "actionList": {"Single" : {'2': ['S2', 'S2' ...]} ...}
        }
        请仅在对应的JSON格式下访问对应的实例属性，若此时访问其他属性则很有可能是之前处理时未更新的实例属性，不具有准确性。
        """
        # TODO: 选手可自行做出其他处理
        try:
            act_cnt = len(self._actionList) if self._actionList is not None else 0
        except Exception:
            act_cnt = 0
        # On lead (no greater action), don't print confusing None/-1/PASS info.
        if Strategy.greaterPos == -1 or self._greaterPos == -1:
            print("轮到我出牌（首出），可选动作 {} 个".format(act_cnt))
        else:
            print(
                "轮到我出牌：当前最大 {}号位 {}；可选动作 {} 个".format(
                    self._greaterPos,
                    _fmt_action(self._greaterAction),
                    act_cnt,
                )
            )

        if (Strategy.greaterPos == -1 or self._greaterPos==-1):
            Strategy.UpdateCurRank(self._curRank)
            self.retValue = PlayCard().FreePlay(self._handCards, self._curRank, self._actionList)
        else:
            Strategy.UpdateCurRank(self._curRank)
            formerAction={"type":self._greaterAction[0], "rank":self._greaterAction[1], "action":self._greaterAction[2]}
            self.retValue = PlayCard().RestrictedPlay(self._handCards, formerAction, self._curRank, self._actionList)

        #print("我方等级：{}， 对方等级：{}， 当前等级{}".format(self._selfRank, self._oppoRank, self._curRank))


    def act_tribute(self):
        """
        进贡阶段，用于该玩家进贡
        形如下所表示的JSON格式:
        {
            "type": "act",
            "handCards": [C3, D3, D3, H5, C5, D5, S6, D6 ... ] ,
            "publicInfo": [
                {'rest': 27, 'playArea': None},
                {'rest': 27, 'playArea': None},
                {'rest': 27, 'playArea': None},
                {'rest': 27, 'playArea': None}
            ],
            "selfRank": ‘2’,
            "oppoRank": ‘3’,
            "curRank": ‘3’,
            "stage": "tribute",
            "curPos": -1,
            "curAction": None,
            "greaterAction": -1,
            "greaterPos": None,
            "actionList": {"tribute": ["S3"]}
        }
        请仅在对应的JSON格式下访问对应的实例属性，若此时访问其他属性则很有可能是之前处理时未更新的实例属性，不具有准确性。
        """
        # TODO: 选手可自行做出其他处理
        print("我方等级：{}， 对方等级：{}， 当前等级{}".format(self._selfRank, self._oppoRank, self._curRank))
        print("轮到自己进贡，可进贡的牌有: ")

    def act_back(self):
        """
        还贡阶段，用于该玩家进贡
        形如下所表示的JSON格式:
        {
            "type": "act",
            "handCards": [C3, D3, D3, H5, C5, D5, S6, D6 ... ] ,
            "publicInfo": [
                {'rest': 27, 'playArea': None},
                {'rest': 27, 'playArea': None},
                {'rest': 27, 'playArea': None},
                {'rest': 27, 'playArea': None}
            ],
            "selfRank": ‘3’,
            "oppoRank": ‘2’,
            "curRank": ‘3’,
            "stage": "back",
            "curPos": -1,
            "curAction": None,
            "greaterAction": -1,
            "greaterPos": None,
            "actionList": [['back', 'back', ['S2']], ['back', 'back', ['H2']]
        }
        请仅在对应的JSON格式下访问对应的实例属性，若此时访问其他属性则很有可能是之前处理时未更新的实例属性，不具有准确性。
        """
        # TODO: 选手可自行做出其他处理

        self.retValue = PlayCard().actBack(self._handCards, self._curRank)

        print("我方等级：{}， 对方等级：{}， 当前等级{}".format(self._selfRank, self._oppoRank, self._curRank))
        print("轮到自己还贡，可还贡的牌有:")
