import argparse
import itertools
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

# Ensure repo root on sys.path so we can import mini_danserver.py from project root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import rule helpers from mini_danserver (guarded by __main__).
import mini_danserver as mds


SUIT_MAP = {
    "♠": "S",
    "♥": "H",
    "♣": "C",
    "♦": "D",
}

CN_TYPE_MAP = {
    "过": "PASS",
    "单张": "Single",
    "对子": "Pair",
    "三张": "Trips",
    "三带二": "ThreeWithTwo",
    "顺子": "Straight",
    "同花顺": "StraightFlush",
    "三连对": "ThreePair",
    "钢板": "TwoTrips",
    "炸弹": "Bomb",
}


@dataclass
class ParsedPlay:
    seat: int
    cn_type: str
    cards: List[str]
    raw: str


_LINE_PLAY = re.compile(r"^(?P<seat>[0-3])号位打出(?P<body>.+)$")


def _parse_cur_rank(text: str) -> Optional[str]:
    m = re.search(r"curRank.*：(?P<rk>10|[2-9TJQKA])", text)
    if not m:
        return None
    rk = m.group("rk")
    return "T" if rk == "10" else rk


def _expand_token(tok: str) -> List[str]:
    tok = tok.strip()
    if not tok:
        return []

    # multiplier
    mult = 1
    if "×" in tok:
        base, n = tok.split("×", 1)
        tok = base
        try:
            mult = int(n)
        except Exception:
            mult = 1

    if tok in {"大JOKER", "大Joker", "大joker"}:
        return ["HR"] * mult
    if tok in {"小JOKER", "小Joker", "小joker"}:
        return ["SB"] * mult

    # suit cards like ♦10 or ♠A
    suit_symbol = tok[0]
    if suit_symbol not in SUIT_MAP:
        return []

    suit = SUIT_MAP[suit_symbol]
    rank = tok[1:]
    rank = rank.upper()
    if rank == "10":
        rank = "T"
    if rank == "JOKER":
        # tolerate "♠JOKER" etc (rare)
        return ["HR"] * mult

    return [f"{suit}{rank}"] * mult


def _parse_cards_part(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    # split by spaces
    toks = [t for t in s.replace("，", " ").replace(",", " ").split() if t]
    cards: List[str] = []
    for t in toks:
        cards.extend(_expand_token(t))
    return cards


def parse_plays(text: str) -> List[ParsedPlay]:
    plays: List[ParsedPlay] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LINE_PLAY.match(line)
        if not m:
            continue
        seat = int(m.group("seat"))
        body = m.group("body")

        # PASS
        if body.startswith("过"):
            plays.append(ParsedPlay(seat=seat, cn_type="过", cards=[], raw=line))
            continue

        # format: <类型>: <cards>(...)
        # e.g. "顺子: ♦4 ♠5 ♥6 ♥3 ♦8（当前最大）"
        mm = re.match(r"(?P<typ>[^:：]+)[:：]\s*(?P<cards>.*)$", body)
        if not mm:
            continue
        cn_type = mm.group("typ").strip()
        cards_part = mm.group("cards")
        # strip trailing parenthesis content
        cards_part = re.split(r"（|\(", cards_part)[0].strip()
        cards = _parse_cards_part(cards_part)
        plays.append(ParsedPlay(seat=seat, cn_type=cn_type, cards=cards, raw=line))

    return plays


def _exists_action_with_declared(cards: List[str], cur_rank: str, declared_type: str) -> bool:
    adviser = mds._adviser_card(cur_rank)
    adv_idx = [i for i, c in enumerate(cards) if c == adviser]
    if not adv_idx:
        try:
            act = mds.classify_action(cards, cur_rank)
        except Exception:
            return False
        return act[0] == declared_type

    # Enumerate substitutions for adviser cards (manageable: 52^w, w usually small)
    for cards_as in itertools.product(mds.ALL_NON_JOKER_CARDS, repeat=len(adv_idx)):
        virtual = list(cards)
        for j, c_as in enumerate(cards_as):
            virtual[adv_idx[j]] = str(c_as)
        try:
            act_v = mds.classify_action(virtual, cur_rank)
        except Exception:
            continue
        if act_v[0] == declared_type:
            return True
    return False


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Check a mini_danserver game log snippet for basic rule consistency")
    ap.add_argument("log", help="path to the text log")
    ap.add_argument("--curRank", default=None, help="override curRank (2-9,T,J,Q,K,A). 10 use T")
    args = ap.parse_args(list(argv) if argv is not None else None)

    with open(args.log, "r", encoding="utf-8") as f:
        text = f.read()

    cur_rank = (args.curRank or _parse_cur_rank(text) or "2").upper()
    if cur_rank == "10":
        cur_rank = "T"

    plays = parse_plays(text)
    if not plays:
        print("No plays parsed.")
        return 2

    inv = Counter()
    inv_violations: List[Tuple[str, int]] = []
    type_violations: List[str] = []

    for i, p in enumerate(plays, 1):
        # inventory check: two decks => each exact card face <= 2
        inv.update(p.cards)
        for c, n in inv.items():
            if n > 2:
                inv_violations.append((c, n))

        declared = CN_TYPE_MAP.get(p.cn_type)
        if declared is None:
            # unknown type string; skip
            continue
        if declared == "PASS":
            continue
        ok = _exists_action_with_declared(p.cards, cur_rank, declared)
        if not ok:
            type_violations.append(f"#{i} {p.raw}")

    print(json.dumps(
        {
            "curRank": cur_rank,
            "plays": len(plays),
            "inventory_violation": len(inv_violations) > 0,
            "type_violation": len(type_violations) > 0,
            "inventory_over_2": inv_violations[:10],
            "first_type_violations": type_violations[:5],
        },
        ensure_ascii=False,
        indent=2,
    ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
