import argparse
import asyncio
import random
import os
import sys
import subprocess
import glob
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import websockets
from websockets.server import ServerConnection

# 复用 torch 客户端使用的牌面编码全集。
# 注意：本仓库的 wintest/ 不是 Python 包；这里按已有脚本做法，
# 通过 sys.path 引入 wintest/torch/util.py。
_THIS_DIR = os.path.dirname(__file__)
_TORCH_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "torch"))
if _TORCH_DIR not in sys.path:
    sys.path.insert(0, _TORCH_DIR)

from util import CardToNum, combine_handcards  # type: ignore


Card = str
Action = List  # [type, key, cards]


class _GuiFallbackToTerminalThisTurn(Exception):
    pass
def format_card_for_human(card: Card) -> str:
    suit_symbol = {"C": "♣", "D": "♦", "S": "♠", "H": "♥"}
    rank_display = {"T": "10"}
    if card in {"SB", "HR"}:
        return "大JOKER" if card == "HR" else "小JOKER"
    s = card[0]
    r = card[-1]
    return f"{suit_symbol.get(s, s)}{rank_display.get(r, r)}"

def format_cards_for_human(cards: List[Card]) -> str:
    # 压缩重复牌：每种牌只显示一次，并用 ×n 表示数量。
    if not cards:
        return ""
    cnt = Counter(cards)
    ordered_unique: List[Card] = []
    seen = set()
    for c in cards:
        if c in seen:
            continue
        seen.add(c)
        ordered_unique.append(c)
    parts = []
    for c in ordered_unique:
        n = int(cnt.get(c, 0))
        if n <= 1:
            parts.append(format_card_for_human(c))
        else:
            parts.append(f"{format_card_for_human(c)}×{n}")
    return " ".join(parts)

def format_action_for_human(action: Optional[Action]) -> str:
    if not action or action[0] == "PASS":
        return "过"

    type_ch = {
        "Single": "单张",
        "Pair": "对子",
        "Trips": "三张",
        "ThreePair": "三连对",
        "ThreeWithTwo": "三带二",
        "TwoTrips": "钢板",
        "Straight": "顺子",
        "Bomb": "炸弹",
    }.get(action[0], action[0])

    return f"{type_ch}: {format_cards_for_human(action[2])}".strip()


class _PersistentHumanActionPicker:
    def __init__(self, *, cur_rank: str):
        import tkinter as tk

        self._tk = tk

        self.suit_symbol = {"C": "♣", "D": "♦", "S": "♠", "H": "♥"}
        self.suit_fg = {"C": "black", "S": "black", "D": "red", "H": "red"}
        self.rank_display = {"T": "10"}

        rank_desc = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
        self.cur_rank = cur_rank
        self.rank_chars = [cur_rank] + [r for r in rank_desc if r != cur_rank]
        # 花色顺序要求：从下到上是 ♦ ♣ ♥ ♠，因此从上到下是 ♠ ♥ ♣ ♦。
        self.suit_rows = ["S", "H", "C", "D"]  # 从上到下

        self._mode: str = "idle"  # 模式：idle | ai_hand | human（空闲 | 选 AI 手牌 | 人类出牌）

        self.seat = 1
        self.remaining = 27
        self.ai_hand: List[Card] = []
        self.ai_counter = Counter()
        self.played_cards = Counter()
        self.greater_action: Optional[Action] = None

        self.selected: List[Card] = []
        self.selected_count = Counter()

        self._result_action: Optional[Action] = None
        self._result_ai_hand: Optional[str] = None
        self._closed_for_turn = False

        self._max_w = 0
        self._max_h = 0
        self._max_preview_w = 0
        self._max_preview_h = 0

        root = tk.Tk()
        self.root = root
        root.title("选牌")
        root.resizable(True, True)

        self.info_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.info_var).pack(padx=8, pady=(8, 2), anchor="w")

        self.context_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.context_var).pack(padx=8, pady=(0, 6), anchor="w")

        # 预览区：使用固定尺寸容器，清空预览时不缩回，避免把下面的牌按钮网格往上顶。
        preview_container = tk.Frame(root)
        preview_container.pack(padx=8, pady=(0, 8), fill="x", anchor="w")
        try:
            preview_container.pack_propagate(False)
        except Exception:
            pass
        self.preview_container = preview_container

        preview_frame = tk.Frame(preview_container)
        preview_frame.pack(fill="both", anchor="w")
        self.preview_frame = preview_frame
        self.preview_cells: Dict[str, tk.Label] = {}

        tk.Label(preview_frame, text="").grid(row=0, column=0, padx=3, pady=1)
        tk.Label(preview_frame, text="大JOKER", fg="red").grid(row=0, column=1, padx=3, pady=1)
        tk.Label(preview_frame, text="小JOKER", fg="black").grid(row=0, column=2, padx=3, pady=1)
        for col, r in enumerate(self.rank_chars, start=3):
            tk.Label(preview_frame, text=self.rank_display.get(r, r)).grid(row=0, column=col, padx=3, pady=1)

        for row, s in enumerate(self.suit_rows, start=1):
            tk.Label(preview_frame, text=self.suit_symbol[s], fg=self.suit_fg[s]).grid(
                row=row, column=0, padx=3, pady=1
            )
            for col, r in enumerate(self.rank_chars, start=3):
                code = f"{s}{r}"
                lbl = tk.Label(preview_frame, text="", width=6, fg=self.suit_fg[s], justify="center")
                lbl.grid(row=row, column=col, padx=1, pady=1)
                self.preview_cells[code] = lbl

        joker_row = len(self.suit_rows) + 1
        tk.Label(preview_frame, text="JOKER").grid(row=joker_row, column=0, padx=3, pady=1, sticky="e")
        for col, code in [(1, "HR"), (2, "SB")]:
            lbl = tk.Label(preview_frame, text="", width=6, fg=self._card_color(code), justify="center")
            lbl.grid(row=joker_row, column=col, padx=1, pady=1)
            self.preview_cells[code] = lbl

        # 布局生成后，将预览容器锁定为初始请求尺寸；后续只允许增大（不允许变小）。
        try:
            root.update_idletasks()
            self._max_preview_w = max(1, int(self.preview_frame.winfo_reqwidth()))
            self._max_preview_h = max(1, int(self.preview_frame.winfo_reqheight()))
            self.preview_container.configure(width=self._max_preview_w, height=self._max_preview_h)
        except Exception:
            pass

        # 牌按钮区
        frame = tk.Frame(root)
        frame.pack(padx=8, pady=8, fill="both", expand=True)
        self.buttons: Dict[str, tk.Button] = {}

        tk.Label(frame, text="").grid(row=0, column=0, padx=4, pady=2)
        for col, r in enumerate(self.rank_chars, start=1):
            tk.Label(frame, text=self.rank_display.get(r, r)).grid(row=0, column=col, padx=4, pady=2)

        for row, s in enumerate(self.suit_rows, start=1):
            tk.Label(frame, text=self.suit_symbol[s], fg=self.suit_fg[s]).grid(row=row, column=0, padx=4, pady=2)
            for col, r in enumerate(self.rank_chars, start=1):
                code = f"{s}{r}"
                btn = tk.Button(
                    frame,
                    width=8,
                    height=2,
                    command=lambda x=code: self._on_pick(x),
                    fg=self.suit_fg[s],
                )
                btn.grid(row=row, column=col, padx=2, pady=2)
                self.buttons[code] = btn

        joker_row = len(self.suit_rows) + 1
        tk.Label(frame, text="JOKER").grid(row=joker_row, column=0, padx=4, pady=2, sticky="e")
        for col, code in enumerate(["HR", "SB"], start=1):
            btn = tk.Button(
                frame,
                width=8,
                height=2,
                command=lambda x=code: self._on_pick(x),
                fg=self._card_color(code),
            )
            btn.grid(row=joker_row, column=col, padx=2, pady=2)
            self.buttons[code] = btn

        # 操作栏（固定在底部，避免上下跳动）
        action_bar = tk.Frame(root)
        action_bar.pack(side="bottom", padx=8, pady=(0, 8), fill="x")

        self.undo_btn = tk.Button(action_bar, text="撤销", command=self._on_undo)
        self.undo_btn.pack(side="left")
        self.pass_btn = tk.Button(action_bar, text="PASS", command=self._on_pass)
        self.pass_btn.pack(side="left", padx=(8, 0))
        self.confirm_btn = tk.Button(action_bar, text="确认", command=self._on_confirm)
        self.confirm_btn.pack(side="right")

        self._done_var = tk.IntVar(value=0)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 默认显示并就绪。
        self._refresh_ui()

    def _card_color(self, card: str) -> str:
        if card == "SB":
            return "black"
        if card == "HR":
            return "red"
        return self.suit_fg.get(card[0], "black")

    def _card_display_button(self, card: str) -> str:
        if card == "HR":
            return "大JOKER"
        if card == "SB":
            return "小JOKER"
        s = card[0]
        r = card[-1]
        return f"{self.suit_symbol.get(s, s)}{self.rank_display.get(r, r)}"

    def _available_for(self, card: str) -> int:
        return int(2 - self.played_cards.get(card, 0) - self.ai_counter.get(card, 0))

    def _clear_selection(self):
        self.selected.clear()
        self.selected_count.clear()

    def _refresh_ui(self):
        # 标题与上下文
        try:
            if self._mode == "ai_hand":
                self.root.title("选择 AI(seat0) 起手牌（点满 27 张后确认）")
            elif self._mode == "human":
                self.root.title(f"Seat{self.seat} 选择出牌（剩余 {self.remaining} 张）")
            else:
                self.root.title("选牌")
        except Exception:
            pass

        if self._mode == "ai_hand":
            self.info_var.set(f"已选 {len(self.selected)}/27")
            self.context_var.set("关闭窗口=随机；确认后窗口会保留（等待手动出牌）")
        elif self._mode == "human":
            self.info_var.set(f"已选 {len(self.selected)}（最多 {self.remaining}）")
            if self.greater_action is None:
                self.context_var.set("首出：请选择要出的牌（不能 PASS）")
            else:
                self.context_var.set(f"当前牌：{format_action_for_human(self.greater_action)}")
        else:
            self.info_var.set("")
            self.context_var.set("等待手动出牌…")

        # 预览格子
        sel_cnt = Counter(self.selected)
        for code, lbl in self.preview_cells.items():
            cnt = int(sel_cnt.get(code, 0))
            if cnt <= 0:
                lbl.configure(text="")
            else:
                tok = format_card_for_human(code)
                lbl.configure(text="\n".join([tok] * cnt))

        # 按钮状态与剩余数量
        for c, btn in self.buttons.items():
            avail = self._available_for(c)
            cnt = int(self.selected_count.get(c, 0))
            remain = max(0, avail - cnt)
            btn.configure(text=f"{self._card_display_button(c)}\n×{remain}", fg=self._card_color(c))

            if self._mode == "idle":
                btn.configure(state="disabled")
                continue

            if remain <= 0 or len(self.selected) >= self.remaining:
                btn.configure(state="disabled")
            else:
                btn.configure(state="normal")

        # 不同模式下的操作按钮
        if self._mode == "ai_hand":
            self.pass_btn.configure(state="disabled")
            self.undo_btn.configure(state="normal")
            self.confirm_btn.configure(state="normal")
        elif self._mode == "human":
            self.undo_btn.configure(state="normal")
            self.confirm_btn.configure(state="normal")
            if self.greater_action is None:
                self.pass_btn.configure(state="disabled")
            else:
                self.pass_btn.configure(state="normal")
        else:
            self.pass_btn.configure(state="disabled")
            self.undo_btn.configure(state="disabled")
            self.confirm_btn.configure(state="disabled")

        # 窗口一旦变大后不再缩小。
        try:
            self.root.update_idletasks()

            # 将预览容器尺寸锁定为“历史最大请求尺寸”，这样清空选择时预览区不会缩小，
            # 也就不会带动下面的网格布局发生位移。
            try:
                pw = int(self.preview_frame.winfo_reqwidth())
                ph = int(self.preview_frame.winfo_reqheight())
                if pw > self._max_preview_w:
                    self._max_preview_w = pw
                if ph > self._max_preview_h:
                    self._max_preview_h = ph
                if self._max_preview_w > 0 and self._max_preview_h > 0:
                    self.preview_container.configure(width=self._max_preview_w, height=self._max_preview_h)
            except Exception:
                pass

            w = int(self.root.winfo_width())
            h = int(self.root.winfo_height())
            if w > 1 and h > 1:
                if w > self._max_w:
                    self._max_w = w
                if h > self._max_h:
                    self._max_h = h
                if self._max_w > 0 and self._max_h > 0:
                    self.root.minsize(self._max_w, self._max_h)
        except Exception:
            pass

    def _on_pick(self, card: str):
        if len(self.selected) >= self.remaining:
            return
        if self.selected_count.get(card, 0) >= self._available_for(card):
            return
        self.selected.append(card)
        self.selected_count[card] += 1
        self._refresh_ui()

    def _on_undo(self):
        if not self.selected:
            return
        last = self.selected.pop()
        self.selected_count[last] -= 1
        if self.selected_count[last] <= 0:
            del self.selected_count[last]
        self._refresh_ui()

    def _on_pass(self):
        if self._mode != "human":
            return
        if self.greater_action is None:
            return
        self._result_action = ["PASS", "PASS", []]
        self._done_var.set(1)

    def _on_confirm(self):
        from tkinter import messagebox

        cards_snapshot = list(self.selected)

        if self._mode == "ai_hand":
            if len(cards_snapshot) != 27:
                messagebox.showwarning("数量不对", f"需要选满 27 张，当前 {len(cards_snapshot)} 张。")
                return
            self._result_ai_hand = " ".join(cards_snapshot)
            self._done_var.set(1)
            return

        if self._mode != "human":
            return

        if not cards_snapshot:
            messagebox.showwarning("未选择", "请至少选择 1 张牌，或点击 PASS。")
            return

        need = Counter(cards_snapshot)
        violated = []
        for c, n in need.items():
            avail = self._available_for(c)
            if avail < n:
                violated.append((c, n, avail))
        if violated:
            details = ", ".join([f"{format_card_for_human(c)} need={n} avail={a}" for c, n, a in violated])
            messagebox.showerror("不可用", f"包含不可用的牌：{details}")
            return

        try:
            act = classify_action(cards_snapshot, self.cur_rank)
        except Exception as e:
            messagebox.showerror("牌型无效", str(e))
            return

        if not beats(act, self.greater_action, self.cur_rank):
            messagebox.showerror("不压过", "不压过当前牌（或牌型不匹配）。")
            return

        self._result_action = act
        self._done_var.set(1)

    def _on_close(self):
        # 用户关闭窗口：
        # - AI 手牌模式：视为随机（返回空字符串），但保留窗口实例供后续复用。
        # - 人类出牌模式：本回合回退到终端输入；下次需要输入时再把窗口显示出来。
        self._closed_for_turn = True
        if self._mode == "ai_hand":
            self._result_ai_hand = ""
        # 只隐藏不销毁，便于后续再次打开。
        try:
            self.root.withdraw()
        except Exception:
            pass
        self._done_var.set(1)

    def _idle(self):
        self._mode = "idle"
        self._clear_selection()
        self._refresh_ui()

    def pick_action(
        self,
        *,
        seat: int,
        remaining: int,
        ai_hand: List[Card],
        played_cards: Counter,
        greater_action: Optional[Action],
    ) -> Action:
        self._mode = "human"
        self.seat = seat
        self.remaining = remaining
        self.ai_hand = ai_hand
        self.ai_counter = Counter(ai_hand)
        self.played_cards = played_cards
        self.greater_action = greater_action
        self._result_action = None
        self._result_ai_hand = None
        self._closed_for_turn = False

        self._clear_selection()
        self._refresh_ui()

        # 等待用户操作
        self._done_var.set(0)
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
        except Exception:
            pass
        self.root.wait_variable(self._done_var)

        if self._closed_for_turn:
            # 本回合回退到终端输入。
            self._idle()
            raise _GuiFallbackToTerminalThisTurn()

        if self._result_action is None:
            self._idle()
            raise _GuiFallbackToTerminalThisTurn()

        act = self._result_action
        self._idle()
        return act

    def pick_ai_hand(self) -> str:
        # AI 手牌选择：不受已出牌/AI 手牌冲突约束，仅检查“两副牌库存”上限。
        self._mode = "ai_hand"
        self.seat = 0
        self.remaining = 27
        self.ai_hand = []
        self.ai_counter = Counter()
        self.played_cards = Counter()
        self.greater_action = None
        self._result_action = None
        self._result_ai_hand = None
        self._closed_for_turn = False

        self._clear_selection()
        self._refresh_ui()

        self._done_var.set(0)
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
        except Exception:
            pass
        self.root.wait_variable(self._done_var)

        # 如果用户关闭窗口：视为随机
        if self._closed_for_turn:
            self._idle()
            return ""

        v = (self._result_ai_hand or "").strip()
        self._idle()
        return v


_HUMAN_PICKER: Optional[_PersistentHumanActionPicker] = None


def _pick_human_action_gui(
    *,
    seat: int,
    remaining: int,
    cur_rank: str,
    ai_hand: List[Card],
    played_cards: Counter,
    greater_action: Optional[Action],
) -> Action:
    global _HUMAN_PICKER

    # 只创建一次，整个会话复用同一个窗口。
    if _HUMAN_PICKER is None:
        _HUMAN_PICKER = _PersistentHumanActionPicker(cur_rank=cur_rank)

    return _HUMAN_PICKER.pick_action(
        seat=seat,
        remaining=remaining,
        ai_hand=ai_hand,
        played_cards=played_cards,
        greater_action=greater_action,
    )


def _parse_human_token_to_card(token: str) -> Optional[Card]:
    """将人类输入的单个牌面 token 解析为内部 Card 编码（如：♥10 -> HT，♣3 -> C3）。

    注意：Joker 需要单独处理，因为纯 'JOKER' 是不明确的（可能是 HR/SB）。
    """

    t = token.strip()
    if not t:
        return None
    up = t.upper()
    if t in {"大JOKER", "大王"} or up in {"BIGJOKER"}:
        return "HR"
    if t in {"小JOKER", "小王"} or up in {"SMALLJOKER"}:
        return "SB"
    if up in {"HR", "SB"}:
        return up

    suit_from_symbol = {"♣": "C", "♦": "D", "♠": "S", "♥": "H"}
    if t[0] in suit_from_symbol:
        suit = suit_from_symbol[t[0]]
        rest = t[1:]
    else:
        suit = up[0]
        rest = up[1:]

    if suit not in {"C", "D", "S", "H"}:
        return None

    rest = rest.strip()
    if rest in {"10", "T"}:
        rank = "T"
    else:
        rank = rest[:1].upper()

    if rank not in set(list("23456789TJQKA")):
        return None
    return f"{suit}{rank}"


BASE_RANK_VALUE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
    "B": 16,
    "R": 17,
}


def rank_value(rank_char: str, cur_rank: str) -> int:
    if rank_char == cur_rank:
        return 15
    return BASE_RANK_VALUE[rank_char]


def make_double_deck() -> List[Card]:
    # 两副牌：允许重复。用字符串表示牌面；重复牌用同一字符串重复出现来表示。
    cards = list(CardToNum.keys())
    return cards + cards


def parse_cards_csv(s: str) -> List[Card]:
    if not s:
        return []
    # 允许用逗号或空格分隔。
    raw = s.replace(",", " ").split()
    return [x.strip() for x in raw if x.strip()]


def remove_multiset(deck: List[Card], cards: List[Card]) -> None:
    deck_counter = Counter(deck)
    need = Counter(cards)
    missing = []
    for c, n in need.items():
        if deck_counter[c] < n:
            missing.append((c, n, deck_counter[c]))
    if missing:
        details = ", ".join([f"{format_card_for_human(c)} need={n} have={h}" for c, n, h in missing])
        raise ValueError(f"ai-hand 超出两副牌库存: {details}")
    for c in cards:
        deck.remove(c)


def action_key_from_cards(cards: List[Card], cur_rank: str) -> Tuple[int, int]:
    # 返回 (主比较值, 长度)，用于粗略比较。
    if not cards:
        return (0, 0)
    ranks = [c[-1] for c in cards]
    # 同点数牌型：主比较值取该点数的权值。
    if len(set(ranks)) == 1:
        return (rank_value(ranks[0], cur_rank), len(cards))
    # 连续牌型（顺子/三连对/钢板）：主比较值取（按 A 高）最大点数的权值。
    values = sorted(rank_value(r, cur_rank) for r in set(ranks))
    return (max(values), len(cards))


def classify_action(cards: List[Card], cur_rank: str) -> Action:
    # 返回 [type, key, cards]
    if not cards:
        return ["PASS", "PASS", []]

    ranks = [c[-1] for c in cards]
    counts = Counter(ranks)
    uniq = len(counts)
    n = len(cards)

    # 单张/对子/三张/炸弹（同点数）
    if uniq == 1:
        r = ranks[0]
        if n == 1:
            return ["Single", r, cards]
        if n == 2:
            return ["Pair", r, cards]
        if n == 3:
            return ["Trips", r, cards]
        if n >= 4:
            return ["Bomb", r, cards]

    # 三带二（3+2）
    if n == 5 and sorted(counts.values()) == [2, 3]:
        trip_rank = [r for r, c in counts.items() if c == 3][0]
        return ["ThreeWithTwo", trip_rank, cards]

    # 三连对（3 个连续对子）
    if n == 6 and sorted(counts.values()) == [2, 2, 2]:
        # 显示用最小点数作为 key；比较时会用计算得到的比较值。
        ordered = sorted(counts.keys(), key=lambda r: rank_value(r, cur_rank))
        # 按点数检查是否连续
        vals = [rank_value(r, cur_rank) for r in ordered]
        if vals[0] + 1 == vals[1] and vals[1] + 1 == vals[2]:
            return ["ThreePair", ordered[0], cards]

    # 钢板（2 个连续三张）
    if n == 6 and sorted(counts.values()) == [3, 3]:
        ordered = sorted(counts.keys(), key=lambda r: rank_value(r, cur_rank))
        vals = [rank_value(r, cur_rank) for r in ordered]
        if vals[0] + 1 == vals[1]:
            return ["TwoTrips", ordered[0], cards]

    # 顺子（长度 5，允许 A2345）
    if n == 5 and all(c == 1 for c in counts.values()):
        ordered = sorted(counts.keys(), key=lambda r: BASE_RANK_VALUE.get(r, 99))
        vals = [BASE_RANK_VALUE[r] for r in ordered]
        # A2345
        if set(vals) == {14, 2, 3, 4, 5}:
            return ["Straight", "5", cards]
        vals_sorted = sorted(vals)
        if vals_sorted[0] + 1 == vals_sorted[1] and vals_sorted[1] + 1 == vals_sorted[2] and vals_sorted[2] + 1 == vals_sorted[3] and vals_sorted[3] + 1 == vals_sorted[4]:
            # key 取最高点数字符（尽力而为）
            high_val = vals_sorted[-1]
            inv = {v: k for k, v in BASE_RANK_VALUE.items() if k in list("23456789TJQKA")}
            key = inv.get(high_val, ordered[-1])
            return ["Straight", key, cards]

    raise ValueError("无法识别牌型（仅支持: PASS/Single/Pair/Trips/Bomb/Straight/ThreeWithTwo/ThreePair/TwoTrips）")


def beats(action: Action, greater_action: Optional[Action], cur_rank: str) -> bool:
    if greater_action is None:
        return action[0] != "PASS"

    if action[0] == "PASS":
        return True

    if greater_action[0] == "PASS":
        return action[0] != "PASS"

    # 炸弹压过所有非炸弹。
    if action[0] == "Bomb" and greater_action[0] != "Bomb":
        return True

    if action[0] != greater_action[0]:
        return False

    a_key, a_len = action_key_from_cards(action[2], cur_rank)
    g_key, g_len = action_key_from_cards(greater_action[2], cur_rank)

    # 同类型必须长度一致。
    if a_len != g_len:
        return False

    return a_key > g_key


def generate_actions_from_hand(hand: List[Card], cur_rank: str) -> List[Action]:
    # 遵循 torch 客户端的动作表示：[type, key, cards]
    card_value_s2v = dict(BASE_RANK_VALUE)
    card_value_s2v[cur_rank] = 15

    sorted_cards, bomb_info = combine_handcards(hand, cur_rank, card_value_s2v)

    actions: List[Action] = []

    # 单张、对子、三张
    for c in sorted_cards.get("Single", []):
        actions.append(["Single", c[-1], [c]])
    for pair in sorted_cards.get("Pair", []):
        actions.append(["Pair", pair[0][-1], pair])
    for trips in sorted_cards.get("Trips", []):
        actions.append(["Trips", trips[0][-1], trips])

    # 顺子/连续结构已由 util.combine_handcards 生成
    for st in sorted_cards.get("Straight", []) or []:
        actions.append(["Straight", st[0][-1], st])

    # 三带二
    if sorted_cards.get("Pair") and sorted_cards.get("Trips"):
        for t in sorted_cards["Trips"]:
            for p in sorted_cards["Pair"]:
                actions.append(["ThreeWithTwo", t[0][-1], t + p])

    # 三连对
    pair_actions = []
    for pair in sorted_cards.get("Pair", []) or []:
        pair_actions.append(["Pair", pair[0][-1], pair])

    pair_actions.sort(key=lambda a: rank_value(a[1], cur_rank))
    for i in range(len(pair_actions) - 2):
        v0 = rank_value(pair_actions[i][1], cur_rank)
        v1 = rank_value(pair_actions[i + 1][1], cur_rank)
        v2 = rank_value(pair_actions[i + 2][1], cur_rank)
        if v0 + 1 == v1 and v1 + 1 == v2:
            cards = pair_actions[i][2] + pair_actions[i + 1][2] + pair_actions[i + 2][2]
            actions.append(["ThreePair", pair_actions[i][1], cards])

    # 钢板
    trips_actions = []
    for t in sorted_cards.get("Trips", []) or []:
        trips_actions.append(["Trips", t[0][-1], t])
    trips_actions.sort(key=lambda a: rank_value(a[1], cur_rank))
    for i in range(len(trips_actions) - 1):
        v0 = rank_value(trips_actions[i][1], cur_rank)
        v1 = rank_value(trips_actions[i + 1][1], cur_rank)
        if v0 + 1 == v1:
            cards = trips_actions[i][2] + trips_actions[i + 1][2]
            actions.append(["TwoTrips", trips_actions[i][1], cards])

    # 炸弹（同点数 4 张及以上）
    for bomb_cards in sorted_cards.get("Bomb", []) or []:
        actions.append(["Bomb", bomb_cards[0][-1], bomb_cards])

    # 尽量稳定（可复现）的排序
    def sort_key(a: Action) -> Tuple[int, int, str]:
        t = a[0]
        type_order = {
            "Single": 1,
            "Pair": 2,
            "Trips": 3,
            "ThreeWithTwo": 4,
            "ThreePair": 5,
            "TwoTrips": 6,
            "Straight": 7,
            "Bomb": 8,
        }.get(t, 99)
        k, ln = action_key_from_cards(a[2], cur_rank)
        return (type_order, ln, str(k))

    actions.sort(key=sort_key)
    return actions


@dataclass
class SeatConn:
    seat: int
    ws: ServerConnection


class MiniDanServer:
    def __init__(
        self,
        *,
        cur_rank: str,
        self_rank: str,
        oppo_rank: str,
        ai_hand: List[Card],
        max_steps: int,
        start_pos: int,
    ):
        self.cur_rank = cur_rank
        self.self_rank = self_rank
        self.oppo_rank = oppo_rank
        self.ai_hand = ai_hand[:]  # 座位 seat0
        self.max_steps = max_steps

        self.seat_conns: Dict[int, SeatConn] = {}

        # 记录已出过的牌（两副牌库存）以及人类座位剩余手牌数量。
        self.played_cards = Counter()  # Card -> 已出过数量
        self.human_remaining = {1: 27, 2: 27, 3: 27}

        self.current_pos = int(start_pos)
        self.greater_action: Optional[Action] = None
        self.greater_pos: Optional[int] = None
        self.passes_since_play = 0

    async def register(self, seat: int, ws: ServerConnection):
        self.seat_conns[seat] = SeatConn(seat=seat, ws=ws)

    async def broadcast(self, payload: dict):
        if not self.seat_conns:
            return
        msg = __import__("json").dumps(payload)
        await asyncio.gather(*[c.ws.send(msg) for c in self.seat_conns.values()])

    async def send_to(self, seat: int, payload: dict):
        msg = __import__("json").dumps(payload)
        await self.seat_conns[seat].ws.send(msg)

    async def recv_from(self, seat: int) -> dict:
        raw = await self.seat_conns[seat].ws.recv()
        return __import__("json").loads(raw)

    async def send_beginning(self, seat: int):
        await self.send_to(
            seat,
            {
                "type": "notify",
                "stage": "beginning",
                "myPos": seat,
                # 兼容 ai2 客户端。
                "handCards": self.ai_hand[:] if seat == 0 else [],
                # 保持为字符串，满足 torch 客户端 prepare() 的要求。
                "curRank": self.cur_rank,
                "selfRank": self.self_rank,
                "oppoRank": self.oppo_rank,
            },
        )

    def prompt_human_action(self, seat: int) -> Action:
        # 优先使用 GUI 选牌。
        try:
            import tkinter  # noqa: F401

            act = _pick_human_action_gui(
                seat=seat,
                remaining=self.human_remaining.get(seat, 27),
                cur_rank=self.cur_rank,
                ai_hand=self.ai_hand,
                played_cards=self.played_cards,
                greater_action=self.greater_action,
            )
            # 将 GUI 中的选择结果回显到终端。
            if act[0] == "PASS":
                print(f"Seat{seat}（窗口）选择：过")
            else:
                print(f"Seat{seat}（窗口）选择：{format_action_for_human(act)}")
            return act
        except _GuiFallbackToTerminalThisTurn:
            # 本回合用户关闭窗口，回退到终端输入。
            pass
        except Exception as e:
            # GUI 不可用则回退到终端输入，但只打印一次失败原因（避免刷屏）。
            if not getattr(self, "_gui_prompt_failed_once", False):
                self._gui_prompt_failed_once = True
                print(f"GUI 选牌窗口不可用，已回退终端输入：{type(e).__name__}: {e}")

        while True:
            remaining = self.human_remaining.get(seat, 27)
            if self.greater_action is None:
                prompt = f"Seat{seat}（剩余 {remaining} 张）出牌（如: ♥3 ♥3 或 ♦10）："
            else:
                prompt = f"Seat{seat}（剩余 {remaining} 张）跟牌（可 PASS）。当前牌: {format_action_for_human(self.greater_action)}："

            s = input(prompt).strip()
            if not s:
                continue
            if s.upper() in {"P", "PASS"}:
                act = ["PASS", "PASS", []]
                if self.greater_action is None:
                    print("先手不能 PASS，请出牌。")
                    continue
                return act

            raw_tokens = s.replace(",", " ").split()

            # 将人类友好输入解析为内部编码。
            ai_counter = Counter(self.ai_hand)
            parsed_cards: List[Card] = []
            parse_error = None
            for tok in raw_tokens:
                up = tok.strip().upper()
                if up == "JOKER":
                    # 确定性地选择一个可用的王：优先 HR，其次 SB。
                    for cand in ("HR", "SB"):
                        available = 2 - self.played_cards.get(cand, 0) - ai_counter.get(cand, 0)
                        if available > 0:
                            parsed_cards.append(cand)
                            break
                    else:
                        parse_error = "JOKER 已不可用（与 AI 手牌冲突/已出过/超出两副牌库存）"
                        break
                    continue

                c = _parse_human_token_to_card(tok)
                if not c:
                    parse_error = f"无法识别牌面: {tok}（支持如 ♥3 ♦10 ♣A JOKER）"
                    break
                parsed_cards.append(c)

            if parse_error:
                print(parse_error)
                continue

            cards = parsed_cards

            if len(cards) > self.human_remaining.get(seat, 27):
                print(f"出牌张数超过 Seat{seat} 剩余手牌：剩余={self.human_remaining.get(seat, 27)} 张。")
                continue

            # 约束：
            # - 不能使用当前在 AI 手牌（seat0）中的牌
            # - 不能使用已经出过的牌（考虑两副牌库存）
            need = Counter(cards)
            violated = []
            for c, n in need.items():
                available = 2 - self.played_cards.get(c, 0) - ai_counter.get(c, 0)
                if available < n:
                    violated.append((c, n, available))
            if violated:
                details = ", ".join(
                    [f"{format_card_for_human(c)} need={n} avail={a}" for c, n, a in violated]
                )
                print(f"出牌包含不可用的牌（与 AI 手牌冲突/已出过/超出两副牌库存）：{details}")
                continue

            try:
                act = classify_action(cards, self.cur_rank)
            except Exception as e:
                print(f"无效输入: {e}")
                continue

            if not beats(act, self.greater_action, self.cur_rank):
                print("不压过当前牌（或牌型不匹配）。")
                continue

            return act

    async def run_game_loop(self):
        # 当前实现：只要求 seat0 连接即可开局。
        if 0 not in self.seat_conns:
            raise RuntimeError("需要 seat0 连接到 /game/client0")

        # 连接后发送 beginning。
        await self.send_beginning(0)

        steps = 0
        stopped_early = False
        while steps < self.max_steps:
            steps += 1

            greater_pos_payload = self.greater_pos if self.greater_pos is not None else -1
            greater_action_payload = self.greater_action if self.greater_action is not None else ["PASS", "PASS", []]

            if self.current_pos == 0:
                # AI 回合
                all_actions = generate_actions_from_hand(self.ai_hand, self.cur_rank)

                if self.greater_action is None:
                    action_list = all_actions
                else:
                    filtered = [a for a in all_actions if beats(a, self.greater_action, self.cur_rank)]
                    action_list = [["PASS", "PASS", []]] + filtered

                if not action_list:
                    # 理论上不应发生（非先手至少能 PASS），这里兜底保证安全。
                    action_list = [["PASS", "PASS", []]]

                msg = {
                    "type": "act",
                    "stage": "play",
                    "handCards": self.ai_hand,
                    "actionList": action_list,
                    "indexRange": len(action_list) - 1,
                    "curRank": self.cur_rank,
                    "selfRank": self.self_rank,
                    "oppoRank": self.oppo_rank,
                    # 兼容 ai2 客户端。
                    "curPos": self.current_pos,
                    "curAction": None,
                    "greaterPos": greater_pos_payload,
                    "greaterAction": greater_action_payload,
                }

                await self.send_to(0, msg)
                resp = await self.recv_from(0)
                act_index = int(resp.get("actIndex", 0))
                if act_index < 0 or act_index >= len(action_list):
                    act_index = 0
                act = action_list[act_index]

                # 扣除 AI 手牌（移除已出的牌）
                if act[0] != "PASS":
                    for c in act[2]:
                        try:
                            self.ai_hand.remove(c)
                        except ValueError:
                            # 如有不一致则忽略；这里是实验服。
                            pass
                    self.played_cards.update(act[2])

            else:
                # 人类回合（服务器控制台输入）
                try:
                    act = self.prompt_human_action(self.current_pos)
                except (EOFError, KeyboardInterrupt):
                    print("输入中断，实验服将停止。")
                    stopped_early = True
                    break
                if act[0] != "PASS":
                    self.human_remaining[self.current_pos] = max(
                        0, self.human_remaining.get(self.current_pos, 27) - len(act[2])
                    )
                    self.played_cards.update(act[2])

            # 计算本轮新状态（ai2 的 notify 需要 greaterPos/greaterAction）
            if act[0] == "PASS":
                new_greater_action = self.greater_action
                new_greater_pos = self.greater_pos
                new_passes = self.passes_since_play + 1
            else:
                new_greater_action = act
                new_greater_pos = self.current_pos
                new_passes = 0

            # 广播 notify(play)
            await self.broadcast(
                {
                    "type": "notify",
                    "stage": "play",
                    "curPos": self.current_pos,
                    "curAction": act,
                    "greaterPos": new_greater_pos if new_greater_pos is not None else -1,
                    "greaterAction": new_greater_action if new_greater_action is not None else ["PASS", "PASS", []],
                }
            )

            # 提交本轮状态
            self.greater_action = new_greater_action
            self.greater_pos = new_greater_pos
            self.passes_since_play = new_passes

            # 自上一次有人出牌后，连续 3 个 PASS 则该墩结束
            if self.greater_action is not None and self.passes_since_play >= 3:
                self.current_pos = int(self.greater_pos)
                self.greater_action = None
                self.greater_pos = None
                self.passes_since_play = 0
            else:
                self.current_pos = (self.current_pos + 1) % 4

            if not self.ai_hand:
                print("AI 已出完牌（实验服将停止）。")
                stopped_early = True
                break

        if not stopped_early and steps >= self.max_steps:
            print("达到 max_steps，实验服将停止。")


async def handler(ws: ServerConnection, server: MiniDanServer):
    # 期望路径格式：/game/client{seat}
    # websockets v15 只传入连接对象；路径可通过 ws.request.path 获取。
    req = getattr(ws, "request", None)
    path = getattr(req, "path", None) or getattr(ws, "path", None) or ""
    seat = None
    if path.startswith("/game/client"):
        try:
            seat = int(path.replace("/game/client", ""))
        except Exception:
            seat = None
    if seat is None:
        await ws.close(code=1008, reason="invalid path")
        return

    await server.register(seat, ws)
    try:
        # 保持连接存活；由主循环驱动消息收发。
        await ws.wait_closed()
    finally:
        server.seat_conns.pop(seat, None)


async def main_async(args):
    deck = make_double_deck()

    while True:
        ai_hand = parse_cards_csv(getattr(args, "ai_hand", ""))
        if not ai_hand:
            random.shuffle(deck)
            ai_hand = deck[:27]
            remove_multiset(deck, ai_hand)
            break

        if len(ai_hand) != 27:
            print(f"AI 手牌数量不对：需要 27 张，当前={len(ai_hand)}。请重新输入或直接回车随机。")
            args.ai_hand = _prompt_ai_hand(args.cur_rank)
            continue

        try:
            remove_multiset(deck, ai_hand)
        except Exception as e:
            print(f"AI 手牌不合法：{e}。请重新输入或直接回车随机。")
            args.ai_hand = _prompt_ai_hand(args.cur_rank)
            continue
        break

    server = MiniDanServer(
        cur_rank=args.cur_rank,
        self_rank=args.self_rank,
        oppo_rank=args.oppo_rank,
        ai_hand=ai_hand,
        max_steps=args.max_steps,
        start_pos=getattr(args, "start_pos", 0),
    )

    repo_root = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
    torch_dir = os.path.abspath(os.path.join(repo_root, "wintest", "torch"))
    child_procs: List[subprocess.Popen] = []

    try:
        try:
            async with websockets.serve(lambda w: handler(w, server), args.host, args.port):
                print(f"mini_danserver listening on ws://{args.host}:{args.port}/game/client0")
                print(f"AI seat0 hand ({len(ai_hand)}): {format_cards_for_human(ai_hand)}")

                if getattr(args, "spawn_ai", True):
                    # 优先使用 torch AI；若缺少模型文件则回退到 wintest/ai2。
                    torch_models_dir = os.path.join(torch_dir, "models")
                    torch_models = glob.glob(os.path.join(torch_models_dir, "ppo*.pth")) if os.path.isdir(torch_models_dir) else []

                    if torch_models:
                        # 在同一个终端启动 ZMQ actor 与 WS AI client。
                        # actor.py 通过相对路径加载模型文件，所以 cwd 需要设为 torch_dir。
                        child_procs.append(
                            subprocess.Popen(
                                [sys.executable, os.path.join(torch_dir, "actor.py"), "--seats", "0"],
                                cwd=torch_dir,
                            )
                        )
                        await asyncio.sleep(0.3)
                        child_procs.append(
                            subprocess.Popen(
                                [sys.executable, os.path.join(torch_dir, "client1.py"), "--seat", "0"],
                                cwd=torch_dir,
                            )
                        )
                    else:
                        ai2_dir = os.path.abspath(os.path.join(repo_root, "wintest", "ai2"))
                        print("未发现 torch 模型文件（wintest/torch/models/ppo*.pth），将回退使用 wintest/ai2 作为 seat0。")
                        child_procs.append(
                            subprocess.Popen(
                                [sys.executable, os.path.join(ai2_dir, "client0.py")],
                                cwd=ai2_dir,
                            )
                        )

                # 运行主循环：会阻塞等待 seat0 连接，以及等待人类的控制台输入。
                while 0 not in server.seat_conns:
                    await asyncio.sleep(0.1)
                await server.run_game_loop()
        except asyncio.CancelledError:
            # asyncio.run() 期间 Ctrl+C 会取消挂起的 await；这里安静退出。
            pass
    finally:
        for p in child_procs:
            try:
                p.terminate()
            except Exception:
                pass


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mini danserver (route3): 指定 AI 起手牌 + 终端手动 3 座位")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=23456)

    # 级牌/段位保持为字符串，兼容 torch 客户端。
    # 根据 UX 约定，curRank 与 AI 手牌在启动时交互输入，因此不强制要求命令行参数。

    p.add_argument("--max-steps", type=int, default=50, help="最多推进多少个出牌动作（避免你不小心打完整局）")

    # 默认单终端模式。
    p.add_argument(
        "--no-spawn-ai",
        action="store_true",
        help="不自动启动 actor/client（需要你自己另开终端启动）。默认会自动启动。",
    )
    return p


def _prompt_cur_rank(default: str = "2") -> str:
    allowed = set(list("23456789TJQKA"))
    while True:
        s = input(f"请输入当前级牌 curRank（默认 {default}，可选 2-9,T,J,Q,K,A）：").strip().upper()
        if not s:
            return default
        if s in allowed:
            return s
        print("输入无效，请重新输入。")


def _prompt_start_pos(default: int = 0) -> int:
    while True:
        s = input(f"请输入开局先手座位（0-3，默认 {default}）：").strip()
        if not s:
            return default
        try:
            v = int(s)
        except Exception:
            print("输入无效，请输入 0-3。")
            continue
        if v in (0, 1, 2, 3):
            return v
        print("输入无效，请输入 0-3。")


def _prompt_ai_hand(cur_rank: str) -> str:
    # 优先 GUI 选择；GUI 不可用则回退文本输入。
    global _HUMAN_PICKER
    try:
        import tkinter  # noqa: F401
    except Exception:
        return input(
            "请输入 AI(seat0) 起手 27 张牌（空=随机；用空格/逗号分隔，如: H2 H2 C4 ...）："
        ).strip()

    if _HUMAN_PICKER is None:
        _HUMAN_PICKER = _PersistentHumanActionPicker(cur_rank=cur_rank)
    return _HUMAN_PICKER.pick_ai_hand().strip()


def main():
    try:
        args = build_arg_parser().parse_args()
        # 交互输入
        args.cur_rank = _prompt_cur_rank("2")
        args.start_pos = _prompt_start_pos(0)
        args.self_rank = "2"
        args.oppo_rank = "2"
        args.ai_hand = _prompt_ai_hand(args.cur_rank)
        args.spawn_ai = not args.no_spawn_ai
        asyncio.run(main_async(args))
    except (KeyboardInterrupt, asyncio.CancelledError, EOFError):
        print("已退出。")
        return


if __name__ == "__main__":
    main()
