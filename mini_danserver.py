"""mini danserver：仓库根目录可直接运行的实验服。

用法：
  python mini_danserver.py

本文件包含完整实现（不再把实现放在目录深处）。

改进点：
- 在“AI(seat0) 起手牌选择”窗口中新增“截图框选识别”按钮：
  1) 截取当前屏幕
  2) 鼠标框选截图中的一块区域
  3) 只在该区域内调用 tools/card_yolo_detect.py 识别牌
  4) 将识别结果自动填入选择窗口
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import itertools
import os
import random
import subprocess
import sys
import tempfile
import time
import threading
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
  from websockets.server import ServerConnection
else:
  ServerConnection = Any  # type: ignore

# 复用 torch 客户端使用的牌面编码全集。
# 注意：本仓库的 wintest/ 不是 Python 包；这里按已有脚本做法，
# 通过 sys.path 引入 wintest/torch/util.py。
_THIS_DIR = os.path.dirname(__file__)
_TORCH_DIR = os.path.abspath(os.path.join(_THIS_DIR, "wintest", "torch"))
if _TORCH_DIR not in sys.path:
  sys.path.insert(0, _TORCH_DIR)

from util import CardToNum, combine_handcards  # type: ignore


Card = str
Action = List  # [type, key, cards]

# 全量重新开始（从选择级牌/先手/AI 手牌重新走）的跨线程信号。
_RESTART_FULL_EVENT = threading.Event()


def _adviser_card(cur_rank: str) -> Card:
  # 参谋：红桃级牌
  return f"H{cur_rank}"


def _is_adviser(card: Card, cur_rank: str) -> bool:
  return str(card) == _adviser_card(cur_rank)


def _action_detail(action: Action) -> Dict[str, Any]:
  # 可选的 action[3]：用于描述参谋补法等额外信息。
  if isinstance(action, list) and len(action) >= 4 and isinstance(action[3], dict):
    return action[3]
  return {}


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


def format_action_for_human(action: Optional[Action], cur_rank: Optional[str] = None) -> str:
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
    "StraightFlush": "同花顺",
    "Bomb": "炸弹",
  }.get(action[0], action[0])

  extra = ""
  try:
    if cur_rank:
      detail = _action_detail(action)
      adv = detail.get("adviser_as")
      cards = action[2] if isinstance(action, list) and len(action) >= 3 else []
      adviser_count = sum(1 for c in cards if _is_adviser(c, cur_rank))
      if adviser_count > 0:
        adv_list: List[str] = []
        if adv:
          # adv 可以是 ['A'] 或 ['Q','Q'] 或 {'A':1,'Q':1}
          if isinstance(adv, dict):
            for r, n in adv.items():
              adv_list.extend([str(r)] * int(n))
          elif isinstance(adv, (list, tuple)):
            adv_list = [str(x) for x in adv]

        # 允许“参谋按本身点数使用”：此时 adv_list 可能比参谋张数短。
        if adv_list:
          if len(adv_list) < adviser_count:
            adv_list.extend([str(cur_rank)] * (adviser_count - len(adv_list)))

          # 压缩重复点数，保证输出简洁且无歧义。
          cnt = Counter(adv_list)
          ordered_unique: List[str] = []
          seen = set()
          for r in adv_list:
            if r in seen:
              continue
            seen.add(r)
            ordered_unique.append(r)

          parts: List[str] = []
          for r in ordered_unique:
            n = int(cnt.get(r, 0))

            rr = str(r)
            try:
              if (
                len(rr) >= 2
                and rr[0] in {"S", "H", "C", "D"}
                and rr[-1] in set(RANKS_NO_JOKER)
                and rr not in set(RANKS_NO_JOKER)
              ):
                rr = format_card_for_human(rr)
            except Exception:
              pass

            if n <= 1:
              parts.append(rr)
            else:
              parts.append(f"{rr}×{n}")

          extra = f"（参谋当: {' '.join(parts)}）"
        else:
          # 没有 detail：提示可能存在歧义。
          extra = "（含参谋）"
  except Exception:
    pass

  return f"{type_ch}: {format_cards_for_human(action[2])}{extra}".strip()


class _ScreenRoiSelector:
  def __init__(self, parent, pil_img, title: str = "框选识别区域（拖拽选择，ESC 取消）"):
    import tkinter as tk

    self._tk = tk
    self._parent = parent
    self._img = pil_img

    self._result: Optional[Tuple[int, int, int, int]] = None
    self._start: Optional[Tuple[int, int]] = None
    self._rect = None

    top = tk.Toplevel(parent)
    self.top = top
    top.title(title)
    top.attributes("-topmost", True)
    top.configure(bg="black")

    sw = int(top.winfo_screenwidth())
    sh = int(top.winfo_screenheight())
    iw, ih = self._img.size
    scale = min(sw / max(1, iw), sh / max(1, ih), 1.0)
    self._scale = float(scale)

    if scale < 1.0:
      disp = self._img.resize((int(iw * scale), int(ih * scale)))
    else:
      disp = self._img

    from PIL import ImageTk

    self._photo = ImageTk.PhotoImage(disp)
    cw = self._photo.width()
    ch = self._photo.height()

    canvas = tk.Canvas(top, width=cw, height=ch, highlightthickness=0)
    self.canvas = canvas
    canvas.pack()
    canvas.create_image(0, 0, image=self._photo, anchor="nw")

    canvas.bind("<Button-1>", self._on_down)
    canvas.bind("<B1-Motion>", self._on_move)
    canvas.bind("<ButtonRelease-1>", self._on_up)
    top.bind("<Escape>", self._on_cancel)
    top.protocol("WM_DELETE_WINDOW", self._on_cancel)

    # 居中显示
    try:
      top.update_idletasks()
      x = max(0, (sw - cw) // 2)
      y = max(0, (sh - ch) // 2)
      top.geometry(f"{cw}x{ch}+{x}+{y}")
    except Exception:
      pass

    top.grab_set()

  def _on_down(self, ev):
    self._start = (int(ev.x), int(ev.y))
    if self._rect is not None:
      try:
        self.canvas.delete(self._rect)
      except Exception:
        pass
    self._rect = self.canvas.create_rectangle(ev.x, ev.y, ev.x, ev.y, outline="cyan", width=2)

  def _on_move(self, ev):
    if not self._start or self._rect is None:
      return
    x0, y0 = self._start
    self.canvas.coords(self._rect, x0, y0, int(ev.x), int(ev.y))

  def _on_up(self, ev):
    if not self._start:
      return
    x0, y0 = self._start
    x1, y1 = int(ev.x), int(ev.y)
    x_min, x_max = (x0, x1) if x0 <= x1 else (x1, x0)
    y_min, y_max = (y0, y1) if y0 <= y1 else (y1, y0)

    # 转回原图坐标
    if self._scale > 0:
      x_min = int(round(x_min / self._scale))
      y_min = int(round(y_min / self._scale))
      x_max = int(round(x_max / self._scale))
      y_max = int(round(y_max / self._scale))

    if x_max - x_min < 5 or y_max - y_min < 5:
      self._result = None
    else:
      self._result = (x_min, y_min, x_max, y_max)

    self._finish()

  def _on_cancel(self, _ev=None):
    self._result = None
    self._finish()

  def _finish(self):
    try:
      self.top.grab_release()
    except Exception:
      pass
    try:
      self.top.destroy()
    except Exception:
      pass

  def select(self) -> Optional[Tuple[int, int, int, int]]:
    self._parent.wait_window(self.top)
    return self._result


def _cards_from_yolo_payload(payload: List[dict], *, target_count: Optional[int] = None) -> List[Card]:
  suit_map = {
    "spades": "S",
    "hearts": "H",
    "clubs": "C",
    "diamonds": "D",
  }

  # 先按 score 从高到低排序；再做两副牌库存限制（每种最多 2）。
  items = sorted(payload, key=lambda d: float(d.get("score", 0.0)), reverse=True)
  out: List[Card] = []
  cnt = Counter()

  for d in items:
    label = str(d.get("label", ""))
    rank = d.get("rank")
    suit = d.get("suit")

    card: Optional[Card] = None
    if rank == "joker":
      # 约定：dw=大王，xw=小王
      if label == "dw":
        card = "HR"
      elif label == "xw":
        card = "SB"
    else:
      if not rank or not suit:
        continue
      s = suit_map.get(str(suit), "")
      r = str(rank)
      if r == "10":
        r = "T"
      r = r.upper()
      if s in {"S", "H", "C", "D"} and r in set(list("23456789TJQKA")):
        card = f"{s}{r}"

    if not card:
      continue

    if cnt[card] >= 2:
      continue

    out.append(card)
    cnt[card] += 1
    if target_count is not None and len(out) >= int(target_count):
      break

  return out


def _filter_by_availability(cards: List[Card], *, available_fn, target_count: int) -> Tuple[List[Card], List[Card]]:
  """Filter a ranked card list by per-card availability and max length.

  Returns (kept, dropped).
  """

  kept: List[Card] = []
  dropped: List[Card] = []
  used = Counter()
  for c in cards:
    if len(kept) >= int(target_count):
      dropped.append(c)
      continue
    try:
      avail = int(available_fn(c))
    except Exception:
      avail = 0
    if avail <= 0 or used[c] >= avail:
      dropped.append(c)
      continue
    kept.append(c)
    used[c] += 1
  return kept, dropped


def _run_yolo_on_image(
  image_path: str,
  *,
  repo_root: str,
  model_dir: str,
) -> List[dict]:
  import json

  script = os.path.join(repo_root, "tools", "card_yolo_detect.py")
  if not os.path.exists(script):
    raise FileNotFoundError(f"Cannot find YOLO detector script: {script}")

  with tempfile.TemporaryDirectory(prefix="dan_yolo_") as td:
    out_json = os.path.join(td, "result.json")
    cmd = [
      sys.executable,
      script,
      "--image",
      image_path,
      "--model-dir",
      model_dir,
      "--input-sizes",
      "416,608",
      "--conf",
      "0.09",
      "--nms",
      "0.70",
      "--max-per-label",
      "2",
      "--dedup-location",
      "--location-dist",
      "20",
      "--location-iou",
      "0.62",
      "--min-score",
      "0.20",
      "--out",
      out_json,
    ]
    p = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if p.returncode != 0:
      stderr = p.stderr or ""
      if "No module named 'cv2'" in stderr or "ModuleNotFoundError" in stderr and "cv2" in stderr:
        raise RuntimeError(
          "YOLO 识别失败：缺少依赖 OpenCV（cv2）。\n\n"
          "请用‘用户级’方式安装（不使用虚拟环境）：\n"
          "  python -m pip install --user opencv-python\n\n"
          "安装完成后重新点击‘截图框选识别’。\n\n"
          "原始错误：\n" + stderr
        )
      raise RuntimeError(
        "YOLO 识别失败：\n"
        + (p.stdout or "")
        + ("\n" if p.stdout and p.stderr else "")
        + (p.stderr or "")
      )

    with open(out_json, "r", encoding="utf-8") as f:
      return json.load(f)


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

    # 新增：截图框选识别（仅 AI 手牌选择模式启用）
    self.detect_btn = tk.Button(action_bar, text="截图框选识别", command=self._on_detect_from_screen)
    self.detect_btn.pack(side="left", padx=(8, 0))

    # 新增：一键取消当前局并重新开始
    self.restart_btn = tk.Button(action_bar, text="重新开局", command=self._on_restart_game)
    self.restart_btn.pack(side="left", padx=(8, 0))

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
            self.context_var.set(f"当前牌：{format_action_for_human(self.greater_action, self.cur_rank)}")
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
      self.detect_btn.configure(state="normal")
      self.restart_btn.configure(state="normal")
    elif self._mode == "human":
      self.undo_btn.configure(state="normal")
      self.confirm_btn.configure(state="normal")
      self.detect_btn.configure(state="normal")
      self.restart_btn.configure(state="normal")
      if self.greater_action is None:
        self.pass_btn.configure(state="disabled")
      else:
        self.pass_btn.configure(state="normal")
    else:
      self.pass_btn.configure(state="disabled")
      self.undo_btn.configure(state="disabled")
      self.confirm_btn.configure(state="disabled")
      self.detect_btn.configure(state="disabled")
      self.restart_btn.configure(state="normal")

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

  def _on_detect_from_screen(self):
    from tkinter import messagebox

    if self._mode not in {"ai_hand", "human"}:
      return

    try:
      from PIL import ImageGrab
    except Exception as e:
      messagebox.showerror("缺少依赖", f"截图需要 Pillow（PIL）。错误：{type(e).__name__}: {e}")
      return

    try:
      self.root.withdraw()
      self.root.update()
      time.sleep(0.15)
    except Exception:
      pass

    try:
      shot = ImageGrab.grab()
    except Exception as e:
      try:
        self.root.deiconify()
      except Exception:
        pass
      messagebox.showerror("截图失败", f"{type(e).__name__}: {e}")
      return

    # 截图已完成，可以恢复主窗口；随后弹出框选窗口。
    try:
      self.root.deiconify()
      self.root.update()
    except Exception:
      pass

    selector = _ScreenRoiSelector(self.root, shot)
    bbox = selector.select()

    if not bbox:
      return

    crop = shot.crop(bbox)

    repo_root = os.path.abspath(_THIS_DIR)
    model_dir = os.path.join(repo_root, "模型")
    if not os.path.isdir(model_dir):
      messagebox.showerror("缺少模型", f"找不到模型目录：{model_dir}")
      return

    # 保存临时图片，调用现成 YOLO 脚本，避免在 GUI 进程内直接 import OpenCV。
    try:
      with tempfile.TemporaryDirectory(prefix="dan_crop_") as td:
        tmp_img = os.path.join(td, "crop.png")
        crop.save(tmp_img)
        payload = _run_yolo_on_image(tmp_img, repo_root=repo_root, model_dir=model_dir)
    except Exception as e:
      messagebox.showerror("识别失败", str(e))
      return

    cards_ranked = _cards_from_yolo_payload(payload, target_count=None)
    if not cards_ranked:
      messagebox.showwarning("未识别到牌", "该区域未识别到任何牌，请重新框选或放大区域。")
      return

    # AI 起手牌需要 27 张；人类出牌模式按 remaining 作为上限。
    if self._mode == "ai_hand":
      target = 27
      cards_sel = _cards_from_yolo_payload(payload, target_count=27)
      dropped: List[Card] = []
    else:
      target = int(max(1, self.remaining))
      # 先按 score/库存过滤成 Card 列表，再按“本局可用库存”二次过滤。
      cards_sel, dropped = _filter_by_availability(
        cards_ranked,
        available_fn=self._available_for,
        target_count=target,
      )

    # 覆盖当前选择
    self.selected = list(cards_sel)
    self.selected_count = Counter(self.selected)
    self._refresh_ui()

    # 需求：框选后不再等待“确认”，直接生效。
    if self._mode == "ai_hand":
      # AI 起手牌：识别到 27 张则直接提交；否则保留结果供手动补齐。
      if len(cards_sel) == 27:
        self._result_ai_hand = " ".join(cards_sel)
        self._done_var.set(1)
      return

    # 人类出牌：直接按当前牌面规则校验并“打出”。
    if self._mode == "human":
      cards_snapshot = list(cards_sel)
      if not cards_snapshot:
        messagebox.showwarning("未识别到牌", "该区域未识别到任何牌，请重新框选或放大区域。")
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
        act = _resolve_action_with_adviser(
          cards_snapshot,
          cur_rank=self.cur_rank,
          greater_action=self.greater_action,
          tk_parent=self.root,
          allow_interactive=True,
        )
      except Exception as e:
        messagebox.showerror("牌型无效", str(e))
        return

      if not beats(act, self.greater_action, self.cur_rank):
        messagebox.showerror("不压过", "不压过当前牌（或牌型不匹配）。")
        return

      self._result_action = act
      self._done_var.set(1)
      return

  def _on_pass(self):
    if self._mode != "human":
      return
    if self.greater_action is None:
      return
    self._result_action = ["PASS", "PASS", []]
    self._done_var.set(1)

  def _on_restart_game(self):
    # 全量重新开始：任何时候都可用。
    _RESTART_FULL_EVENT.set()
    # 若当前正等待出牌/选牌，立刻解除阻塞。
    if self._mode == "human":
      self._result_action = ["RESTART_FULL", "RESTART_FULL", []]
    elif self._mode == "ai_hand":
      self._result_ai_hand = ""
      self._closed_for_turn = True
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
      act = _resolve_action_with_adviser(
        cards_snapshot,
        cur_rank=self.cur_rank,
        greater_action=self.greater_action,
        tk_parent=self.root,
        allow_interactive=True,
      )
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


RANKS_NO_JOKER = list("23456789TJQKA")

# 参谋可变成王以外的任何牌：点数 + 花色均可变（不含 Joker）。
ALL_NON_JOKER_CARDS = [f"{s}{r}" for s in "SHCD" for r in RANKS_NO_JOKER]


def _action_key_from_action(action: Action, cur_rank: str) -> Tuple[int, int]:
  """Return (strength_key, length) using the declared action, not raw card ranks.

  This avoids ambiguity when actions contain 参谋（红桃级牌）作为癞子补牌。
  """
  if not action or action[0] == "PASS":
    return (0, 0)
  t = action[0]
  k = action[1]
  ln = len(action[2]) if isinstance(action, list) and len(action) >= 3 else 0

  if t in {"Single", "Pair", "Trips", "Bomb"}:
    return (rank_value(str(k), cur_rank), ln)

  if t == "ThreeWithTwo":
    return (rank_value(str(k), cur_rank), 5)

  if t == "Straight":
    # key 用最高点数（A2345 时为 '5'）
    kk = str(k)
    if kk == "5":
      return (BASE_RANK_VALUE["5"], 5)
    return (BASE_RANK_VALUE.get(kk, 0), 5)

  if t == "StraightFlush":
    # 同花顺：比较规则同顺子（key 用最高点数，A2345 时为 '5'）
    kk = str(k)
    if kk == "5":
      return (BASE_RANK_VALUE["5"], 5)
    return (BASE_RANK_VALUE.get(kk, 0), 5)

  if t == "ThreePair":
    # key 是最小点数，比较时以最大点数为准（跨度 2）
    kk = str(k)
    # 特例：A23（三连对中 A 与 2 相连视为最小）
    if kk == "A":
      return (BASE_RANK_VALUE["3"], 6)
    return (BASE_RANK_VALUE.get(kk, 0) + 2, 6)

  if t == "TwoTrips":
    # key 是最小点数，比较时以最大点数为准（跨度 1）
    kk = str(k)
    # 特例：A2（钢板中 A 与 2 相连视为最小）
    if kk == "A":
      return (BASE_RANK_VALUE["2"], 6)
    return (BASE_RANK_VALUE.get(kk, 0) + 1, 6)

  # 兜底：回退到 key
  return (rank_value(str(k), cur_rank), ln)


def _resolve_action_with_adviser(
  cards: List[Card],
  *,
  cur_rank: str,
  greater_action: Optional[Action] = None,
  tk_parent=None,
  allow_interactive: bool = True,
) -> Action:
  """Resolve an input card list into a single declared Action.

  - 无参谋：等价于 classify_action(cards)
  - 有参谋：枚举参谋补法；
      - 0 种：拒绝
      - 1 种：采用
      - 多种：弹窗/终端提示选择
  """
  adviser = _adviser_card(cur_rank)
  adv_idx = [i for i, c in enumerate(cards) if c == adviser]
  if not adv_idx:
    return classify_action(cards, cur_rank)

  # 枚举参谋->任意非 Joker 牌 的替代（点数+花色均可变）。
  options: List[Action] = []
  seen = set()

  for cards_as in itertools.product(ALL_NON_JOKER_CARDS, repeat=len(adv_idx)):
    virtual = list(cards)
    for j, c_as in enumerate(cards_as):
      virtual[adv_idx[j]] = str(c_as)
    try:
      act_v = classify_action(virtual, cur_rank)
    except Exception:
      continue

    # 用虚拟牌面去重：同一声明动作 + 同一虚拟牌面视为同一种补法
    sig = (act_v[0], act_v[1], tuple(sorted(virtual)))
    if sig in seen:
      continue
    seen.add(sig)

    act = [act_v[0], act_v[1], list(cards)]
    # 记录参谋补法（记录参谋变成的牌面，含花色）
    detail: Dict[str, Any] = {}
    # 参谋按本身牌面（H+cur_rank）不额外标注；只有发生替换时才标注。
    adv_as_cards = [c for c in cards_as if str(c) != adviser]
    if adv_as_cards:
      detail["adviser_as"] = list(adv_as_cards)
    detail["virtual_ranks"] = [c[-1] for c in virtual]
    detail["virtual_cards"] = list(virtual)
    if detail:
      act.append(detail)
    options.append(act)

  if not options:
    raise ValueError("包含参谋，但不存在任何合法补法。")

  # 跟牌场景：只保留能压过当前牌的补法，避免“先选参谋方案，后提示压不过”。
  if greater_action and isinstance(greater_action, list) and greater_action[0] != "PASS":
    beatable = [a for a in options if beats(a, greater_action, cur_rank)]
    if beatable:
      options = beatable
    else:
      raise ValueError("这手牌无论参谋怎么当，都压不过当前牌（需要更大的同类牌型或炸弹/同花顺）。")

  if len(options) == 1 or not allow_interactive:
    return options[0]

  # 多解：让用户选择。
  try:
    import tkinter as tk  # noqa: F401
    from tkinter import simpledialog

    lines = []
    for i, a in enumerate(options, 1):
      d = _action_detail(a)
      adv = d.get("adviser_as")
      adv_s = ""
      if adv:
        if isinstance(adv, (list, tuple)):
          def _disp(x: Any) -> str:
            s = str(x)
            try:
              if (
                len(s) >= 2
                and s[0] in {"S", "H", "C", "D"}
                and s[-1] in set(RANKS_NO_JOKER)
                and s not in set(RANKS_NO_JOKER)
              ):
                return format_card_for_human(s)
            except Exception:
              pass
            return s

          adv_s = f" 参谋当:{' '.join([_disp(x) for x in adv])}"
        elif isinstance(adv, dict):
          adv_s = " 参谋当:" + " ".join([f"{k}×{v}" if int(v) > 1 else str(k) for k, v in adv.items()])
      lines.append(f"{i}) {format_action_for_human(a, cur_rank)}{adv_s}")

    msg = "这手牌包含参谋，存在多种合法补法，请选择一种：\n\n" + "\n".join(lines)
    pick = simpledialog.askinteger(
      "选择参谋补法",
      msg,
      minvalue=1,
      maxvalue=len(options),
      parent=tk_parent,
    )
    if not pick:
      raise ValueError("已取消选择参谋补法。")
    return options[int(pick) - 1]
  except Exception:
    # Tk 不可用：回退到终端选择。
    print("这手牌包含参谋，存在多种合法补法，请选择：")
    for i, a in enumerate(options, 1):
      print(f"  {i}) {format_action_for_human(a, cur_rank)}")
    while True:
      s = input(f"请输入 1-{len(options)}（空=取消）：").strip()
      if not s:
        raise ValueError("已取消选择参谋补法。")
      try:
        v = int(s)
      except Exception:
        continue
      if 1 <= v <= len(options):
        return options[v - 1]


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
    ordered = sorted(counts.keys(), key=lambda r: BASE_RANK_VALUE.get(r, 99))
    # 按点数检查是否连续
    vals = [BASE_RANK_VALUE[r] for r in ordered]
    # A23：A 与 2 相连视为最小
    if set(vals) == {14, 2, 3}:
      return ["ThreePair", "A", cards]
    if vals[0] + 1 == vals[1] and vals[1] + 1 == vals[2]:
      return ["ThreePair", ordered[0], cards]

  # 钢板（2 个连续三张）
  if n == 6 and sorted(counts.values()) == [3, 3]:
    ordered = sorted(counts.keys(), key=lambda r: BASE_RANK_VALUE.get(r, 99))
    vals = [BASE_RANK_VALUE[r] for r in ordered]
    # A2：A 与 2 相连视为最小
    if set(vals) == {14, 2}:
      return ["TwoTrips", "A", cards]
    if vals[0] + 1 == vals[1]:
      return ["TwoTrips", ordered[0], cards]

  # 顺子（长度 5，允许 A2345）
  if n == 5 and all(c == 1 for c in counts.values()):
    ordered = sorted(counts.keys(), key=lambda r: BASE_RANK_VALUE.get(r, 99))
    vals = [BASE_RANK_VALUE[r] for r in ordered]
    suits = [c[0] for c in cards]
    # A2345
    if set(vals) == {14, 2, 3, 4, 5}:
      if len(set(suits)) == 1:
        return ["StraightFlush", "5", cards]
      return ["Straight", "5", cards]
    vals_sorted = sorted(vals)
    if (
      vals_sorted[0] + 1 == vals_sorted[1]
      and vals_sorted[1] + 1 == vals_sorted[2]
      and vals_sorted[2] + 1 == vals_sorted[3]
      and vals_sorted[3] + 1 == vals_sorted[4]
    ):
      # key 取最高点数字符（尽力而为）
      high_val = vals_sorted[-1]
      inv = {v: k for k, v in BASE_RANK_VALUE.items() if k in list("23456789TJQKA")}
      key = inv.get(high_val, ordered[-1])
      if len(set(suits)) == 1:
        return ["StraightFlush", key, cards]
      return ["Straight", key, cards]

  raise ValueError(
    "无法识别牌型（仅支持: PASS/Single/Pair/Trips/Bomb/Straight/StraightFlush/ThreeWithTwo/ThreePair/TwoTrips）"
  )


def beats(action: Action, greater_action: Optional[Action], cur_rank: str) -> bool:
  if greater_action is None:
    return action[0] != "PASS"

  if action[0] == "PASS":
    return True

  if greater_action[0] == "PASS":
    return action[0] != "PASS"

  def _is_bomb_like(a: Action) -> bool:
    try:
      return a[0] in {"Bomb", "StraightFlush"}
    except Exception:
      return False

  def _bomb_strength(a: Action) -> Tuple[int, int]:
    # 同花顺视为“5.5 炸”：大于五炸，小于六炸。
    t = a[0]
    ln = len(a[2]) if isinstance(a, list) and len(a) >= 3 else 0
    if t == "StraightFlush":
      size_rank = 55
      key, _ = _action_key_from_action(a, cur_rank)
      return (size_rank, int(key))
    # 普通炸弹：按张数排序
    size_rank = int(ln) * 10
    key, _ = _action_key_from_action(a, cur_rank)
    return (size_rank, int(key))

  # 炸弹类压过所有非炸弹类。
  if _is_bomb_like(action) and not _is_bomb_like(greater_action):
    return True
  if _is_bomb_like(action) and _is_bomb_like(greater_action):
    return _bomb_strength(action) > _bomb_strength(greater_action)

  if action[0] != greater_action[0]:
    return False

  a_key, a_len = _action_key_from_action(action, cur_rank)
  g_key, g_len = _action_key_from_action(greater_action, cur_rank)

  # 同类型必须长度一致。
  if a_len != g_len:
    return False

  return a_key > g_key


def generate_actions_from_hand(hand: List[Card], cur_rank: str) -> List[Action]:
  # 遵循客户端动作表示：[type, key, cards]。
  # 当包含参谋（红桃级牌）且发生补牌时，会在 action[3] 追加补法信息。
  adviser = _adviser_card(cur_rank)
  advisers = [c for c in hand if c == adviser]
  w = len(advisers)
  others = [c for c in hand if c != adviser]

  by_rank: Dict[str, List[Card]] = {}
  for c in others:
    by_rank.setdefault(c[-1], []).append(c)
  for v in by_rank.values():
    v.sort()

  out: List[Action] = []
  seen = set()

  def push(t: str, k: str, cards: List[Card], detail: Optional[Dict[str, Any]] = None):
    sig = (t, k, tuple(sorted(cards)))
    if sig in seen:
      return
    seen.add(sig)
    a: Action = [t, k, cards]
    if detail:
      a.append(detail)
    out.append(a)

  # 单张：去重即可（两副牌同牌面视为等价选择）
  for c in sorted(set(hand)):
    push("Single", c[-1], [c])

  # 同点数：对子/三张/炸弹。Joker 不允许参谋补。
  ranks_in_hand = sorted(set([c[-1] for c in others] + ["B", "R"]))
  for r in ranks_in_hand:
    base_cards = by_rank.get(r, [])
    can_use_adv = r in RANKS_NO_JOKER

    for need, typ in [(2, "Pair"), (3, "Trips")]:
      max_total = len(base_cards) + (w if can_use_adv else 0)
      if max_total < need:
        continue
      for adv_use in range(0, min(w if can_use_adv else 0, need) + 1):
        take = need - adv_use
        if len(base_cards) < take:
          continue
        cards = base_cards[:take] + advisers[:adv_use]
        detail = None
        if adv_use > 0 and r != cur_rank:
          detail = {"adviser_as": [r] * adv_use}
        if adv_use > 0:
          d = detail if detail is not None else {}
          d["virtual_ranks"] = [r] * len(cards)
          detail = d
        push(typ, r, cards, detail)

    max_total = len(base_cards) + (w if can_use_adv else 0)
    if max_total >= 4:
      for size in range(4, max_total + 1):
        for adv_use in range(0, min(w if can_use_adv else 0, size) + 1):
          take = size - adv_use
          if len(base_cards) < take:
            continue
          cards = base_cards[:take] + advisers[:adv_use]
          detail = None
          if adv_use > 0 and r != cur_rank:
            detail = {"adviser_as": [r] * adv_use}
          if adv_use > 0:
            d = detail if detail is not None else {}
            d["virtual_ranks"] = [r] * len(cards)
            detail = d
          push("Bomb", r, cards, detail)

  # 三带二：参谋可分配到三张或对子。
  for trip_r in RANKS_NO_JOKER:
    for pair_r in RANKS_NO_JOKER:
      if pair_r == trip_r:
        continue
      trip_cards = by_rank.get(trip_r, [])
      pair_cards = by_rank.get(pair_r, [])
      for adv_t in range(0, min(w, 3) + 1):
        for adv_p in range(0, min(w - adv_t, 2) + 1):
          if len(trip_cards) < 3 - adv_t:
            continue
          if len(pair_cards) < 2 - adv_p:
            continue
          cards = (
            trip_cards[: (3 - adv_t)]
            + advisers[:adv_t]
            + pair_cards[: (2 - adv_p)]
            + advisers[adv_t : adv_t + adv_p]
          )
          adv_as: List[str] = []
          if adv_t and trip_r != cur_rank:
            adv_as += [trip_r] * adv_t
          if adv_p and pair_r != cur_rank:
            adv_as += [pair_r] * adv_p
          detail = {"adviser_as": adv_as} if adv_as else {}
          if adv_t or adv_p:
            # cards 的顺序与拼接一致：先三张再对子
            detail["virtual_ranks"] = [trip_r] * 3 + [pair_r] * 2
          if not detail:
            detail = None
          push("ThreeWithTwo", trip_r, list(cards), detail)

  # 顺子（长度 5，允许 A2345）：参谋既可补缺失，也可替换已有点数。
  base_order = list("23456789TJQKA")
  seqs: List[List[str]] = []
  for i in range(0, len(base_order) - 4):
    seqs.append(base_order[i : i + 5])
  seqs.append(["A", "2", "3", "4", "5"])  # 特例

  for ranks in seqs:
    missing = [r for r in ranks if len(by_rank.get(r, [])) == 0]
    if len(missing) > w:
      continue
    for adv_use in range(len(missing), min(w, 5) + 1):
      for adv_pos in itertools.combinations(range(5), adv_use):
        # 必须覆盖缺失点数
        ok = True
        for r in missing:
          if ranks.index(r) not in adv_pos:
            ok = False
            break
        if not ok:
          continue
        cards: List[Card] = []
        adv_as: List[str] = []
        vranks: List[str] = []
        adv_taken = 0
        for idx, r in enumerate(ranks):
          if idx in adv_pos:
            cards.append(advisers[adv_taken])
            if r != cur_rank:
              adv_as.append(r)
            adv_taken += 1
          else:
            cards.append(by_rank[r][0])
          vranks.append(r)
        key = "5" if set(ranks) == {"A", "2", "3", "4", "5"} else ranks[-1]
        detail = {"adviser_as": adv_as} if adv_as else {}
        if adv_use > 0:
          detail["virtual_ranks"] = vranks
        if not detail:
          detail = None
        push("Straight", key, cards, detail)

  # 同花顺（长度 5）：同顺子，但要求同花色；参谋可变花色，因此可用于任意花色的同花顺。
  for suit in ["S", "H", "C", "D"]:
    by_rank_suit: Dict[str, List[Card]] = {}
    for c in others:
      if c[0] != suit:
        continue
      by_rank_suit.setdefault(c[-1], []).append(c)
    for v in by_rank_suit.values():
      v.sort()

    suit_adv = w
    if suit_adv == 0 and not by_rank_suit:
      continue

    for ranks in seqs:
      missing = [r for r in ranks if len(by_rank_suit.get(r, [])) == 0]
      if len(missing) > suit_adv:
        continue
      for adv_use in range(len(missing), min(suit_adv, 5) + 1):
        for adv_pos in itertools.combinations(range(5), adv_use):
          ok = True
          for r in missing:
            if ranks.index(r) not in adv_pos:
              ok = False
              break
          if not ok:
            continue
          cards: List[Card] = []
          adv_as: List[str] = []
          vranks: List[str] = []
          vcards: List[str] = []
          adv_taken = 0
          for idx, r in enumerate(ranks):
            if idx in adv_pos:
              cards.append(advisers[adv_taken])
              # 记录参谋变成的目标牌面（含花色），用于无歧义展示。
              vcards.append(f"{suit}{r}")
              if f"H{cur_rank}" != f"{suit}{r}":
                adv_as.append(f"{suit}{r}")
              adv_taken += 1
            else:
              cards.append(by_rank_suit[r][0])
              vcards.append(by_rank_suit[r][0])
            vranks.append(r)
          key = "5" if set(ranks) == {"A", "2", "3", "4", "5"} else ranks[-1]
          detail = {"adviser_as": adv_as} if adv_as else {}
          if adv_use > 0:
            detail["virtual_ranks"] = vranks
            detail["virtual_cards"] = vcards
          if not detail:
            detail = None
          push("StraightFlush", key, cards, detail)

  # 三连对（3 个连续对子）：参谋可补/替换。
  ordered = sorted(RANKS_NO_JOKER, key=lambda r: BASE_RANK_VALUE.get(r, 99))
  for i in range(0, len(ordered) - 2):
    r0, r1, r2 = ordered[i], ordered[i + 1], ordered[i + 2]
    if not (
      BASE_RANK_VALUE.get(r0, 99) + 1 == BASE_RANK_VALUE.get(r1, 99)
      and BASE_RANK_VALUE.get(r1, 99) + 1 == BASE_RANK_VALUE.get(r2, 99)
    ):
      continue
    # 分配参谋到三对中（每对最多 2）
    for use0 in range(0, min(w, 2) + 1):
      for use1 in range(0, min(w - use0, 2) + 1):
        for use2 in range(0, min(w - use0 - use1, 2) + 1):
          use_total = use0 + use1 + use2
          if use_total > w:
            continue
          if len(by_rank.get(r0, [])) < 2 - use0:
            continue
          if len(by_rank.get(r1, [])) < 2 - use1:
            continue
          if len(by_rank.get(r2, [])) < 2 - use2:
            continue
          cards = (
            by_rank.get(r0, [])[: (2 - use0)]
            + advisers[:use0]
            + by_rank.get(r1, [])[: (2 - use1)]
            + advisers[use0 : use0 + use1]
            + by_rank.get(r2, [])[: (2 - use2)]
            + advisers[use0 + use1 : use0 + use1 + use2]
          )
          adv_as: List[str] = []
          if use0 and r0 != cur_rank:
            adv_as += [r0] * use0
          if use1 and r1 != cur_rank:
            adv_as += [r1] * use1
          if use2 and r2 != cur_rank:
            adv_as += [r2] * use2
          detail = {"adviser_as": adv_as} if adv_as else {}
          if use_total > 0:
            detail["virtual_ranks"] = [r0, r0, r1, r1, r2, r2]
          if not detail:
            detail = None
          push("ThreePair", r0, list(cards), detail)

  # 三连对特例：A23（A 与 2 相连视为最小）
  r0, r1, r2 = "A", "2", "3"
  for use0 in range(0, min(w, 2) + 1):
    for use1 in range(0, min(w - use0, 2) + 1):
      for use2 in range(0, min(w - use0 - use1, 2) + 1):
        use_total = use0 + use1 + use2
        if use_total > w:
          continue
        if len(by_rank.get(r0, [])) < 2 - use0:
          continue
        if len(by_rank.get(r1, [])) < 2 - use1:
          continue
        if len(by_rank.get(r2, [])) < 2 - use2:
          continue
        cards = (
          by_rank.get(r0, [])[: (2 - use0)]
          + advisers[:use0]
          + by_rank.get(r1, [])[: (2 - use1)]
          + advisers[use0 : use0 + use1]
          + by_rank.get(r2, [])[: (2 - use2)]
          + advisers[use0 + use1 : use0 + use1 + use2]
        )
        adv_as: List[str] = []
        if use0 and r0 != cur_rank:
          adv_as += [r0] * use0
        if use1 and r1 != cur_rank:
          adv_as += [r1] * use1
        if use2 and r2 != cur_rank:
          adv_as += [r2] * use2
        detail = {"adviser_as": adv_as} if adv_as else {}
        if use_total > 0:
          detail["virtual_ranks"] = [r0, r0, r1, r1, r2, r2]
        if not detail:
          detail = None
        push("ThreePair", "A", list(cards), detail)

  # 钢板（2 个连续三张）：参谋可补/替换。
  for i in range(0, len(ordered) - 1):
    r0, r1 = ordered[i], ordered[i + 1]
    if BASE_RANK_VALUE.get(r0, 99) + 1 != BASE_RANK_VALUE.get(r1, 99):
      continue
    for use0 in range(0, min(w, 3) + 1):
      for use1 in range(0, min(w - use0, 3) + 1):
        if len(by_rank.get(r0, [])) < 3 - use0:
          continue
        if len(by_rank.get(r1, [])) < 3 - use1:
          continue
        cards = (
          by_rank.get(r0, [])[: (3 - use0)]
          + advisers[:use0]
          + by_rank.get(r1, [])[: (3 - use1)]
          + advisers[use0 : use0 + use1]
        )
        adv_as: List[str] = []
        if use0 and r0 != cur_rank:
          adv_as += [r0] * use0
        if use1 and r1 != cur_rank:
          adv_as += [r1] * use1
        detail = {"adviser_as": adv_as} if adv_as else {}
        if (use0 + use1) > 0:
          detail["virtual_ranks"] = [r0, r0, r0, r1, r1, r1]
        if not detail:
          detail = None
        push("TwoTrips", r0, list(cards), detail)

  # 钢板特例：A2（A 与 2 相连视为最小）
  r0, r1 = "A", "2"
  for use0 in range(0, min(w, 3) + 1):
    for use1 in range(0, min(w - use0, 3) + 1):
      if len(by_rank.get(r0, [])) < 3 - use0:
        continue
      if len(by_rank.get(r1, [])) < 3 - use1:
        continue
      cards = (
        by_rank.get(r0, [])[: (3 - use0)]
        + advisers[:use0]
        + by_rank.get(r1, [])[: (3 - use1)]
        + advisers[use0 : use0 + use1]
      )
      adv_as: List[str] = []
      if use0 and r0 != cur_rank:
        adv_as += [r0] * use0
      if use1 and r1 != cur_rank:
        adv_as += [r1] * use1
      detail = {"adviser_as": adv_as} if adv_as else {}
      if (use0 + use1) > 0:
        detail["virtual_ranks"] = [r0, r0, r0, r1, r1, r1]
      if not detail:
        detail = None
      push("TwoTrips", "A", list(cards), detail)

  # 稳定排序（用声明动作的 key/长度）
  def sort_key(a: Action) -> Tuple[int, int, int, str]:
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
    k, ln = _action_key_from_action(a, cur_rank)
    return (type_order, ln, k, str(a[1]))

  out.sort(key=sort_key)
  return out


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
    start_pos: int,
  ):
    self.cur_rank = cur_rank
    self.self_rank = self_rank
    self.oppo_rank = oppo_rank
    self.ai_hand = ai_hand[:]  # 座位 seat0

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

  def reset_for_new_game(self, *, ai_hand: List[Card], start_pos: int):
    # 重置对局状态（保留连接）。
    self.ai_hand = ai_hand[:]
    self.played_cards = Counter()
    self.human_remaining = {1: 27, 2: 27, 3: 27}
    self.current_pos = int(start_pos)
    self.greater_action = None
    self.greater_pos = None
    self.passes_since_play = 0

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
        prompt = (
          f"Seat{seat}（剩余 {remaining} 张）跟牌（可 PASS）。当前牌: {format_action_for_human(self.greater_action, self.cur_rank)}："
        )

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
        details = ", ".join([f"{format_card_for_human(c)} need={n} avail={a}" for c, n, a in violated])
        print(f"出牌包含不可用的牌（与 AI 手牌冲突/已出过/超出两副牌库存）：{details}")
        continue

      try:
        act = _resolve_action_with_adviser(
          cards,
          cur_rank=self.cur_rank,
          greater_action=self.greater_action,
          allow_interactive=True,
        )
      except Exception as e:
        print(f"无效输入: {e}")
        continue

      if not beats(act, self.greater_action, self.cur_rank):
        print("不压过当前牌（或牌型不匹配）。")
        continue

      return act

  async def run_game_loop(self) -> str:
    # 当前实现：只要求 seat0 连接即可开局。
    if 0 not in self.seat_conns:
      raise RuntimeError("需要 seat0 连接到 /game/client0")

    # 连接后发送 beginning。
    await self.send_beginning(0)

    stopped_early = False
    while True:

      if _RESTART_FULL_EVENT.is_set():
        return "restart_full"

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

        if act and isinstance(act, list) and act[0] == "RESTART_FULL":
          return "restart_full"
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

      # 结束条件：
      # - AI 出完牌
      # - AI 没出完，但敌方已出完（seat1 和 seat3 都出完）
      if int(self.human_remaining.get(1, 27)) <= 0 and int(self.human_remaining.get(3, 27)) <= 0:
        print("敌方（Seat1/Seat3）已出完牌（实验服将停止）。")
        stopped_early = True
        break

    return "stopped" if stopped_early else "finished"


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
  try:
    import websockets  # type: ignore
  except Exception as e:
    raise SystemExit(
      "缺少依赖 websockets，无法启动实验服。\n"
      "请安装后重试：\n"
      "  python -m pip install --user websockets\n\n"
      f"错误：{type(e).__name__}: {e}"
    )

  # 首局 AI 手牌：允许从参数/GUI 输入；后续“重新开局”默认随机发牌。
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
    start_pos=getattr(args, "start_pos", 0),
  )

  repo_root = os.path.abspath(_THIS_DIR)
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
          torch_models = (
            glob.glob(os.path.join(torch_models_dir, "ppo*.pth")) if os.path.isdir(torch_models_dir) else []
          )

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
          if _RESTART_FULL_EVENT.is_set():
            return "restart_full"
          await asyncio.sleep(0.1)

        status = await server.run_game_loop()
        if status == "restart_full":
          return "restart_full"
    except asyncio.CancelledError:
      # asyncio.run() 期间 Ctrl+C 会取消挂起的 await；这里安静退出。
      pass
  finally:
    for p in child_procs:
      try:
        p.terminate()
      except Exception:
        pass

  return "stopped"


def build_arg_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="Mini danserver (route3): 指定 AI 起手牌 + 终端手动 3 座位")
  p.add_argument("--host", default="127.0.0.1")
  p.add_argument("--port", type=int, default=23456)

  # 级牌/段位保持为字符串，兼容 torch 客户端。
  # 根据 UX 约定，curRank 与 AI 手牌在启动时交互输入，因此不强制要求命令行参数。

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
    while True:
      _RESTART_FULL_EVENT.clear()
      args = build_arg_parser().parse_args()
      # 交互输入
      args.cur_rank = _prompt_cur_rank("2")
      args.start_pos = _prompt_start_pos(0)
      args.self_rank = "2"
      args.oppo_rank = "2"
      args.ai_hand = _prompt_ai_hand(args.cur_rank)
      args.spawn_ai = not args.no_spawn_ai

      status = asyncio.run(main_async(args))
      if status == "restart_full":
        continue
      break
  except (KeyboardInterrupt, asyncio.CancelledError, EOFError):
    print("已退出。")
    return


if __name__ == "__main__":
  main()
