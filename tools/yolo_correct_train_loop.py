"""Standalone correction -> train -> reload -> retry loop for card recognition.

This script is intentionally NOT part of the gameplay flow.

Scope (per requirement): only handle *classification mistakes* for already-detected boxes.
YOLO still provides bboxes; we only learn to map each bbox crop -> correct label.

Workflow:
    - Grab a screenshot, let user select ROI
    - Run tools/card_yolo_detect.py to get bboxes
    - Show numbered overlay + predicted labels
    - User enters corrections like: 3=♠A 7=xw 9=♦10
    - Save corrected crops into tmp/yolo_online_dataset
    - Train a lightweight classifier (numpy softmax) and save it
    - Reload classifier and retry
    - Repeat until user confirms all correct

Run:
    python yolo_correct_train_loop.py
"""

from __future__ import annotations

import json
import os
import sys
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.yolo_online_classifier import (
    SoftmaxModel,
    append_samples,
    init_model,
    load_class_names,
    load_dataset,
    load_model,
    predict_label,
    save_model,
    train_sgd,
)
_MODEL_DIR = os.path.join(_REPO_ROOT, "模型")

_DATASET_DIR = os.path.join(_REPO_ROOT, "tmp", "yolo_online_dataset")
_MODEL_PATH = os.path.join(_REPO_ROOT, "tmp", "yolo_online_classifier", "model.npz")


@dataclass(frozen=True)
class DetItem:
    label: str
    score: float
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2
    origin: str = "yolo"  # 'yolo' or 'manual'


def _card_label_to_display(label: str) -> str:
    # Convert coco.names label into a more user-friendly token.
    if label == "dw":
        return "小JOKER"
    if label == "xw":
        return "大JOKER"

    suit_map = {"1": "♠", "2": "♥", "3": "♣", "4": "♦"}
    if label and label[0] in suit_map:
        s = suit_map[label[0]]
        r = label[1:]
        return f"{s}{r}"
    return label


def _parse_user_card(text: str) -> Optional[str]:
    """Parse user input token into coco.names label.

    Accepts:
      - dw/xw
      - 小JOKER/大JOKER
      - ♠A ♥10 ♣K ♦2
      - SA/HA/CA/DA (legacy)
      - S10/H10/C10/D10
    """

    t = (text or "").strip()
    if not t:
        return None

    t = t.replace(" ", "")
    if t.lower() in {"dw", "xw"}:
        return t.lower()
    if t in {"小JOKER", "小王", "小鬼"}:
        return "dw"
    if t in {"大JOKER", "大王", "大鬼"}:
        return "xw"

    sym_to_suit = {"♠": "1", "♥": "2", "♣": "3", "♦": "4"}
    if t and t[0] in sym_to_suit:
        suit = sym_to_suit[t[0]]
        rank = t[1:]
        return suit + rank

    letter_to_suit = {"S": "1", "H": "2", "C": "3", "D": "4"}
    if t and t[0].upper() in letter_to_suit:
        suit = letter_to_suit[t[0].upper()]
        rank = t[1:]
        return suit + rank

    return None


def parse_corrections(text: str, *, max_id: int) -> Dict[int, str]:
    """Parse corrections like: '3=♠A 7=xw 9=♦10' -> {2:'1A', 6:'xw', 8:'410'}"""
    out: Dict[int, str] = {}
    raw = (text or "").strip()
    if not raw:
        return out

    parts = raw.replace("，", " ").replace(",", " ").split()
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k.isdigit():
            continue
        idx1 = int(k)
        if idx1 < 1 or idx1 > int(max_id):
            continue
        lab = _parse_user_card(v)
        if not lab:
            continue
        out[idx1 - 1] = lab
    return out


def run_yolo_on_image(image_path: str) -> List[DetItem]:
    script = os.path.join(_REPO_ROOT, "tools", "card_yolo_detect.py")
    if not os.path.exists(script):
        raise FileNotFoundError(script)

    with tempfile.TemporaryDirectory(prefix="yolo_loop_") as td:
        out_json = os.path.join(td, "out.json")
        cmd = [
            os.sys.executable,
            script,
            "--image",
            image_path,
            "--model-dir",
            _MODEL_DIR,
            "--out",
            out_json,
        ]
        cp = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"YOLO failed: code={cp.returncode}\n{cp.stdout}\n{cp.stderr}")

        payload = json.loads(open(out_json, "r", encoding="utf-8").read())

    items: List[DetItem] = []
    for d in payload:
        bbox = d.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        items.append(
            DetItem(
                label=str(d.get("label") or ""),
                score=float(d.get("score", 0.0)),
                bbox=(x1, y1, x2, y2),
                origin="yolo",
            )
        )

    items.sort(key=lambda it: (-it.score, it.bbox[1], it.bbox[0]))
    return items


def draw_overlay(roi_bgr: np.ndarray, items: List[Tuple]) -> "object":
    """Return a PIL Image with numbered boxes.

    items: [(det, ...), ...]  # only det.bbox is used; extra fields are ignored.
    """
    from PIL import Image, ImageDraw, ImageFont

    rgb = roi_bgr[:, :, ::-1]
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for i, it in enumerate(items, start=1):
        det = it[0]
        x1, y1, x2, y2 = det.bbox
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
        # Only draw the id on the image; detailed predictions are listed in the text area below.
        txt = f"{i}"
        ty = max(0, y1 - 18)
        draw.text((x1 + 2, ty), txt, fill=(255, 0, 0), font=font)

    return img


def _clamp_box(box: Tuple[int, int, int, int], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _add_boxes_dialog(root, roi_pil, items: List[DetItem]) -> List[DetItem]:
    """Let user draw new boxes on the ROI image.

    Returns a list of newly added DetItem (origin='manual').
    """
    import tkinter as tk
    from tkinter import simpledialog
    from PIL import ImageTk

    img = roi_pil
    w, h = img.size

    # Scale down for usability (avoid giant windows).
    max_w, max_h = 1200, 800
    scale = min(1.0, max_w / max(1, w), max_h / max(1, h))
    disp = img
    if scale < 1.0:
        disp = img.resize((int(w * scale), int(h * scale)))

    top = tk.Toplevel(root)
    top.title("添加漏检框：拖拽画框，松开后输入牌面")

    imgtk = ImageTk.PhotoImage(disp)
    canvas = tk.Canvas(top, width=disp.size[0], height=disp.size[1])
    canvas.pack(padx=8, pady=8)
    canvas.create_image(0, 0, anchor="nw", image=imgtk)

    # Draw existing boxes in green.
    for i, it in enumerate(items, start=1):
        x1, y1, x2, y2 = it.bbox
        x1 = int(round(x1 * scale))
        y1 = int(round(y1 * scale))
        x2 = int(round(x2 * scale))
        y2 = int(round(y2 * scale))
        canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=2)
        canvas.create_text(x1 + 6, max(0, y1 - 10), text=str(i), fill="red", anchor="nw")

    state = {"x0": 0, "y0": 0, "rect": None}
    added: List[DetItem] = []

    def on_down(ev):
        state["x0"], state["y0"] = ev.x, ev.y
        if state["rect"] is not None:
            canvas.delete(state["rect"])
            state["rect"] = None

    def on_move(ev):
        x0, y0 = state["x0"], state["y0"]
        x1, y1 = ev.x, ev.y
        if state["rect"] is not None:
            canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(x0, y0, x1, y1, outline="red", width=2)

    def on_up(ev):
        x0, y0 = state["x0"], state["y0"]
        x1, y1 = ev.x, ev.y
        xa, xb = sorted([x0, x1])
        ya, yb = sorted([y0, y1])
        if xb - xa < 5 or yb - ya < 5:
            return

        # Map back to original image coords.
        ox1 = int(round(xa / scale))
        oy1 = int(round(ya / scale))
        ox2 = int(round(xb / scale))
        oy2 = int(round(yb / scale))
        box = _clamp_box((ox1, oy1, ox2, oy2), w, h)
        if box is None:
            return

        s = simpledialog.askstring("新框牌面", "输入该框的牌面：例如 ♠A / ♦10 / xw / dw", parent=top)
        lab = _parse_user_card(s or "")
        if not lab:
            return

        added.append(DetItem(label=str(lab), score=1.0, bbox=box, origin="manual"))

        # Paint it and keep allow adding more.
        dx1 = int(round(box[0] * scale))
        dy1 = int(round(box[1] * scale))
        dx2 = int(round(box[2] * scale))
        dy2 = int(round(box[3] * scale))
        canvas.create_rectangle(dx1, dy1, dx2, dy2, outline="cyan", width=2)
        canvas.create_text(dx1 + 6, max(0, dy1 - 10), text=f"+{len(added)}", fill="cyan", anchor="nw")

    canvas.bind("<ButtonPress-1>", on_down)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<ButtonRelease-1>", on_up)

    btns = tk.Frame(top)
    btns.pack(pady=(0, 8))

    def on_done():
        top.destroy()

    tk.Button(btns, text="完成", command=on_done).pack(side=tk.LEFT, padx=8)
    tk.Button(btns, text="取消", command=on_done).pack(side=tk.LEFT, padx=8)

    top.grab_set()
    top.wait_window()
    return added


class ReviewDialog:
    def __init__(self, root, overlay_img, lines: List[str], *, on_add_boxes=None):
        import tkinter as tk
        from PIL import ImageTk

        self._tk = tk
        self._top = tk.Toplevel(root)
        self._top.title("牌识别纠错（独立流程）")

        self._imgtk = ImageTk.PhotoImage(overlay_img)
        tk.Label(self._top, image=self._imgtk).pack(side=tk.TOP, padx=8, pady=8)

        msg = "\n".join(lines)
        text = tk.Text(self._top, height=min(18, max(6, len(lines) + 1)), width=60)
        text.insert("1.0", msg)
        text.configure(state="disabled")
        text.pack(side=tk.TOP, padx=8)

        tk.Label(self._top, text="纠错输入：例如 3=♠A 7=xw  (留空表示都正确)").pack(side=tk.TOP, padx=8, pady=(8, 0))
        self._entry = tk.Entry(self._top, width=60)
        self._entry.pack(side=tk.TOP, padx=8, pady=6)

        self._result: Optional[str] = None

        btns = tk.Frame(self._top)
        btns.pack(side=tk.TOP, padx=8, pady=(0, 10))

        def on_train_retry():
            self._result = self._entry.get()
            self._top.destroy()

        def on_done():
            # Treat as "back without training".
            self._result = ""
            self._top.destroy()

        if on_add_boxes is not None:
            tk.Button(btns, text="添加新框", command=on_add_boxes).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="确认并训练", command=on_train_retry).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="返回（不训练）", command=on_done).pack(side=tk.LEFT, padx=6)

        self._top.grab_set()
        self._top.protocol("WM_DELETE_WINDOW", on_done)

    def run(self) -> str:
        self._top.wait_window()
        return str(self._result or "")


def _select_roi_from_screen():
    """Deprecated: kept for compatibility; use _select_roi_from_screen_with_root."""
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    try:
        return _select_roi_from_screen_with_root(root)
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def _select_roi_from_screen_with_root(root):
    import tkinter as tk
    from PIL import ImageGrab, ImageTk

    # Hide the start window before capturing, so the UI won't show up in the screenshot.
    try:
        root.withdraw()
        root.update()
        time.sleep(0.12)
    except Exception:
        pass

    shot = ImageGrab.grab()

    top = tk.Toplevel(root)
    top.title("框选识别区域")

    imgtk = ImageTk.PhotoImage(shot)
    canvas = tk.Canvas(top, width=shot.width, height=shot.height)
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=imgtk)

    state = {"x0": 0, "y0": 0, "rect": None, "done": False, "bbox": None}

    def on_down(ev):
        state["x0"], state["y0"] = ev.x, ev.y
        if state["rect"] is not None:
            canvas.delete(state["rect"])
            state["rect"] = None

    def on_move(ev):
        x0, y0 = state["x0"], state["y0"]
        x1, y1 = ev.x, ev.y
        if state["rect"] is not None:
            canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(x0, y0, x1, y1, outline="red", width=2)

    def on_up(ev):
        x0, y0 = state["x0"], state["y0"]
        x1, y1 = ev.x, ev.y
        xa, xb = sorted([x0, x1])
        ya, yb = sorted([y0, y1])
        if xb - xa < 5 or yb - ya < 5:
            return
        state["bbox"] = (xa, ya, xb, yb)
        state["done"] = True
        top.destroy()

    canvas.bind("<ButtonPress-1>", on_down)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<ButtonRelease-1>", on_up)

    top.grab_set()
    top.wait_window()

    bbox = state.get("bbox")
    # Keep root alive for the rest of the flow.
    return shot, bbox


def _crop_and_save(shot, bbox) -> str:
    crop = shot.crop(bbox)
    with tempfile.TemporaryDirectory(prefix="yolo_roi_") as td:
        path = os.path.join(td, "roi.png")
        crop.save(path)
        # We must return a stable path; copy out.
        out = os.path.join(_REPO_ROOT, "tmp", "yolo_online_last", "roi.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        crop.save(out)
        return out


def _ensure_dirs():
    os.makedirs(os.path.join(_DATASET_DIR, "crops"), exist_ok=True)
    os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)


def _save_corrected_crops(roi_path: str, items: List[DetItem], corr: Dict[int, str]) -> int:
    from PIL import Image

    _ensure_dirs()

    roi = Image.open(roi_path).convert("RGB")
    samples: List[dict] = []

    # Per requirement assumption: after the user correction, all boxes in this ROI are correct.
    # Therefore we treat this ROI as fully labeled: all detections are saved.
    labels_by_idx: Dict[int, str] = {i: str(items[i].label) for i in range(len(items))}
    for k, v in corr.items():
        try:
            labels_by_idx[int(k)] = str(v)
        except Exception:
            pass

    for idx0 in range(len(items)):
        lab = labels_by_idx.get(int(idx0))
        if not lab:
            continue

        det = items[int(idx0)]
        x1, y1, x2, y2 = det.bbox
        crop = roi.crop((x1, y1, x2, y2))

        sid = f"{int(time.time())}_{random.randint(1000, 9999)}_{idx0}"
        rel = os.path.join("crops", f"{sid}.png")
        abs_path = os.path.join(_DATASET_DIR, rel)
        crop.save(abs_path)

        samples.append(
            {
                "crop_path": rel.replace("\\", "/"),
                "label": lab,
                "source": {
                    "roi_image": os.path.relpath(roi_path, _REPO_ROOT).replace("\\", "/"),
                    "bbox": [x1, y1, x2, y2],
                    "score": float(det.score),
                    "yolo_label": det.label,
                    "det_index": int(idx0),
                    "origin": str(getattr(det, "origin", "yolo")),
                },
            }
        )

    if samples:
        append_samples(_DATASET_DIR, samples)
    return len(samples)


def _num_unique_labels_in_dataset() -> int:
    try:
        import json
        from collections import Counter

        path = os.path.join(_DATASET_DIR, "samples.jsonl")
        if not os.path.exists(path):
            return 0
        ctr = Counter()
        for ln in open(path, "r", encoding="utf-8"):
            s = ln.strip()
            if not s:
                continue
            try:
                d = json.loads(s)
            except Exception:
                continue
            lab = str(d.get("label") or "")
            if lab:
                ctr[lab] += 1
        return int(len(ctr))
    except Exception:
        return 0


def _predict_labels_for_items(
    roi_bgr: np.ndarray,
    items: List[DetItem],
    model: Optional[SoftmaxModel],
    classes: List[str],
):
    from PIL import Image

    roi_rgb = roi_bgr[:, :, ::-1]
    roi_img = Image.fromarray(roi_rgb)

    results = []
    for it in items:
        x1, y1, x2, y2 = it.bbox
        crop = roi_img.crop((x1, y1, x2, y2)).convert("L").resize((32, 32))
        x = (np.asarray(crop, dtype=np.float32) / 255.0).reshape(-1)
        if model is not None:
            pred, conf = predict_label(model, x.astype(np.float32))
            results.append((it, pred, float(conf), "model"))
        else:
            # Fallback to YOLO label.
            pred = it.label
            conf = float(it.score)
            results.append((it, pred, float(conf), "yolo"))

    return results


def main() -> int:
    if not os.path.isdir(_MODEL_DIR):
        raise FileNotFoundError(_MODEL_DIR)

    classes = load_class_names(_MODEL_DIR)

    import tkinter as tk

    root = tk.Tk()
    root.title("YOLO 纠错训练（独立流程）")

    # Outer loop state.
    state = {
        "model": load_model(_MODEL_PATH),
        "running": False,
    }

    frm = tk.Frame(root)
    frm.pack(padx=14, pady=14)

    tk.Label(frm, text="点击开始后：截屏 → 框选视窗 → 显示标注图 → 输入纠错").pack(anchor="w")

    btn_row = tk.Frame(frm)
    btn_row.pack(anchor="w", pady=(10, 0))

    start_btn = tk.Button(btn_row, text="开始")
    start_btn.pack(side=tk.LEFT)
    tk.Button(btn_row, text="退出", command=root.destroy).pack(side=tk.LEFT, padx=10)

    def run_one_round() -> None:
        if state.get("running"):
            return
        state["running"] = True
        try:
            # Each round re-grabs screen and re-runs.
            shot, bbox = _select_roi_from_screen_with_root(root)
            if not bbox:
                return

            roi_path = _crop_and_save(shot, bbox)

            items = run_yolo_on_image(roi_path)
            if not items:
                print("[yolo-loop] no detections in ROI; please reselect a clearer ROI")
                return

            import cv2

            bgr = cv2.imdecode(np.fromfile(roi_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            # Interactive review loop: allow add boxes + corrections, then confirm before training.
            while True:
                preds = _predict_labels_for_items(bgr, items, state.get("model"), classes)
                overlay = draw_overlay(bgr, preds)

                lines = []
                for i, (det, pred, pred_conf, src) in enumerate(preds, start=1):
                    yolo_show = "(manual)" if getattr(det, "origin", "yolo") == "manual" else _card_label_to_display(det.label)
                    lines.append(
                        f"{i}. pred={_card_label_to_display(pred)}  (src={src} pred_conf={pred_conf:.2f})  "
                        f"(yolo={yolo_show} yolo_score={det.score:.2f})"
                    )

                from PIL import Image

                roi_pil = Image.open(roi_path).convert("RGB")

                def on_add_boxes():
                    new_items = _add_boxes_dialog(root, roi_pil, items)
                    if new_items:
                        items.extend(list(new_items))

                dlg = ReviewDialog(root, overlay, lines, on_add_boxes=on_add_boxes)
                text = dlg.run()

                # Bring back the start window.
                try:
                    root.deiconify()
                    root.update()
                except Exception:
                    pass

                if not text.strip():
                    # User chose "返回（不训练）".
                    print("[yolo-loop] back to start without training")
                    return

                corr = parse_corrections(text, max_id=len(items))
                if not corr and text.strip():
                    # User clicked confirm but provided unparsable content: let them try again.
                    from tkinter import messagebox

                    messagebox.showwarning("纠错解析失败", "未能解析纠错输入，请按格式输入：3=♠A 7=xw")
                    continue

                # Training confirmation.
                from tkinter import messagebox

                msg = f"将以当前 ROI 的 {len(items)} 个框生成训练数据并训练。\n\n"
                msg += "请确认：所有框都已正确且无遗漏。\n\n继续训练？"
                if not messagebox.askyesno("训练确认", msg):
                    continue

                # Proceed to save/train.
                added = _save_corrected_crops(roi_path, items, corr)
                print(f"[yolo-loop] saved {added} labeled crops (full ROI)")

                # If dataset only has <2 unique labels, training a classifier is meaningless and will collapse.
                uniq = _num_unique_labels_in_dataset()
                if uniq < 2:
                    messagebox.showinfo(
                        "样本不足",
                        f"当前训练集只有 {uniq} 个不同牌面标签。\n\n"
                        "请再纠错/添加一张不同牌面的框后再训练（至少2类）。",
                    )
                    state["model"] = None
                    return

                X, y, cls = load_dataset(_DATASET_DIR, model_dir=_MODEL_DIR)
                model = state.get("model")
                if model is None or model.classes != cls:
                    model = init_model(dim=1024, classes=cls, seed=0)

                info = train_sgd(model, X, y, epochs=120, lr=0.25, l2=1e-4, batch_size=32, seed=0)
                save_model(_MODEL_PATH, model)
                print(f"[yolo-loop] trained model: n={int(info['n'])} loss={info['loss']:.4f} acc={info['acc']:.3f}")

                state["model"] = load_model(_MODEL_PATH)
                print("[yolo-loop] reloaded model; back to start")
                break

        finally:
            state["running"] = False

    start_btn.configure(command=run_one_round)

    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
