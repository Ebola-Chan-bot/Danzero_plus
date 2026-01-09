"""Detect playing cards in a screenshot using the provided YOLO (Darknet) model.

This repo includes a pre-trained model under:
  - 模型/model/detect.cfg
  - 模型/model/params.model          (Darknet weights, despite the extension)
  - 模型/model/data/coco.names       (54 classes: 52 cards + 2 jokers)

Notes:
- The shipped detect.cfg contains a stray token line ("Testing") that OpenCV's
  Darknet parser rejects. This script sanitizes the cfg on the fly.

Example:
    python tools/card_yolo_detect.py --image "C:/path/to/screenshot.png" --debug debug_yolo
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Det:
    label: str
    score: float
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2


def load_bgr(image_path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return img


def read_class_names(names_path: str) -> List[str]:
    with open(names_path, "r", encoding="utf-8", errors="ignore") as f:
        names = [ln.strip() for ln in f.readlines()]
    return [n for n in names if n]


def sanitize_darknet_cfg(cfg_path: str) -> str:
    """Return path to a temp cfg that OpenCV can parse."""
    lines = open(cfg_path, "r", encoding="utf-8", errors="ignore").read().splitlines()
    kept: List[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            kept.append(line)
            continue
        if s.startswith("#") or s.startswith("[") or ("=" in s):
            kept.append(line)
            continue
        # Drop stray tokens like "Testing" (Darknet tolerates them; OpenCV doesn't).
        continue

    fd, tmp = tempfile.mkstemp(suffix=".cfg")
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
    return tmp


def decode_label_to_rank_suit(label: str) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort decode for labels in coco.names.

    Label format in this model:
      - "12".."1A" : suit=1, rank=2..A
      - "110"      : suit=1, rank=10
      - similarly for suit 2/3/4
      - "dw", "xw" : jokers

    Suit mapping is not documented in the repo; we keep it symbolic by default.
    If you know the true mapping, adjust SUIT_MAP.
    """
    # User-provided mapping: 1♠ 2♥ 3♣ 4♦
    SUIT_MAP = {
        "1": "spades",
        "2": "hearts",
        "3": "clubs",
        "4": "diamonds",
    }

    if label in ("dw", "xw"):
        return ("joker", label)

    if not label:
        return (None, None)

    s0 = label[0]
    if s0 not in SUIT_MAP:
        return (None, None)

    suit = SUIT_MAP[s0]
    rest = label[1:]
    if rest == "10":
        rank = "10"
    elif rest in {"A", "K", "Q", "J"}:
        rank = rest
    elif rest.isdigit() and rest in {"2", "3", "4", "5", "6", "7", "8", "9"}:
        rank = rest
    else:
        rank = rest or None

    return (rank, suit)


def suit_to_symbol(suit: Optional[str]) -> str:
    return {
        "spades": "♠",
        "hearts": "♥",
        "clubs": "♣",
        "diamonds": "♦",
    }.get(str(suit), "")


def yolo_forward(
    net: cv2.dnn.Net,
    bgr: np.ndarray,
    input_size: int = 416,
) -> List[np.ndarray]:
    blob = cv2.dnn.blobFromImage(bgr, scalefactor=1 / 255.0, size=(input_size, input_size), swapRB=True, crop=False)
    net.setInput(blob)
    out_names = net.getUnconnectedOutLayersNames()
    return net.forward(out_names)


def collect_candidates(
    outs: List[np.ndarray],
    img_shape: Tuple[int, int, int],
    *,
    conf_th: float,
) -> Tuple[List[Tuple[int, int, int, int]], List[float], List[int]]:
    """Collect raw YOLO candidates (before NMS)."""
    h, w = img_shape[:2]

    boxes: List[Tuple[int, int, int, int]] = []
    scores: List[float] = []
    class_ids: List[int] = []

    for out in outs:
        # out shape: (N, 5 + num_classes)
        for det in out:
            obj = float(det[4])
            if obj <= 0:
                continue
            cls_scores = det[5:]
            cls_id = int(np.argmax(cls_scores))
            cls_conf = float(cls_scores[cls_id])
            score = obj * cls_conf
            if score < conf_th:
                continue

            cx, cy, bw, bh = (float(det[0]) * w, float(det[1]) * h, float(det[2]) * w, float(det[3]) * h)
            x1 = int(round(cx - bw / 2))
            y1 = int(round(cy - bh / 2))
            x2 = int(round(cx + bw / 2))
            y2 = int(round(cy + bh / 2))

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append((x1, y1, x2 - x1, y2 - y1))
            scores.append(score)
            class_ids.append(cls_id)

    return boxes, scores, class_ids


def nms_to_dets(
    boxes: List[Tuple[int, int, int, int]],
    scores: List[float],
    class_ids: List[int],
    class_names: List[str],
    *,
    conf_th: float,
    nms_th: float,
) -> List[Det]:
    if not boxes:
        return []

    idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_th, nms_th)
    if len(idxs) == 0:
        return []

    dets: List[Det] = []
    for i in idxs.flatten().tolist():
        x, y, bw, bh = boxes[i]
        x1, y1, x2, y2 = x, y, x + bw, y + bh
        cls_id = class_ids[i]
        label = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        dets.append(Det(label=label, score=float(scores[i]), bbox=(x1, y1, x2, y2)))

    dets.sort(key=lambda d: (-d.score, d.bbox[1], d.bbox[0]))
    return dets


def iter_tiles(
    bgr: np.ndarray,
    *,
    rows: int = 1,
    cols: int = 1,
    overlap: float = 0.15,
) -> List[Tuple[np.ndarray, int, int]]:
    """Split image into an overlapping grid of tiles.

    Returns a list of (tile_bgr, x0, y0) where x0/y0 are the top-left offsets.
    Tiling improves small-object recall because each tile is resized to the
    network input size.
    """

    rows = int(max(1, rows))
    cols = int(max(1, cols))
    overlap = float(max(0.0, min(0.49, overlap)))

    h, w = bgr.shape[:2]
    tile_w = int(np.ceil(w / cols))
    tile_h = int(np.ceil(h / rows))

    step_x = max(1, int(round(tile_w * (1.0 - overlap))))
    step_y = max(1, int(round(tile_h * (1.0 - overlap))))

    xs: List[int] = []
    ys: List[int] = []

    # Generate starts, ensure last tile reaches the end.
    x = 0
    while True:
        xs.append(x)
        if x + tile_w >= w:
            break
        x = min(w - tile_w, x + step_x)

    y = 0
    while True:
        ys.append(y)
        if y + tile_h >= h:
            break
        y = min(h - tile_h, y + step_y)

    tiles: List[Tuple[np.ndarray, int, int]] = []
    for y0 in ys:
        for x0 in xs:
            x1 = min(w, x0 + tile_w)
            y1 = min(h, y0 + tile_h)
            tile = bgr[y0:y1, x0:x1]
            tiles.append((tile, int(x0), int(y0)))

    return tiles


def postprocess(
    outs: List[np.ndarray],
    img_shape: Tuple[int, int, int],
    class_names: List[str],
    conf_th: float = 0.25,
    nms_th: float = 0.45,
) -> List[Det]:
    boxes, scores, class_ids = collect_candidates(outs, img_shape, conf_th=conf_th)
    return nms_to_dets(boxes, scores, class_ids, class_names, conf_th=conf_th, nms_th=nms_th)


def _iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-6)


def dedup_corner_dets(
    dets: List[Det],
    *,
    max_center_dist: float = 10.0,
    iou_th: float = 0.25,
) -> List[Det]:
    """Merge near-identical detections.

    This model often outputs multiple very-close boxes for the same corner index.
    We keep the highest-score one.
    """

    kept: List[Det] = []

    def center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = b
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    for d in sorted(dets, key=lambda x: -x.score):
        cx, cy = center(d.bbox)
        dup = False
        for k in kept:
            kx, ky = center(k.bbox)
            if d.label != k.label:
                continue
            # Be conservative: require BOTH close centers and high IoU.
            if (abs(cx - kx) + abs(cy - ky)) <= max_center_dist and _iou_xyxy(d.bbox, k.bbox) >= iou_th:
                dup = True
                break
        if not dup:
            kept.append(d)

    kept.sort(key=lambda d: (-d.score, d.bbox[1], d.bbox[0]))
    return kept


def dedup_by_location(
    dets: List[Det],
    *,
    max_center_dist: float = 10.0,
    iou_th: float = 0.35,
) -> List[Det]:
    """Deduplicate detections that likely refer to the same physical corner.

    Unlike `dedup_corner_dets`, this works across different labels too.
    It's useful when the model fires twice on the same card with slightly
    different boxes or even different predicted classes.
    """

    kept: List[Det] = []

    def center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = b
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    for d in sorted(dets, key=lambda x: -x.score):
        cx, cy = center(d.bbox)
        dup = False
        for k in kept:
            kx, ky = center(k.bbox)
            # Be conservative: only merge when it's almost certainly the same physical corner.
            if (abs(cx - kx) + abs(cy - ky)) <= max_center_dist and _iou_xyxy(d.bbox, k.bbox) >= iou_th:
                dup = True
                break
        if not dup:
            kept.append(d)

    kept.sort(key=lambda d: (-d.score, d.bbox[1], d.bbox[0]))
    return kept


def cap_per_label(
    dets: List[Det],
    max_per_label: int = 2,
    *,
    min_second_score_ratio: float = 0.0,
    min_second_score_abs: float = 0.0,
) -> List[Det]:
    """Keep at most N detections per class label (highest scores).

    Extra (optional) quality gate for the 2nd copy of a label:
    if the second detection is *much* weaker than the first, it's often a
    false duplicate on the same card corner. This gate is disabled by default.
    """

    max_per_label = int(max(1, max_per_label))
    min_second_score_ratio = float(max(0.0, min(1.0, min_second_score_ratio)))
    min_second_score_abs = float(max(0.0, min_second_score_abs))

    buckets: dict[str, List[Det]] = {}
    for d in dets:
        buckets.setdefault(d.label, []).append(d)

    kept: List[Det] = []
    for label, ds in buckets.items():
        ds_sorted = sorted(ds, key=lambda x: -x.score)
        if not ds_sorted:
            continue

        kept_label: List[Det] = [ds_sorted[0]]
        if max_per_label >= 2 and len(ds_sorted) >= 2:
            cand = ds_sorted[1]
            best = ds_sorted[0]
            if min_second_score_abs > 0.0 and cand.score < min_second_score_abs:
                pass
            elif min_second_score_ratio > 0.0 and cand.score < best.score * min_second_score_ratio:
                pass
            else:
                kept_label.append(cand)

        kept.extend(kept_label[:max_per_label])

    kept.sort(key=lambda d: (-d.score, d.bbox[1], d.bbox[0]))
    return kept


def _bbox_center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _bbox_area(b: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = b
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def merge_dets_for_overlay(
    dets: List[Det],
    *,
    max_center_dist: float = 3.0,
    iou_th: float = 0.85,
    min_area_ratio: float = 0.70,
) -> List[List[Det]]:
    """Group detections that look like the *same* physical corner for drawing.

    Important: this ONLY affects visualization (overlay). We keep JSON / counts intact.
    We use very strict thresholds to avoid merging two adjacent different cards.
    """

    groups: List[List[Det]] = []
    for d in sorted(dets, key=lambda x: -x.score):
        cx, cy = _bbox_center(d.bbox)
        da = _bbox_area(d.bbox)
        placed = False

        for g in groups:
            rep = g[0]
            rx, ry = _bbox_center(rep.bbox)
            if (abs(cx - rx) + abs(cy - ry)) > max_center_dist:
                continue
            iou = _iou_xyxy(d.bbox, rep.bbox)
            if iou < iou_th:
                continue
            ra = _bbox_area(rep.bbox)
            if min(da, ra) / (max(da, ra) + 1e-6) < min_area_ratio:
                continue
            g.append(d)
            placed = True
            break

        if not placed:
            groups.append([d])

    # Keep groups stable-ish for drawing (by rep score, then position).
    groups.sort(key=lambda g: (-g[0].score, g[0].bbox[1], g[0].bbox[0]))
    return groups


def draw_overlay(
    bgr: np.ndarray,
    dets: List[Det],
    out_path: str,
    *,
    merge_for_overlay: bool = False,
    overlay_merge_dist: float = 3.0,
    overlay_merge_iou: float = 0.85,
    overlay_merge_area: float = 0.70,
    low_score_th: float = 0.0,
    font_size: int = 26,
    box_thickness: int = 2,
    main_bgr: Tuple[int, int, int] = (255, 0, 0),
) -> None:
    img = bgr.copy()

    low_score_th = float(max(0.0, low_score_th))
    font_size = int(max(10, font_size))
    box_thickness = int(max(1, box_thickness))
    main_bgr = (int(main_bgr[0]), int(main_bgr[1]), int(main_bgr[2]))
    low_dets = [d for d in dets if low_score_th > 0.0 and float(d.score) < low_score_th]
    main_dets = dets

    groups = (
        merge_dets_for_overlay(
            main_dets,
            max_center_dist=float(overlay_merge_dist),
            iou_th=float(overlay_merge_iou),
            min_area_ratio=float(overlay_merge_area),
        )
        if merge_for_overlay
        else [[d] for d in main_dets]
    )

    # Draw boxes with OpenCV first.
    for g in groups:
        rep = g[0]
        x1, y1, x2, y2 = rep.bbox
        # Main color for normal boxes.
        cv2.rectangle(img, (x1, y1), (x2, y2), main_bgr, box_thickness)

    # Draw low-confidence boxes on top in red (even if overlay-merge is on).
    for d in low_dets:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), box_thickness)

    # Draw text with Pillow to support Unicode suit symbols (♠♥♣♦).
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)

        # Pick a font that includes suit symbols.
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont
        font_paths = [
            r"C:\\Windows\\Fonts\\seguisym.ttf",  # Segoe UI Symbol
            r"C:\\Windows\\Fonts\\segoeui.ttf",
            r"C:\\Windows\\Fonts\\arial.ttf",
        ]
        font = ImageFont.load_default()
        for fp in font_paths:
            try:
                if os.path.exists(fp):
                    font = ImageFont.truetype(fp, font_size)
                    break
            except Exception:
                continue

        # Pillow uses RGB; our main_bgr is BGR.
        main_rgb = (main_bgr[2], main_bgr[1], main_bgr[0])

        for g in groups:
            rep = g[0]
            x1, y1, x2, y2 = rep.bbox
            rank, suit = decode_label_to_rank_suit(rep.label)
            base = f"{rank or rep.label}{suit_to_symbol(suit)} {rep.score:.2f}"
            if len(g) > 1:
                # Append short hint that multiple predictions overlap here.
                base += f" (+{len(g) - 1})"
            draw.text((x1, max(0, y1 - (font_size + 4))), base, fill=main_rgb, font=font)

        # Red labels for low-confidence boxes.
        for d in low_dets:
            x1, y1, x2, y2 = d.bbox
            rank, suit = decode_label_to_rank_suit(d.label)
            txt = f"LOW {rank or d.label}{suit_to_symbol(suit)} {d.score:.2f}"
            draw.text((x1, max(0, y1 - (font_size + 4))), txt, fill=(255, 0, 0), font=font)

        rgb2 = np.array(pil_img)
        img = cv2.cvtColor(rgb2, cv2.COLOR_RGB2BGR)
    except Exception:
        # Fallback: ASCII-only overlay (won't show suits, but keeps the tool usable).
        for g in groups:
            rep = g[0]
            x1, y1, x2, y2 = rep.bbox
            rank, suit = decode_label_to_rank_suit(rep.label)
            sym = suit_to_symbol(suit)
            safe = sym if sym and sym.isascii() else suit[:1].upper() if suit else ""
            txt = f"{rank or rep.label}{safe} {rep.score:.2f}"
            if len(g) > 1:
                txt += f"(+{len(g) - 1})"
            cv2.putText(
                img,
                txt,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.6, font_size / 30.0),
                main_bgr,
                box_thickness,
                cv2.LINE_AA,
            )

        for d in low_dets:
            x1, y1, x2, y2 = d.bbox
            rank, suit = decode_label_to_rank_suit(d.label)
            sym = suit_to_symbol(suit)
            safe = sym if sym and sym.isascii() else suit[:1].upper() if suit else ""
            txt = f"LOW {rank or d.label}{safe} {d.score:.2f}"
            cv2.putText(
                img,
                txt,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.6, font_size / 30.0),
                (0, 0, 255),
                box_thickness,
                cv2.LINE_AA,
            )

    cv2.imwrite(out_path, img)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model-dir", default=os.path.join("模型"))
    ap.add_argument(
        "--input-sizes",
        default="416",
        help="Comma-separated YOLO input sizes. Example: 416 or 416,608 (multi-scale, then NMS on union)",
    )
    ap.add_argument("--tile-rows", type=int, default=1, help="Tiled inference: number of rows (default: 1 = off)")
    ap.add_argument("--tile-cols", type=int, default=1, help="Tiled inference: number of cols (default: 1 = off)")
    ap.add_argument(
        "--tile-overlap",
        type=float,
        default=0.15,
        help="Tiled inference: overlap ratio in [0, 0.49] (default: 0.15)",
    )
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--dedup", action="store_true", help="Deduplicate near-identical corner detections")
    ap.add_argument("--center-dist", type=float, default=10.0, help="Dedup: max L1 center distance")
    ap.add_argument("--dedup-iou", type=float, default=0.25, help="Dedup: IoU threshold")
    ap.add_argument(
        "--dedup-location",
        action="store_true",
        help="Deduplicate by location even if labels differ (removes double boxes on same card)",
    )
    ap.add_argument("--location-dist", type=float, default=10.0, help="Location-dedup: max L1 center distance")
    ap.add_argument("--location-iou", type=float, default=0.35, help="Location-dedup: IoU threshold")
    ap.add_argument("--max-per-label", type=int, default=2, help="Keep at most N detections per label (default: 2)")
    ap.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="If >0, drop detections with score < this threshold (useful to remove obvious false positives)",
    )
    ap.add_argument(
        "--min-second-score-ratio",
        type=float,
        default=0.0,
        help="If >0, drop the 2nd detection of a label when score < best*ratio (helps remove weak duplicates)",
    )
    ap.add_argument(
        "--min-second-score",
        type=float,
        default=0.0,
        help="If >0, drop the 2nd detection of a label when score < this value (helps remove weak duplicates)",
    )
    ap.add_argument(
        "--overlay-merge",
        action="store_true",
        help="Only for overlay: merge near-identical overlapping boxes so you don't see double-rectangles",
    )
    ap.add_argument("--overlay-merge-dist", type=float, default=3.0, help="Overlay-merge: max L1 center distance")
    ap.add_argument("--overlay-merge-iou", type=float, default=0.85, help="Overlay-merge: IoU threshold")
    ap.add_argument(
        "--overlay-merge-area",
        type=float,
        default=0.70,
        help="Overlay-merge: min area similarity ratio (min/ max)",
    )
    ap.add_argument(
        "--overlay-low-score",
        type=float,
        default=0.0,
        help="If >0, draw detections with score < this threshold in red and also write suspicious.json in --debug",
    )
    ap.add_argument("--overlay-font-size", type=int, default=26, help="Overlay text font size (default: 26)")
    ap.add_argument("--overlay-thickness", type=int, default=2, help="Overlay box/text thickness (default: 2)")
    ap.add_argument(
        "--expected-count",
        type=int,
        default=0,
        help="If >0, validate final detection count equals this value (no trimming).",
    )
    ap.add_argument("--out", default="")
    ap.add_argument("--debug", default="")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    cfg_path = os.path.join(model_dir, "detect.cfg")
    weights_path = os.path.join(model_dir, "params.model")
    # Support both flattened layout (模型/coco.names) and old layout (模型/model/data/coco.names)
    names_path = os.path.join(model_dir, "coco.names")
    if not os.path.exists(names_path):
        names_path = os.path.join(model_dir, "data", "coco.names")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(cfg_path)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(weights_path)
    if not os.path.exists(names_path):
        raise FileNotFoundError(names_path)

    class_names = read_class_names(names_path)

    tmp_cfg = sanitize_darknet_cfg(cfg_path)
    try:
        net = cv2.dnn.readNetFromDarknet(tmp_cfg, weights_path)
    finally:
        try:
            os.remove(tmp_cfg)
        except Exception:
            pass

    bgr = load_bgr(args.image)

    sizes: List[int] = []
    for tok in str(args.input_sizes).split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            sizes.append(int(t))
        except Exception:
            raise ValueError(f"Invalid --input-sizes token: {t!r}")
    if not sizes:
        sizes = [416]

    tiles = iter_tiles(
        bgr,
        rows=int(args.tile_rows),
        cols=int(args.tile_cols),
        overlap=float(args.tile_overlap),
    )

    boxes_all: List[Tuple[int, int, int, int]] = []
    scores_all: List[float] = []
    class_ids_all: List[int] = []
    for tile_bgr, x0, y0 in tiles:
        for sz in sizes:
            outs = yolo_forward(net, tile_bgr, input_size=int(sz))
            boxes, scores, class_ids = collect_candidates(outs, tile_bgr.shape, conf_th=float(args.conf))
            # Translate tile-local boxes back to full-image coordinates.
            for (bx, by, bw, bh), sc, cid in zip(boxes, scores, class_ids):
                boxes_all.append((int(bx + x0), int(by + y0), int(bw), int(bh)))
                scores_all.append(float(sc))
                class_ids_all.append(int(cid))

    dets_raw = nms_to_dets(
        boxes_all,
        scores_all,
        class_ids_all,
        class_names,
        conf_th=float(args.conf),
        nms_th=float(args.nms),
    )
    dets_stage1 = (
        dedup_corner_dets(dets_raw, max_center_dist=float(args.center_dist), iou_th=float(args.dedup_iou))
        if bool(args.dedup)
        else dets_raw
    )
    dets_stage2 = cap_per_label(
        dets_stage1,
        max_per_label=int(args.max_per_label),
        min_second_score_ratio=float(args.min_second_score_ratio),
        min_second_score_abs=float(args.min_second_score),
    )
    dets_stage3 = (
        dedup_by_location(
            dets_stage2,
            max_center_dist=float(args.location_dist),
            iou_th=float(args.location_iou),
        )
        if bool(args.dedup_location)
        else dets_stage2
    )
    dets_stage4 = (
        [d for d in dets_stage3 if float(d.score) >= float(args.min_score)]
        if float(args.min_score) > 0.0
        else dets_stage3
    )
    dets = dets_stage4

    payload = []
    for d in dets:
        rank, suit = decode_label_to_rank_suit(d.label)
        payload.append(
            {
                "label": d.label,
                "rank": rank,
                "suit": suit,
                "score": round(float(d.score), 6),
                "bbox": list(map(int, d.bbox)),
            }
        )

    if args.debug:
        os.makedirs(args.debug, exist_ok=True)
        draw_overlay(
            bgr,
            dets,
            os.path.join(args.debug, "overlay.png"),
            merge_for_overlay=bool(args.overlay_merge),
            overlay_merge_dist=float(args.overlay_merge_dist),
            overlay_merge_iou=float(args.overlay_merge_iou),
            overlay_merge_area=float(args.overlay_merge_area),
            low_score_th=float(args.overlay_low_score),
            font_size=int(args.overlay_font_size),
            box_thickness=int(args.overlay_thickness),
            main_bgr=(255, 0, 0),
        )

        if float(args.overlay_low_score) > 0.0:
            low = [
                {
                    "label": d.label,
                    "score": round(float(d.score), 6),
                    "bbox": list(map(int, d.bbox)),
                    "rank": decode_label_to_rank_suit(d.label)[0],
                    "suit": decode_label_to_rank_suit(d.label)[1],
                }
                for d in dets
                if float(d.score) < float(args.overlay_low_score)
            ]
            with open(os.path.join(args.debug, "suspicious.json"), "w", encoding="utf-8") as f:
                json.dump(low, f, ensure_ascii=False, indent=2)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    msg = f"\nYOLO detections: raw={len(dets_raw)}"
    if bool(args.dedup):
        msg += f" -> dedup={len(dets_stage1)}"
    msg += f" -> cap{int(args.max_per_label)}={len(dets_stage2)}"
    if bool(args.dedup_location):
        msg += f" -> locdedup={len(dets_stage3)}"
    if float(args.min_score) > 0.0:
        msg += f" -> minscore={len(dets_stage4)}"
    print(msg)

    # Helpful summary: how many of each label (should be <= max-per-label for most labels).
    counts = Counter([d.label for d in dets])
    top = counts.most_common(10)
    print("Top label counts:", " ".join([f"{k}x{v}" for k, v in top]))

    if int(args.expected_count) > 0 and len(dets) != int(args.expected_count):
        print(f"WARNING: expected {int(args.expected_count)} cards, got {len(dets)}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
