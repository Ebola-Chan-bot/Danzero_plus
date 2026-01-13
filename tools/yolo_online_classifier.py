"""A tiny online classifier for card crops.

This is intentionally lightweight: numpy-only softmax regression trained via SGD.
It is designed to support an interactive correction loop:
  1) run YOLO to get bboxes
  2) user corrects labels for some boxes
  3) train/update classifier on corrected crops
  4) reload and re-run recognition

The detector (YOLO/OpenCV DNN) still provides bboxes; this model only replaces
classification of each bbox crop.

Dataset format (folder):
  tmp/yolo_online_dataset/
    samples.jsonl   # one JSON per line
    crops/<id>.png  # cropped card images

Each samples.jsonl line:
  {
    "crop_path": "crops/xxxx.png",
    "label": "1A" | "dw" | "xw" | ... (must exist in 模型/coco.names),
    "source": {"roi_image": "...", "bbox": [x1,y1,x2,y2], "score": 0.9}
  }

Model file:
  tmp/yolo_online_classifier/model.npz
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def _write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_class_names(model_dir: str) -> List[str]:
    names_path = os.path.join(model_dir, "coco.names")
    if not os.path.exists(names_path):
        names_path = os.path.join(model_dir, "data", "coco.names")
    with open(names_path, "r", encoding="utf-8", errors="ignore") as f:
        names = [ln.strip() for ln in f.readlines()]
    return [n for n in names if n]


def _load_gray_32x32(image_path: str) -> np.ndarray:
    # Prefer PIL to avoid OpenCV dependency in the trainer.
    from PIL import Image

    img = Image.open(image_path).convert("L").resize((32, 32))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def featurize_crop(image_path: str) -> np.ndarray:
    g = _load_gray_32x32(image_path)
    # Flatten + add a tiny bias-like constant feature can help.
    x = g.reshape(-1)
    return x.astype(np.float32)


@dataclass
class SoftmaxModel:
    W: np.ndarray  # (D, C)
    b: np.ndarray  # (C,)
    classes: List[str]

    @property
    def dim(self) -> int:
        return int(self.W.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.W.shape[1])


def init_model(*, dim: int, classes: List[str], seed: int = 0) -> SoftmaxModel:
    rng = np.random.default_rng(int(seed))
    C = len(classes)
    # Small init.
    W = (rng.standard_normal((dim, C)).astype(np.float32)) * 0.01
    b = np.zeros((C,), dtype=np.float32)
    return SoftmaxModel(W=W, b=b, classes=list(classes))


def save_model(path: str, model: SoftmaxModel) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, W=model.W, b=model.b, classes=np.array(model.classes, dtype=object))


def load_model(path: str) -> Optional[SoftmaxModel]:
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    classes = data["classes"].tolist()
    return SoftmaxModel(W=data["W"].astype(np.float32), b=data["b"].astype(np.float32), classes=list(classes))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / (np.sum(ez) + 1e-12)


def predict_proba(model: SoftmaxModel, x: np.ndarray) -> np.ndarray:
    z = x @ model.W + model.b
    return _softmax(z.astype(np.float32))


def predict_label(model: SoftmaxModel, x: np.ndarray) -> Tuple[str, float]:
    p = predict_proba(model, x)
    idx = int(np.argmax(p))
    return model.classes[idx], float(p[idx])


def _label_to_index(classes: List[str]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(classes)}


def train_sgd(
    model: SoftmaxModel,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 80,
    lr: float = 0.2,
    l2: float = 1e-4,
    batch_size: int = 32,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(int(seed))
    n = int(X.shape[0])
    if n == 0:
        return {"n": 0.0, "loss": 0.0, "acc": 0.0}

    C = model.num_classes

    def loss_and_acc() -> Tuple[float, float]:
        loss = 0.0
        correct = 0
        for i in range(n):
            z = X[i] @ model.W + model.b
            # stable softmax
            z = z - np.max(z)
            ez = np.exp(z)
            p = ez / (np.sum(ez) + 1e-12)
            yi = int(y[i])
            loss -= float(np.log(float(p[yi]) + 1e-12))
            if int(np.argmax(p)) == yi:
                correct += 1
        loss /= max(1, n)
        # L2
        loss += float(l2) * float(np.sum(model.W * model.W))
        acc = correct / max(1, n)
        return loss, float(acc)

    for _ in range(int(epochs)):
        idx = np.arange(n)
        rng.shuffle(idx)
        for s in range(0, n, int(batch_size)):
            bidx = idx[s : s + int(batch_size)]
            xb = X[bidx]
            yb = y[bidx]

            # logits: (B,C)
            Z = xb @ model.W + model.b
            Z = Z - np.max(Z, axis=1, keepdims=True)
            P = np.exp(Z)
            P /= np.sum(P, axis=1, keepdims=True) + 1e-12

            # gradient
            G = P
            G[np.arange(G.shape[0]), yb] -= 1.0
            G /= max(1, G.shape[0])

            dW = xb.T @ G + (2.0 * float(l2) * model.W)
            db = np.sum(G, axis=0)

            model.W -= float(lr) * dW.astype(np.float32)
            model.b -= float(lr) * db.astype(np.float32)

    loss, acc = loss_and_acc()
    return {"n": float(n), "loss": float(loss), "acc": float(acc)}


def load_dataset(dataset_dir: str, *, model_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    classes = load_class_names(model_dir)
    rows = _read_jsonl(os.path.join(dataset_dir, "samples.jsonl"))
    if not rows:
        return np.zeros((0, 1024), dtype=np.float32), np.zeros((0,), dtype=np.int64), classes

    cmap = _label_to_index(classes)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    for r in rows:
        lab = str(r.get("label") or "").strip()
        rel = str(r.get("crop_path") or "").strip()
        if not lab or not rel:
            continue
        if lab not in cmap:
            continue
        crop_path = os.path.join(dataset_dir, rel)
        if not os.path.exists(crop_path):
            continue
        x = featurize_crop(crop_path)
        if x.shape[0] != 1024:
            continue
        X_list.append(x)
        y_list.append(int(cmap[lab]))

    if not X_list:
        return np.zeros((0, 1024), dtype=np.float32), np.zeros((0,), dtype=np.int64), classes

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, classes


def append_samples(dataset_dir: str, samples: List[dict]) -> None:
    path = os.path.join(dataset_dir, "samples.jsonl")
    existing = _read_jsonl(path)
    existing.extend(samples)
    _write_jsonl(path, existing)
