"""Shared helpers for the IDK cascade experiments."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FIG_DIR = REPO_ROOT / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def load_subset(subset: str, timing_root: Path | None = None) -> dict[str, np.ndarray]:
    """Load the cached arrays for one ImageNet-V2 variant (data/<subset>/).

    timing_root: optional alternate root for the timing_*.npy arrays, e.g.
    data/arm to combine platform-independent softmax with timings measured on
    an ARM board.
    """
    d = DATA_DIR / subset
    t = (timing_root / subset) if timing_root is not None else d
    return {
        "probs_a": np.load(d / "softmax_resnet18.npy"),
        "probs_b": np.load(d / "softmax_resnet34.npy"),
        "probs_c": np.load(d / "softmax_resnet152.npy"),
        "t_a": np.load(t / "timing_resnet18.npy"),
        "t_b": np.load(t / "timing_resnet34.npy"),
        "t_c": np.load(t / "timing_resnet152.npy"),
        "labels": np.load(d / "labels.npy"),
    }


@contextmanager
def timer():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def softmax_features(probs: np.ndarray) -> dict[str, np.ndarray]:
    """Return the three Katikaneni-2025 features per row.

    probs: (N, K) softmax probabilities.
    """
    sorted_probs = np.sort(probs, axis=1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2]
    confidence = top1
    margin = top1 - top2
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    return {"confidence": confidence, "margin": margin, "entropy": entropy}


def top1(probs: np.ndarray) -> np.ndarray:
    return probs.argmax(axis=1)


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
