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
