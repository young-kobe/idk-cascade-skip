"""Run pretrained ResNet-18/34/152 over an ImageNet-V2 subset and cache softmax outputs + per-image timings.

Usage:
    python cache_softmax.py --subset top --n 10000

Outputs (in ../data/):
    softmax_resnet18.npy   shape (N, 1000)
    softmax_resnet34.npy   shape (N, 1000)
    softmax_resnet152.npy  shape (N, 1000)
    timing_resnet18.npy    shape (N,)
    timing_resnet34.npy    shape (N,)
    timing_resnet152.npy   shape (N,)
    labels.npy             shape (N,)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from tqdm import tqdm

from utils import DATA_DIR, device

MODEL_FACTORIES = {
    "resnet18":  (models.resnet18,  models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34":  (models.resnet34,  models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1),
}


def load_imagenetv2(subset: str, n: int):
    """Load N images + labels from ImageNet-V2 via Hugging Face datasets.

    Tries several known repositories in order. Dataset structure on HF has
    shifted over time — vaishaal/ImageNetV2 used to expose per-subset configs
    (topimages / matched-frequency / threshold0.7) but now only exposes
    'default'. We try modern timm-hosted subsets first, then fall back.
    """
    from datasets import load_dataset

    # Subset alias -> ordered list of (repo_id, config, split) candidates.
    #
    # WARNING: clip-benchmark/wds_imagenetv2 serves the MATCHED-FREQUENCY variant,
    # not TopImages — so the "top" alias below silently yields matched-frequency
    # data (this is what the cached data/ arrays contain; standalone top-1
    # accuracies confirm it). Follow-up: point "top" at the per-variant
    # vaishaal/ImageNetV2 release files and re-cache (~2 CPU-hours).
    candidates = {
        "top": [
            ("clip-benchmark/wds_imagenetv2", None, "test"),
            ("vaishaal/ImageNetV2", "default", "test"),
        ],
        "matched": [
            ("vaishaal/ImageNetV2", "default", "test"),
        ],
        "threshold": [
            ("vaishaal/ImageNetV2", "default", "test"),
        ],
    }

    last_err = None
    for repo, cfg, split in candidates[subset]:
        try:
            print(f"Trying {repo} (config={cfg}, split={split}) ...")
            if cfg is None:
                ds = load_dataset(repo, split=split, streaming=False)
            else:
                ds = load_dataset(repo, cfg, split=split, streaming=False)
            print(f"  loaded {len(ds)} examples; columns={ds.column_names}")
            break
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {e}")
            last_err = e
            ds = None

    if ds is None:
        raise RuntimeError(f"All candidate ImageNet-V2 sources failed: {last_err}")

    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    return ds


def get_image_and_label(ex: dict):
    """Robust accessor over slightly different ImageNet-V2 schemas.

    Different HF repos use different field names: 'image' vs 'img' vs 'jpg',
    'label' vs 'cls' vs 'labels'. Normalize here so the rest of the script
    doesn't care.
    """
    img = ex.get("image") or ex.get("img") or ex.get("jpg") or ex.get("webp") or ex.get("png")
    lbl = ex.get("label")
    if lbl is None:
        lbl = ex.get("cls")
    if lbl is None:
        lbl = ex.get("labels")
    return img, int(lbl)


def make_transform(weights):
    return weights.transforms()


@torch.inference_mode()
def run_model(model_name: str, ds, batch_size: int = 1, force_cpu: bool = False):
    dev = torch.device("cpu") if force_cpu else device()
    factory, weights = MODEL_FACTORIES[model_name]
    model = factory(weights=weights).to(dev).eval()
    tfm = make_transform(weights)

    N = len(ds)
    K = 1000
    softmax = np.zeros((N, K), dtype=np.float32)
    timings = np.zeros(N, dtype=np.float64)
    labels = np.zeros(N, dtype=np.int64)

    # warm-up
    dummy = torch.zeros(1, 3, 224, 224, device=dev)
    for _ in range(50):
        _ = model(dummy)

    from io import BytesIO
    from PIL import Image

    for i in tqdm(range(N), desc=model_name):
        ex = ds[i]
        img, lbl = get_image_and_label(ex)
        if isinstance(img, (bytes, bytearray)):
            img = Image.open(BytesIO(img))
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        x = tfm(img).unsqueeze(0).to(dev)
        labels[i] = lbl

        # Average over 5 trials per image (matches typical RT measurement protocol).
        trial_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            logits = model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            trial_times.append(time.perf_counter() - t0)
        timings[i] = float(np.mean(trial_times))
        softmax[i] = F.softmax(logits, dim=1).cpu().numpy()[0]

    return softmax, timings, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="top", choices=["top", "matched", "threshold"])
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--models", nargs="+",
                        default=["resnet18", "resnet34", "resnet152"])
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA is available (recommended for paper).")
    args = parser.parse_args()

    dev = torch.device("cpu") if args.cpu else device()
    print(f"Device: {dev}")
    ds = load_imagenetv2(args.subset, args.n)
    print(f"Loaded {len(ds)} images from ImageNet-V2 {args.subset}.")

    labels_saved = False
    for model_name in args.models:
        softmax, timings, labels = run_model(model_name, ds, force_cpu=args.cpu)
        np.save(DATA_DIR / f"softmax_{model_name}.npy", softmax)
        np.save(DATA_DIR / f"timing_{model_name}.npy", timings)
        if not labels_saved:
            np.save(DATA_DIR / "labels.npy", labels)
            labels_saved = True
        print(f"[{model_name}] mean={timings.mean()*1000:.2f} ms  "
              f"p99={np.percentile(timings, 99)*1000:.2f} ms")

    print(f"Saved arrays to {DATA_DIR}")


if __name__ == "__main__":
    main()
