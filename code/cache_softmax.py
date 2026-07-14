"""Run pretrained ResNet-18/34/152 over an ImageNet-V2 variant and cache softmax outputs + per-image timings.

Usage:
    python3 cache_softmax.py --subset matched --cpu
    python3 cache_softmax.py --subset thr07 --cpu --outdir ../data/arm/thr07   # e.g. on an ARM board

Each variant is downloaded as the canonical per-variant tarball from the
vaishaal/ImageNetV2 Hugging Face dataset repo (the original Recht et al.
release files), so the variant identity is exact by construction.

Outputs (in ../data/<subset>/ unless --outdir is given):
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
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from tqdm import tqdm

from utils import DATA_DIR, device

# Canonical per-variant release files of Recht et al. (2019), mirrored on the
# Hugging Face Hub. Downloading the exact file makes the variant identity
# certain; earlier revisions of this script loaded via `datasets` repos whose
# variant labeling had drifted (the old "top" alias actually served
# Matched-Frequency data).
HF_REPO = "vaishaal/ImageNetV2"
VARIANT_FILES = {
    "matched": "imagenetv2-matched-frequency.tar.gz",
    "top":     "imagenetv2-top-images.tar.gz",
    "thr07":   "imagenetv2-threshold0.7.tar.gz",
}

MODEL_FACTORIES = {
    "resnet18":  (models.resnet18,  models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34":  (models.resnet34,  models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1),
}


def load_imagenetv2(subset: str, n: int):
    """Return a sorted list of (image_path, label) for one ImageNet-V2 variant.

    Downloads the canonical per-variant tarball from vaishaal/ImageNetV2 and
    extracts it under the Hugging Face cache. Tarball layout:
    imagenetv2-<variant>-format-val/<class_id>/<image>.jpeg, class_id 0-999.
    Sorting by (label, filename) makes the image order deterministic across
    machines, so softmax arrays cached on one host align with timing arrays
    cached on another (e.g. ARM).
    """
    from huggingface_hub import hf_hub_download

    tar_path = Path(hf_hub_download(HF_REPO, VARIANT_FILES[subset], repo_type="dataset"))
    extract_root = tar_path.parent / f"{VARIANT_FILES[subset]}.extracted"
    if not extract_root.exists():
        print(f"Extracting {tar_path.name} ...")
        tmp = extract_root.with_suffix(".tmp")
        with tarfile.open(tar_path) as tf:
            tf.extractall(tmp)
        tmp.rename(extract_root)

    items = []
    for class_dir in extract_root.glob("*/*"):
        if not (class_dir.is_dir() and class_dir.name.isdigit()):
            continue
        lbl = int(class_dir.name)
        for img_path in class_dir.iterdir():
            items.append((img_path, lbl))
    if not items:
        raise RuntimeError(f"No images found under {extract_root}")
    items.sort(key=lambda t: (t[1], t[0].name))
    if n is not None:
        items = items[:n]
    return items


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

    from PIL import Image

    for i in tqdm(range(N), desc=model_name):
        img_path, lbl = ds[i]
        img = Image.open(img_path).convert("RGB")
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
    parser.add_argument("--subset", default="matched", choices=sorted(VARIANT_FILES))
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--models", nargs="+",
                        default=["resnet18", "resnet34", "resnet152"])
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA is available (recommended for paper).")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Output directory (default ../data/<subset>). "
                             "Use e.g. ../data/arm/<subset> when caching timings on another platform.")
    args = parser.parse_args()

    outdir = args.outdir if args.outdir is not None else DATA_DIR / args.subset
    outdir.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cpu") if args.cpu else device()
    print(f"Device: {dev}")
    ds = load_imagenetv2(args.subset, args.n)
    print(f"Loaded {len(ds)} images from ImageNet-V2 {args.subset}.")

    labels_saved = False
    for model_name in args.models:
        softmax, timings, labels = run_model(model_name, ds, force_cpu=args.cpu)
        np.save(outdir / f"softmax_{model_name}.npy", softmax)
        np.save(outdir / f"timing_{model_name}.npy", timings)
        if not labels_saved:
            np.save(outdir / "labels.npy", labels)
            labels_saved = True
        acc = (softmax.argmax(axis=1) == labels).mean()
        print(f"[{model_name}] top-1={acc:.3f}  mean={timings.mean()*1000:.2f} ms  "
              f"p99={np.percentile(timings, 99)*1000:.2f} ms")

    print(f"Saved arrays to {outdir}")


if __name__ == "__main__":
    main()
