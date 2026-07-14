"""Reproduce Table II of Nguyen et al., RTSS 2024.

For each ResNet-18 confidence threshold tau in {0.0, 0.1, ..., 0.9}, simulate the
A->B->C cascade where B is skipped when conf(A) < tau, then compute average
per-image latency, top-1 accuracy, and skip rate using the cached softmax outputs
and timings produced by cache_softmax.py.

Outputs:
    figures/threshold_sweep.pdf
    figures/threshold_sweep.png
    paper/table_threshold_sweep.tex   (LaTeX booktabs table)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import DATA_DIR, FIG_DIR, REPO_ROOT, softmax_features, top1

ACC_THRESHOLD = 0.65  # An IDK if max softmax < 0.65 (matches Nguyen 2024)


def cascade_simulate(probs_a, probs_b, probs_c, t_a, t_b, t_c, labels, skip_decision):
    """Run a single cascade pass.

    skip_decision: bool[N]. True => skip B and run C directly when A is IDK.
    Returns dict of (latency_ms, accuracy, skip_rate).
    """
    N = probs_a.shape[0]
    pred_a = top1(probs_a)
    conf_a = probs_a.max(axis=1)
    idk_a = conf_a < ACC_THRESHOLD

    pred_b = top1(probs_b)
    conf_b = probs_b.max(axis=1)
    idk_b = conf_b < ACC_THRESHOLD

    pred_c = top1(probs_c)

    # Per-image latency and final committed prediction.
    latency = np.zeros(N, dtype=np.float64)
    final_pred = np.zeros(N, dtype=np.int64)
    skipped_b = np.zeros(N, dtype=bool)

    for i in range(N):
        latency[i] = t_a[i]
        if not idk_a[i]:
            final_pred[i] = pred_a[i]
            continue
        # A returned IDK. Decide whether to skip B.
        if skip_decision[i]:
            latency[i] += t_c[i]
            final_pred[i] = pred_c[i]
            skipped_b[i] = True
        else:
            latency[i] += t_b[i]
            if not idk_b[i]:
                final_pred[i] = pred_b[i]
            else:
                latency[i] += t_c[i]
                final_pred[i] = pred_c[i]

    return {
        "latency_ms": latency.mean() * 1000,
        "accuracy": (final_pred == labels).mean(),
        "skip_rate": skipped_b.mean(),
        "latency_arr": latency,
    }


def main():
    probs_a = np.load(DATA_DIR / "softmax_resnet18.npy")
    probs_b = np.load(DATA_DIR / "softmax_resnet34.npy")
    probs_c = np.load(DATA_DIR / "softmax_resnet152.npy")
    t_a = np.load(DATA_DIR / "timing_resnet18.npy")
    t_b = np.load(DATA_DIR / "timing_resnet34.npy")
    t_c = np.load(DATA_DIR / "timing_resnet152.npy")
    labels = np.load(DATA_DIR / "labels.npy")

    feats = softmax_features(probs_a)
    conf_a = feats["confidence"]

    # Baseline: never skip B.
    base = cascade_simulate(probs_a, probs_b, probs_c, t_a, t_b, t_c, labels,
                            skip_decision=np.zeros(len(probs_a), dtype=bool))

    rows = [(0.0, base["latency_ms"], base["accuracy"], 0.0)]
    for tau in np.arange(0.1, 1.0, 0.1):
        skip = conf_a < tau
        r = cascade_simulate(probs_a, probs_b, probs_c, t_a, t_b, t_c, labels, skip)
        rows.append((tau, r["latency_ms"], r["accuracy"], r["skip_rate"]))

    print(f"{'tau':>6}  {'latency (ms)':>14}  {'acc':>6}  {'skip':>6}")
    for tau, lat, acc, sk in rows:
        print(f"{tau:>6.1f}  {lat:>14.3f}  {acc:>6.3f}  {sk:>6.3f}")

    # Plot
    taus = [r[0] for r in rows]
    lats = [r[1] for r in rows]
    accs = [r[2] for r in rows]
    fig, ax1 = plt.subplots(figsize=(5.0, 3.2))
    ax1.plot(taus, lats, "o-", color="C0", label="latency (ms)")
    ax1.set_xlabel(r"ResNet-18 confidence threshold $\tau$")
    ax1.set_ylabel("avg latency (ms)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    ax2.plot(taus, accs, "s--", color="C1", label="top-1 accuracy")
    ax2.set_ylabel("top-1 accuracy", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "threshold_sweep.pdf")
    fig.savefig(FIG_DIR / "threshold_sweep.png", dpi=150)
    print(f"Saved figure to {FIG_DIR / 'threshold_sweep.pdf'}")

    # LaTeX table
    out = REPO_ROOT / "paper" / "table_threshold_sweep.tex"
    with out.open("w") as f:
        f.write("\\begin{tabular}{cccc}\n")
        f.write("\\toprule\n")
        f.write("$\\tau$ & Latency (ms) & Accuracy & Skip rate \\\\\n")
        f.write("\\midrule\n")
        for tau, lat, acc, sk in rows:
            f.write(f"{tau:.1f} & {lat:.2f} & {acc:.3f} & {sk:.3f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"Saved LaTeX table to {out}")


if __name__ == "__main__":
    main()
