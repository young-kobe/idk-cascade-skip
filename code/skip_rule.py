"""Train + evaluate the four cascade variants.

Variants:
  1. Baseline (no skip)
  2. Static threshold tau=0.3
  3. Random Forest on {confidence, entropy, margin}  -- re-implements Katikaneni 2025
  4. Logistic regression on {confidence, margin}     -- our lightweight proposal

Outputs:
  figures/variant_comparison.pdf
  paper/table_variants.tex
  paper/skip_predictor_metrics.tex
  paper/predictor_overhead.tex
  paper/logistic_coefficients.tex
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from threshold_sweep import ACC_THRESHOLD, cascade_simulate
from utils import DATA_DIR, FIG_DIR, REPO_ROOT, softmax_features, top1


def build_skip_labels(probs_b):
    """Skip-positive (y=1) when running B would be wasted: B itself outputs IDK.

    Matches the waste condition in both prior papers (Nguyen 2024: time is only
    saved by skipping B when B would output IDK; Katikaneni 2025: the predictor
    target is whether B will output IDK). If B would commit, running B is never
    wasted: it is the cheapest path to a committed answer (t_B < t_C).
    """
    conf_b = probs_b.max(axis=1)
    idk_b = conf_b < ACC_THRESHOLD
    return idk_b.astype(int)


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
    margin_a = feats["margin"]
    entropy_a = feats["entropy"]
    idk_a = conf_a < ACC_THRESHOLD

    skip_labels = build_skip_labels(probs_b)

    # Train predictors only on inputs where A returned IDK (the only place skip applies).
    idx_idk = np.where(idk_a)[0]
    X_rf = np.stack([conf_a, entropy_a, margin_a], axis=1)[idx_idk]
    X_lr = np.stack([conf_a, margin_a], axis=1)[idx_idk]
    y = skip_labels[idx_idk]

    X_rf_train, X_rf_test, X_lr_train, X_lr_test, y_train, y_test, idx_train, idx_test = \
        train_test_split(X_rf, X_lr, y, idx_idk, test_size=0.30, random_state=42, stratify=y)

    # Random Forest baseline (Katikaneni 2025 hyperparams)
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=4, min_samples_leaf=40,
        class_weight="balanced", random_state=42,
    )
    rf.fit(X_rf_train, y_train)
    rf_pred_test = rf.predict(X_rf_test)

    # Logistic regression (ours)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    lr.fit(X_lr_train, y_train)
    lr_pred_test = lr.predict(X_lr_test)

    metrics = {
        "rf": {
            "precision": float(precision_score(y_test, rf_pred_test, zero_division=0)),
            "recall":    float(recall_score(y_test, rf_pred_test, zero_division=0)),
            "f1":        float(f1_score(y_test, rf_pred_test, zero_division=0)),
            "coef":      None,
        },
        "logistic": {
            "precision": float(precision_score(y_test, lr_pred_test, zero_division=0)),
            "recall":    float(recall_score(y_test, lr_pred_test, zero_division=0)),
            "f1":        float(f1_score(y_test, lr_pred_test, zero_division=0)),
            "coef":      lr.coef_.tolist(),
            "intercept": lr.intercept_.tolist(),
        },
    }
    print(json.dumps(metrics, indent=2))

    # Per-sample predictor overhead: batch size 1, after warm-up, on test samples.
    # The logistic is timed in its deployed form (dot product + sigmoid), not via
    # sklearn's predict(), since that is what a real-time system would run.
    w_lr = lr.coef_[0]
    b_lr = lr.intercept_[0]

    def logistic_eval(x):
        return 1.0 / (1.0 + np.exp(-(x @ w_lr + b_lr)))

    n_prof = min(1000, len(X_rf_test))
    for _ in range(50):  # warm-up
        rf.predict(X_rf_test[:1])
        logistic_eval(X_lr_test[0])
    rf_us, lr_us = [], []
    for i in range(n_prof):
        t0 = time.perf_counter()
        rf.predict(X_rf_test[i:i + 1])
        rf_us.append((time.perf_counter() - t0) * 1e6)
        t0 = time.perf_counter()
        logistic_eval(X_lr_test[i])
        lr_us.append((time.perf_counter() - t0) * 1e6)
    rf_us, lr_us = np.array(rf_us), np.array(lr_us)
    overhead_ratio = np.median(rf_us) / np.median(lr_us)
    print(f"\n=== Predictor overhead (batch=1, n={n_prof}) ===")
    print(f"RF        mean={rf_us.mean():.1f} us  median={np.median(rf_us):.1f} us")
    print(f"logistic  mean={lr_us.mean():.1f} us  median={np.median(lr_us):.1f} us")
    print(f"median ratio = {overhead_ratio:.0f}x")

    out3 = REPO_ROOT / "paper" / "predictor_overhead.tex"
    with out3.open("w") as f:
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Predictor & Mean ($\\mu$s) & Median ($\\mu$s) & Relative \\\\\n\\midrule\n")
        f.write(f"Random Forest & {rf_us.mean():.1f} & {np.median(rf_us):.1f} & "
                f"${overhead_ratio:.0f}\\times$ \\\\\n")
        f.write(f"\\textbf{{Logistic (ours)}} & {lr_us.mean():.1f} & {np.median(lr_us):.1f} & "
                f"$1\\times$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    out4 = REPO_ROOT / "paper" / "logistic_coefficients.tex"
    with out4.open("w") as f:
        f.write("\\[ \\mathrm{logit}\\,P(\\mathrm{skip}\\mid \\mathbf{x}) = "
                f"{b_lr:.3f} {w_lr[0]:+.3f}\\,\\mathrm{{conf}}(A) "
                f"{w_lr[1]:+.3f}\\,\\mathrm{{margin}}(A) . \\]\n")

    # Evaluate cascade latency on the TEST subset only (to avoid leakage).
    N = probs_a.shape[0]
    test_mask = np.zeros(N, dtype=bool)
    test_mask[idx_test] = True
    sl = slice(None)  # operate on full arrays; cascade only triggers skip on idk_a anyway

    def cascade_for_skip(skip_full):
        return cascade_simulate(
            probs_a[test_mask], probs_b[test_mask], probs_c[test_mask],
            t_a[test_mask],     t_b[test_mask],     t_c[test_mask],
            labels[test_mask],
            skip_decision=skip_full[test_mask],
        )

    skip_baseline = np.zeros(N, dtype=bool)
    skip_static = (conf_a < 0.30) & idk_a  # skip rule only fires when A is IDK
    skip_rf = np.zeros(N, dtype=bool)
    skip_rf[idx_idk] = rf.predict(X_rf).astype(bool)
    skip_lr = np.zeros(N, dtype=bool)
    skip_lr[idx_idk] = lr.predict(X_lr).astype(bool)

    rows = [
        ("Baseline (no skip)", cascade_for_skip(skip_baseline)),
        ("Static $\\tau=0.3$",   cascade_for_skip(skip_static)),
        ("Random Forest",        cascade_for_skip(skip_rf)),
        ("\\textbf{Logistic (ours)}", cascade_for_skip(skip_lr)),
    ]

    out = REPO_ROOT / "paper" / "table_variants.tex"
    with out.open("w") as f:
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Variant & Latency (ms) & Accuracy & Skip rate \\\\\n\\midrule\n")
        for name, r in rows:
            f.write(f"{name} & {r['latency_ms']:.2f} & {r['accuracy']:.3f} & {r['skip_rate']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    out2 = REPO_ROOT / "paper" / "skip_predictor_metrics.tex"
    with out2.open("w") as f:
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Predictor & Precision & Recall & F1 \\\\\n\\midrule\n")
        f.write(f"Random Forest & {metrics['rf']['precision']:.3f} & "
                f"{metrics['rf']['recall']:.3f} & {metrics['rf']['f1']:.3f} \\\\\n")
        f.write(f"\\textbf{{Logistic (ours)}} & {metrics['logistic']['precision']:.3f} & "
                f"{metrics['logistic']['recall']:.3f} & {metrics['logistic']['f1']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # Bar plot: latency comparison
    names = [n.replace("$\\tau=0.3$", "tau=0.3").replace("\\textbf{", "").replace("}", "") for n, _ in rows]
    lats  = [r["latency_ms"] for _, r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    bars = ax.bar(names, lats, color=["C7", "C0", "C2", "C3"])
    ax.set_ylabel("avg latency (ms)")
    ax.set_title("Cascade variants on ImageNet-V2 Matched-Frequency (A-IDK test split)")
    for bar, v in zip(bars, lats):
        ax.annotate(f"{v:.1f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "variant_comparison.pdf")
    fig.savefig(FIG_DIR / "variant_comparison.png", dpi=150)
    print(f"Saved {FIG_DIR / 'variant_comparison.pdf'}")

    print("\n=== Logistic coefficients ===")
    print(f"intercept = {lr.intercept_[0]:+.3f}")
    print(f"w_conf    = {lr.coef_[0][0]:+.3f}")
    print(f"w_margin  = {lr.coef_[0][1]:+.3f}")


if __name__ == "__main__":
    main()
