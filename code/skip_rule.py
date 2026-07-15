"""Train + evaluate the four cascade variants.

Default protocol (cross-dataset, matching Katikaneni 2025): train the learned
skip predictors on the A-IDK samples of ImageNet-V2 Matched-Frequency +
TopImages, evaluate on the full A-IDK population of Threshold-0.7. Train and
test come from disjoint datasets, so no within-dataset split is needed and the
entire test-subset A-IDK population is used for evaluation.

--single-subset runs the legacy within-Matched-Frequency 70/30 protocol over
10 random splits and prints mean +/- std (no paper tables; kept for
continuity with the course version of this work).

--timing-root DIR sources the test subset's timing_*.npy from DIR/thr07
instead of data/thr07, so platform-independent softmax can be combined with
per-image timings measured on another machine. The predictor-overhead
profiling always runs on the *current* host (see --skip-overhead): run this
script on the deployment target for its overhead numbers.

Variants:
  1. Baseline (no skip)
  2. Static threshold tau=0.3
  3. Random Forest on {confidence, entropy, margin}  -- re-implements Katikaneni 2025
  4. Logistic regression on {confidence, margin}     -- our lightweight proposal

Outputs (cross-dataset mode):
  figures/variant_comparison.pdf
  paper/table_variants.tex
  paper/skip_predictor_metrics.tex   (RF baseline + logistic feature ablation)
  paper/predictor_overhead.tex       (host-measured; see --skip-overhead)
  paper/logistic_coefficients.tex
  paper/paired_diff.tex
"""

from __future__ import annotations

import argparse
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
from utils import FIG_DIR, REPO_ROOT, load_subset, softmax_features

TRAIN_SUBSETS = ["matched", "top"]
TEST_SUBSET = "thr07"
RF_FEATURES = ["confidence", "entropy", "margin"]  # Katikaneni 2025 feature set
LR_FEATURES = ["confidence", "margin"]
BOOT_B = 10_000
BOOT_SEED = 0


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


def prep(subset, timing_root=None):
    d = load_subset(subset, timing_root)
    f = softmax_features(d["probs_a"])
    d["features"] = f
    d["idk_a"] = f["confidence"] < ACC_THRESHOLD
    d["skip_labels"] = build_skip_labels(d["probs_b"])
    return d


def feature_matrix(d, names, mask=None):
    cols = np.stack([d["features"][n] for n in names], axis=1)
    return cols if mask is None else cols[mask]


def make_rf():
    # Katikaneni 2025 hyperparams
    return RandomForestClassifier(
        n_estimators=50, max_depth=4, min_samples_leaf=40,
        class_weight="balanced", random_state=42,
    )


def make_lr():
    return LogisticRegression(class_weight="balanced", max_iter=1000)


def prf(y_true, y_pred):
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Bootstrap machinery (percentile CIs, chunked to bound memory)
# ---------------------------------------------------------------------------

def bootstrap_ci(stat_fn, n, B=BOOT_B, seed=BOOT_SEED, chunk=1000):
    """95% percentile CI of a statistic under resampling of n items.

    stat_fn(idx) receives an int array (chunk, n) of resampled indices and
    returns (chunk,) statistics.
    """
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(B // chunk):
        idx = rng.integers(0, n, size=(chunk, n))
        stats.append(np.asarray(stat_fn(idx)))
    lo, hi = np.percentile(np.concatenate(stats), [2.5, 97.5])
    return float(lo), float(hi)


def mean_ci(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return bootstrap_ci(lambda idx: vals[idx].mean(axis=1), len(vals))


def f1_ci(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).astype(np.float64)
    fp = ((y_true == 0) & (y_pred == 1)).astype(np.float64)
    fn = ((y_true == 1) & (y_pred == 0)).astype(np.float64)

    def stat(idx):
        TP, FP, FN = tp[idx].sum(axis=1), fp[idx].sum(axis=1), fn[idx].sum(axis=1)
        return 2 * TP / np.maximum(2 * TP + FP + FN, 1e-12)

    return bootstrap_ci(stat, len(y_true))


def fmt_ci(point, lo, hi, prec=2):
    return f"{point:.{prec}f}\\,{{\\scriptsize$[{lo:.{prec}f},\\,{hi:.{prec}f}]$}}"


# ---------------------------------------------------------------------------
# Cross-dataset protocol (default)
# ---------------------------------------------------------------------------

def run_cross_dataset(timing_root, skip_overhead=False):
    train = [prep(s) for s in TRAIN_SUBSETS]
    test = prep(TEST_SUBSET, timing_root)

    X_rf_train = np.concatenate([feature_matrix(d, RF_FEATURES, d["idk_a"]) for d in train])
    X_lr_train = np.concatenate([feature_matrix(d, LR_FEATURES, d["idk_a"]) for d in train])
    y_train = np.concatenate([d["skip_labels"][d["idk_a"]] for d in train])

    m = test["idk_a"]
    X_rf_test = feature_matrix(test, RF_FEATURES, m)
    X_lr_test = feature_matrix(test, LR_FEATURES, m)
    y_test = test["skip_labels"][m]

    print(f"Train (A-IDK of {'+'.join(TRAIN_SUBSETS)}): {len(y_train)} samples, "
          f"P(skip)={y_train.mean():.3f}")
    print(f"Test  (A-IDK of {TEST_SUBSET}): {len(y_test)} samples, P(skip)={y_test.mean():.3f}")

    rf = make_rf()
    rf.fit(X_rf_train, y_train)
    rf_pred = rf.predict(X_rf_test)

    lr = make_lr()
    lr.fit(X_lr_train, y_train)
    lr_pred = lr.predict(X_lr_test)

    metrics = {"rf": prf(y_test, rf_pred), "logistic": prf(y_test, lr_pred)}
    metrics["rf"]["f1_ci"] = f1_ci(y_test, rf_pred)
    metrics["logistic"]["f1_ci"] = f1_ci(y_test, lr_pred)
    print(json.dumps(metrics, indent=2))

    write_predictor_table(train, test, metrics["rf"])
    if not skip_overhead:
        profile_overhead(rf, lr, X_rf_test, X_lr_test)
    write_coefficients(lr)

    # Cascade evaluation on the full A-IDK population of the test subset.
    def cascade_for_skip(skip_full):
        return cascade_simulate(
            test["probs_a"][m], test["probs_b"][m], test["probs_c"][m],
            test["t_a"][m], test["t_b"][m], test["t_c"][m],
            test["labels"][m],
            skip_decision=skip_full[m],
        )

    N = test["probs_a"].shape[0]
    conf_a = test["features"]["confidence"]
    skip_baseline = np.zeros(N, dtype=bool)
    skip_static = (conf_a < 0.30) & test["idk_a"]
    skip_rf = np.zeros(N, dtype=bool)
    skip_rf[m] = rf.predict(X_rf_test).astype(bool)
    skip_lr = np.zeros(N, dtype=bool)
    skip_lr[m] = lr.predict(X_lr_test).astype(bool)

    rows = [
        ("Baseline (no skip)",        cascade_for_skip(skip_baseline)),
        ("Static $\\tau=0.3$",        cascade_for_skip(skip_static)),
        ("Random Forest",             cascade_for_skip(skip_rf)),
        ("\\textbf{Logistic (ours)}", cascade_for_skip(skip_lr)),
    ]
    write_variants_table(rows)
    write_paired_diff(rows[2][1], rows[3][1])
    plot_variants(rows)

    print("\n=== Logistic coefficients ===")
    print(f"intercept = {lr.intercept_[0]:+.3f}")
    for name, w in zip(LR_FEATURES, lr.coef_[0]):
        print(f"w_{name:<8} = {w:+.3f}")


def write_predictor_table(train, test, rf_metrics):
    """One merged table: the RF baseline plus the logistic nested-feature
    ablation (the feature-selection evidence), same columns throughout."""
    m = test["idk_a"]
    y_train = np.concatenate([d["skip_labels"][d["idk_a"]] for d in train])
    y_test = test["skip_labels"][m]
    sets = [
        (["confidence"],                      "Logistic \\{conf\\}"),
        (["confidence", "margin"],            "\\textbf{Logistic \\{conf, margin\\} (ours)}"),
        (["confidence", "margin", "entropy"], "Logistic \\{conf, margin, entropy\\}"),
    ]
    out = REPO_ROOT / "paper" / "skip_predictor_metrics.tex"
    print("\n=== Predictor quality + feature ablation ===")
    with out.open("w") as f:
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Predictor & Precision & Recall & F1 \\\\\n\\midrule\n")
        f.write(f"RF \\{{conf, entropy, margin\\}} & {rf_metrics['precision']:.3f} & "
                f"{rf_metrics['recall']:.3f} & "
                f"{fmt_ci(rf_metrics['f1'], *rf_metrics['f1_ci'], prec=3)} \\\\\n")
        for names, label in sets:
            X_train = np.concatenate([feature_matrix(d, names, d["idk_a"]) for d in train])
            X_test = feature_matrix(test, names, m)
            lr = make_lr()
            lr.fit(X_train, y_train)
            pred = lr.predict(X_test)
            mm = prf(y_test, pred)
            lo, hi = f1_ci(y_test, pred)
            print(f"{'+'.join(names):<28} P={mm['precision']:.3f} R={mm['recall']:.3f} "
                  f"F1={mm['f1']:.3f} [{lo:.3f}, {hi:.3f}]")
            f.write(f"{label} & {mm['precision']:.3f} & {mm['recall']:.3f} & "
                    f"{fmt_ci(mm['f1'], lo, hi, prec=3)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def profile_overhead(rf, lr, X_rf_test, X_lr_test):
    """Per-sample predictor cost: batch size 1, after warm-up, on test samples.

    The logistic is timed in its deployed form (dot product + sigmoid), not via
    sklearn's predict(), since that is what a real-time system would run.
    """
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
    ratio = np.median(rf_us) / np.median(lr_us)

    def stats(v):
        return v.mean(), np.median(v), np.percentile(v, 99), v.max()

    print(f"\n=== Predictor overhead (batch=1, n={n_prof}) ===")
    for name, v in [("RF", rf_us), ("logistic", lr_us)]:
        mean, med, p99, mx = stats(v)
        print(f"{name:<9} mean={mean:.1f} us  median={med:.1f} us  "
              f"p99={p99:.1f} us  max={mx:.1f} us")
    print(f"median ratio = {ratio:.0f}x")

    out = REPO_ROOT / "paper" / "predictor_overhead.tex"
    with out.open("w") as f:
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("Predictor & Mean & Median & p99 & Max & Relative \\\\\n")
        f.write(" & ($\\mu$s) & ($\\mu$s) & ($\\mu$s) & ($\\mu$s) & (median) \\\\\n\\midrule\n")
        mean, med, p99, mx = stats(rf_us)
        f.write(f"Random Forest & {mean:.1f} & {med:.1f} & {p99:.1f} & {mx:.1f} & "
                f"${ratio:.0f}\\times$ \\\\\n")
        mean, med, p99, mx = stats(lr_us)
        f.write(f"\\textbf{{Logistic (ours)}} & {mean:.1f} & {med:.1f} & {p99:.1f} & {mx:.1f} & "
                f"$1\\times$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def write_coefficients(lr):
    w_lr = lr.coef_[0]
    b_lr = lr.intercept_[0]
    out = REPO_ROOT / "paper" / "logistic_coefficients.tex"
    with out.open("w") as f:
        f.write("\\[ \\mathrm{logit}\\,P(\\mathrm{skip}\\mid \\mathbf{x}) = "
                f"{b_lr:.3f} {w_lr[0]:+.3f}\\,\\mathrm{{conf}}(A) "
                f"{w_lr[1]:+.3f}\\,\\mathrm{{margin}}(A) . \\]\n")


def write_variants_table(rows):
    out = REPO_ROOT / "paper" / "table_variants.tex"
    with out.open("w") as f:
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Variant & Latency (ms) & Accuracy & Skip rate \\\\\n\\midrule\n")
        for name, r in rows:
            lat_ci = mean_ci(r["latency_arr"] * 1000)
            acc_ci = mean_ci(r["correct_arr"].astype(np.float64))
            f.write(f"{name} & {fmt_ci(r['latency_ms'], *lat_ci, prec=2)} & "
                    f"{fmt_ci(r['accuracy'], *acc_ci, prec=3)} & {r['skip_rate']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    for name, r in rows:
        print(f"{name:<28} lat={r['latency_ms']:.2f} ms  acc={r['accuracy']:.3f}  "
              f"skip={r['skip_rate']:.3f}")


def write_paired_diff(r_rf, r_lr):
    """Paired bootstrap CI of (RF - logistic) per-image differences.

    Written as a sentence fragment the paper \\input{}s, so the statistical
    claim in the text is regenerated with the tables.
    """
    lat_diff = (r_rf["latency_arr"] - r_lr["latency_arr"]) * 1000
    acc_diff = r_rf["correct_arr"].astype(np.float64) - r_lr["correct_arr"].astype(np.float64)
    lat_lo, lat_hi = mean_ci(lat_diff)
    acc_lo, acc_hi = mean_ci(acc_diff)
    print(f"\nPaired diff (RF - logistic): latency [{lat_lo:+.3f}, {lat_hi:+.3f}] ms, "
          f"accuracy [{acc_lo:+.4f}, {acc_hi:+.4f}]")
    out = REPO_ROOT / "paper" / "paired_diff.tex"
    with out.open("w") as f:
        f.write(f"$[{lat_lo:+.2f}, {lat_hi:+.2f}]$~ms for latency and "
                f"$[{acc_lo:+.3f}, {acc_hi:+.3f}]$ for accuracy\n")


def plot_variants(rows):
    names = [n.replace("$\\tau=0.3$", "tau=0.3").replace("\\textbf{", "").replace("}", "")
             for n, _ in rows]
    lats = [r["latency_ms"] for _, r in rows]
    errs = []
    for _, r in rows:
        lo, hi = mean_ci(r["latency_arr"] * 1000)
        errs.append([r["latency_ms"] - lo, hi - r["latency_ms"]])
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    bars = ax.bar(names, lats, color=["C7", "C0", "C2", "C3"],
                  yerr=np.array(errs).T, capsize=3)
    ax.set_ylabel("avg latency (ms)")
    ax.set_title("Cascade variants on ImageNet-V2 Threshold-0.7 (A-IDK population)")
    for bar, v in zip(bars, lats):
        ax.annotate(f"{v:.1f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "variant_comparison.pdf")
    fig.savefig(FIG_DIR / "variant_comparison.png", dpi=150)
    print(f"Saved {FIG_DIR / 'variant_comparison.pdf'}")


# ---------------------------------------------------------------------------
# Legacy single-subset protocol (10 random 70/30 splits, printed only)
# ---------------------------------------------------------------------------

def run_single_subset(n_seeds=10):
    d = prep("matched")
    m = d["idk_a"]
    X_rf = feature_matrix(d, RF_FEATURES, m)
    X_lr = feature_matrix(d, LR_FEATURES, m)
    y = d["skip_labels"][m]

    rf_f1s, lr_f1s = [], []
    for seed in range(n_seeds):
        X_rf_tr, X_rf_te, X_lr_tr, X_lr_te, y_tr, y_te = train_test_split(
            X_rf, X_lr, y, test_size=0.30, random_state=seed, stratify=y)
        rf = make_rf()
        rf.fit(X_rf_tr, y_tr)
        rf_f1s.append(f1_score(y_te, rf.predict(X_rf_te)))
        lr = make_lr()
        lr.fit(X_lr_tr, y_tr)
        lr_f1s.append(f1_score(y_te, lr.predict(X_lr_te)))

    rf_f1s, lr_f1s = np.array(rf_f1s), np.array(lr_f1s)
    print(f"Single-subset (matched, {n_seeds} random 70/30 splits):")
    print(f"  RF       skip-F1 = {rf_f1s.mean():.3f} +/- {rf_f1s.std():.3f}")
    print(f"  logistic skip-F1 = {lr_f1s.mean():.3f} +/- {lr_f1s.std():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-subset", action="store_true",
                        help="Legacy within-Matched-Frequency 70/30 protocol (printed only).")
    parser.add_argument("--timing-root", type=Path, default=None,
                        help="Alternate root for the test subset's timing_*.npy "
                             "(e.g. ../data/arm for ARM-measured timings).")
    parser.add_argument("--skip-overhead", action="store_true",
                        help="Do not regenerate predictor_overhead.tex (keeps "
                             "board-measured numbers when re-running tables on another host).")
    args = parser.parse_args()

    if args.single_subset:
        run_single_subset()
    else:
        run_cross_dataset(args.timing_root, args.skip_overhead)


if __name__ == "__main__":
    main()
