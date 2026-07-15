# IDK Cascade Skip Decisions — Code

Reproduces the Nguyen et al. (RTSS 2024) threshold sweep and evaluates a
lightweight two-feature logistic skip predictor against a re-implementation of
the Random Forest of Katikaneni et al. (RTSS 2025), under their cross-dataset
protocol (train on ImageNet-V2 Matched-Frequency + TopImages, test on
Threshold-0.7), measured on a Jetson Orin Nano.

## Setup

```bash
pip3 install -r requirements.txt
```

`torch`, `torchvision`, and `huggingface_hub` only matter for
`cache_softmax.py`. The analysis scripts (`threshold_sweep.py`,
`skip_rule.py`) only need numpy, scipy, scikit-learn, and matplotlib.

## Run

`../data/` ships pre-cached (board-measured), so the analysis reproduces in
seconds — the repo-root Makefile is the entry point:

```bash
make -C .. sweep    # threshold sweep table + figure (full 10k Matched-Frequency)
make -C .. rules    # cross-dataset RF + logistic training, all paper tables
```

`skip_rule.py` flags:

- `--single-subset` — legacy within-Matched-Frequency 70/30 protocol,
  10 random splits, printed only.
- `--timing-root DIR` — source the test subset's `timing_*.npy` from
  `DIR/thr07` (combine platform-independent softmax with timings from
  another machine).
- `--skip-overhead` — don't overwrite `paper/predictor_overhead.tex`; the
  overhead table must be measured on the deployment target, and the committed
  numbers are from the Jetson Orin Nano.

`cache_softmax.py --subset {matched,top,thr07} --cpu` downloads the canonical
per-variant ImageNet-V2 release tarball (vaishaal/ImageNetV2 on the Hugging
Face Hub), runs ResNet-18/34/152 over it, and saves softmax probs + per-image
timings to `../data/<subset>/` (`--outdir` overrides, e.g. for per-platform
timing sets). Variant identity is verified by the standalone top-1 accuracies
it prints. `arm_bootstrap.sh` is the self-contained board pipeline used to
produce the committed data (deps → cache all three subsets → full analysis).

## Headline results (cross-dataset, thr07 A-IDK population, Jetson Orin Nano CPU)

| Variant | Latency (ms) | Accuracy | Skip-F1 |
|---|---|---|---|
| Baseline (no skip) | 374.86 | 0.549 | — |
| Static τ=0.3 (Nguyen 2024) | 363.65 | 0.552 | — |
| Random Forest (Katikaneni 2025) | 363.83 | 0.553 | 0.682 |
| **Logistic, 2-feature (ours)** | 365.28 | 0.553 | 0.662 |

Measured per-decision predictor cost (batch 1, post-warm-up, on-board): RF
median 3.80 ms vs. logistic 19 µs — a **205×** gap, larger than the RF's
entire paired cascade-latency advantage (≤ 2.6 ms).

Trained logistic: `logit P(skip) = 1.841 − 5.481·conf + 1.785·margin`.

The skip label follows both prior papers: skip is correct iff the middle
classifier itself would output IDK (`conf_B < 0.65`).

## Files

| file | purpose |
|---|---|
| `utils.py` | shared helpers (paths, per-subset loading, feature extraction) |
| `cache_softmax.py` | download an ImageNet-V2 variant, run models, cache softmax + timings |
| `threshold_sweep.py` | reproduce the Nguyen 2024 sweep protocol |
| `skip_rule.py` | train + evaluate skip rules, bootstrap CIs, ablation, overhead profile |
| `arm_bootstrap.sh` | self-contained pipeline for an ARM board |
| `run.sh` | end-to-end driver (cache-if-missing + analysis) |
| `requirements.txt` | Python deps |
