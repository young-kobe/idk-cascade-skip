# IDK Cascade Skip Decisions — Code

Reproduces the Nguyen et al. (RTSS 2024) threshold sweep and evaluates a
lightweight two-feature logistic skip predictor against a re-implementation of
the Random Forest of Katikaneni et al. (RTSS 2025) on ImageNet-V2
Matched-Frequency.

## Setup

```bash
pip3 install -r requirements.txt
```

`torch`, `torchvision`, and `datasets` only matter for `cache_softmax.py`. The
analysis scripts (`threshold_sweep.py`, `skip_rule.py`) only need numpy,
scikit-learn, and matplotlib.

## Run

`../data/` ships pre-cached, so the analysis reproduces in seconds:

```bash
python3 threshold_sweep.py   # threshold sweep table + figure (full 10k set)
python3 skip_rule.py         # RF + logistic training, variant/F1/overhead tables
```

`bash run.sh` runs everything end-to-end, including the caching step —
**~2 h CPU** and it overwrites the cached timings, so it skips step 1 when
`../data/` already exists.

`cache_softmax.py --subset top --n 10000 --cpu` runs ResNet-18/34/152 over
ImageNet-V2 and saves softmax probs + per-image timings. Note the warning in
its loader: the `top` alias currently resolves to a Hugging Face repo that
serves the **Matched-Frequency** variant (verified via standalone top-1
accuracies); pointing it at true per-variant TopImages files is a noted
follow-up.

## Headline results (Matched-Frequency, held-out A-IDK test split, x86 CPU)

| Variant | Latency (ms) | Accuracy | Skip-F1 |
|---|---|---|---|
| Baseline (no skip) | 137.21 | 0.464 | — |
| Static τ=0.3 (Nguyen 2024) | 130.44 | 0.467 | — |
| Random Forest (Katikaneni 2025) | 129.45 | 0.469 | 0.715 |
| **Logistic, 2-feature (ours)** | 129.47 | 0.470 | 0.706 |

Measured per-decision predictor cost (batch 1, post-warm-up): RF median
2.80 ms vs. logistic 33 µs — an **84×** gap.

Trained logistic: `logit P(skip) = 1.787 − 5.111·conf + 1.086·margin`.

The skip label follows both prior papers: skip is correct iff the middle
classifier itself would output IDK (`conf_B < 0.65`).

## Files

| file | purpose |
|---|---|
| `utils.py` | shared helpers (paths, feature extraction, lazy torch import) |
| `cache_softmax.py` | run pretrained models, cache softmax + timings |
| `threshold_sweep.py` | reproduce the Nguyen 2024 sweep protocol |
| `skip_rule.py` | train + evaluate RF and logistic skip rules, profile predictor cost |
| `requirements.txt` | Python deps |
| `run.sh` | end-to-end driver |
