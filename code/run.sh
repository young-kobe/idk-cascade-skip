#!/usr/bin/env bash
# End-to-end pipeline. Run from code/.
set -euo pipefail

cd "$(dirname "$0")"

# 1. Cache softmax outputs + timings (slowest step; ~2 hr CPU on 10k images).
#    Skipped when ../data/ is already populated -- re-running overwrites the
#    cached timings that the paper's numbers are based on.
if [ -f ../data/softmax_resnet152.npy ]; then
    echo "data/ already cached; skipping cache_softmax.py (delete data/*.npy to force)."
else
    python3 cache_softmax.py --subset top --n 10000 --cpu
fi

# 2. Reproduce Nguyen 2024 threshold sweep.
python3 threshold_sweep.py

# 3. Train + evaluate the 4 cascade variants.
python3 skip_rule.py

echo "Done. Figures in ../figures/, LaTeX tables in ../paper/."
