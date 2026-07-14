#!/usr/bin/env bash
# End-to-end pipeline. Run from code/.
set -euo pipefail

cd "$(dirname "$0")"

# 1. Cache softmax outputs + timings per ImageNet-V2 variant (slowest step;
#    hours of CPU per subset). Skipped per subset when ../data/<subset>/ is
#    already populated -- re-running overwrites the cached timings that the
#    paper's numbers are based on.
for subset in matched top thr07; do
    if [ -f "../data/${subset}/softmax_resnet152.npy" ]; then
        echo "data/${subset}/ already cached; skipping (delete data/${subset}/*.npy to force)."
    else
        python3 cache_softmax.py --subset "${subset}" --cpu
    fi
done

# 2. Reproduce Nguyen 2024 threshold sweep (Matched-Frequency).
python3 threshold_sweep.py

# 3. Train + evaluate the 4 cascade variants (cross-dataset protocol:
#    train matched+top, test thr07). Pass --timing-root ../data/arm to use
#    ARM-measured timings for the test subset.
python3 skip_rule.py

echo "Done. Figures in ../figures/, LaTeX tables in ../paper/."
