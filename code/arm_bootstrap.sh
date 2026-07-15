#!/usr/bin/env bash
# Self-contained pipeline bootstrap for an ARM board (Jetson Orin Nano).
#
# On the board, from the repo root:
#     nohup bash code/arm_bootstrap.sh > arm.log 2>&1 &
#
# Caches all three ImageNet-V2 subsets (softmax + per-image timings, hours per
# subset) and then runs the full analysis, so every generated table/figure is
# measured on this board. Safe to re-run: cached subsets are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data figures paper

echo "=== $(hostname) preflight ==="
uname -m
nproc
free -h | head -2
df -h . | tail -1

# Prefer a venv; fall back to a pip --user install when python3-venv is
# unavailable (e.g. JetPack images holding python3.10 at an older build).
# Guard on importability, not venv existence, so an interrupted install is
# repaired on re-run instead of silently skipped.
DEPS="import numpy, torch, torchvision, sklearn, matplotlib, huggingface_hub"
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
if ! $PY -c "$DEPS" 2>/dev/null; then
    if python3 -m venv .venv 2>/dev/null && [ -x .venv/bin/pip ]; then
        PY=.venv/bin/python
        $PY -m pip install --upgrade pip
        # On aarch64, PyPI torch wheels are CPU-only, which is what we want.
        $PY -m pip install -r code/requirements.txt
    else
        echo "venv unavailable; installing into user site-packages."
        rm -rf .venv
        PY=python3
        $PY -m pip install --user -r code/requirements.txt
    fi
    $PY -c "$DEPS"   # fail loudly here if the install did not produce a working env
fi

for subset in matched top thr07; do
    if [ -f "data/${subset}/softmax_resnet152.npy" ]; then
        echo "data/${subset}/ already cached; skipping."
    else
        $PY code/cache_softmax.py --subset "${subset}" --cpu
    fi
done

$PY code/threshold_sweep.py
$PY code/skip_rule.py

echo "=== ARM pipeline complete: data/, ../figures/, ../paper/*.tex are board-measured ==="
