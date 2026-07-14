# IDK cascade skip predictor — end-to-end pipeline.
#
#   make            analysis + paper
#   make cache      cache softmax + timings for all three ImageNet-V2 subsets
#                   (hours of CPU; skipped per subset if data/<subset>/ exists)
#   make sweep      Nguyen 2024 threshold-sweep table + figure
#   make rules      cross-dataset skip-rule training/eval -> all paper tables
#   make rules-arm  same, sourcing test-subset timings from data/arm/<subset>/
#   make paper      compile paper/final_paper.pdf (pdflatex + bibtex)
#   make venv       .venv with CPU torch (only needed for caching)
#   make clean      remove LaTeX build artifacts

PY      := python3
VENV    := .venv
VENV_PY := $(VENV)/bin/python

SUBSETS := matched top thr07
CACHE_SENTINELS := $(foreach s,$(SUBSETS),data/$(s)/softmax_resnet152.npy)

.PHONY: all analysis sweep rules rules-arm cache paper venv clean

all: analysis paper

analysis: sweep rules

venv: $(VENV_PY)

$(VENV_PY):
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(VENV)/bin/pip install --quiet -r code/requirements.txt

cache: $(CACHE_SENTINELS)

# One cached subset. Deliberately a file target: re-running never overwrites
# the timings the paper's numbers are based on (delete data/<subset>/*.npy to force).
data/%/softmax_resnet152.npy: | $(VENV_PY)
	$(VENV_PY) code/cache_softmax.py --subset $* --cpu

sweep: data/matched/softmax_resnet152.npy
	$(PY) code/threshold_sweep.py

rules: $(CACHE_SENTINELS)
	$(PY) code/skip_rule.py

rules-arm: $(CACHE_SENTINELS)
	$(PY) code/skip_rule.py --timing-root data/arm

paper:
	cd paper && pdflatex -interaction=nonstopmode final_paper \
	  && bibtex final_paper \
	  && pdflatex -interaction=nonstopmode final_paper \
	  && pdflatex -interaction=nonstopmode final_paper

clean:
	rm -f paper/*.aux paper/*.log paper/*.bbl paper/*.blg paper/*.out paper/*.synctex.gz
