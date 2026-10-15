# FusedBEV3D - the same five commands the README documents.
#
# PYTHON defaults to the local .venv when one exists (recommended), otherwise to
# the `python3` on your PATH. Override with `make PYTHON=... <target>`.

PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
EPOCHS ?= 12
SCENES ?= 40
VAL_SCENES ?= 10

.DEFAULT_GOAL := help
.PHONY: help data train eval track ablation test lint clean

help:
	@echo "make data      - materialise the synth train/val splits into data/cache"
	@echo "make train     - train the fused model into runs/fused (EPOCHS=$(EPOCHS))"
	@echo "make eval      - evaluate runs/fused/ckpt.pt -> results/detection_fused.json"
	@echo "make track     - track + CLEAR-MOT -> results/tracking_fused.json"
	@echo "make ablation  - camera-only / lidar-only / fused ablation -> results/ablation.*"
	@echo "make test      - ruff check + pytest"
	@echo "make clean     - drop caches, runs and generated results"

data:
	$(PYTHON) scripts/make_dataset.py --split train --scenes $(SCENES)
	$(PYTHON) scripts/make_dataset.py --split val --scenes $(VAL_SCENES)

train:
	$(PYTHON) scripts/train.py --config configs/fused.yaml --epochs $(EPOCHS) --out runs/fused --summary runs/fused/summary.json

eval:
	$(PYTHON) scripts/eval.py --ckpt runs/fused/ckpt.pt --out results/detection_fused.json \
		--occupancy-out results/occupancy_fused.json

track:
	$(PYTHON) scripts/track.py --ckpt runs/fused/ckpt.pt --out results/tracking_fused.json

ablation:
	$(PYTHON) scripts/run_ablation.py --epochs $(EPOCHS)

test:
	ruff check .
	$(PYTHON) -m pytest -q

lint:
	ruff check .

clean:
	rm -rf data/cache runs results/plots results/qualitative results/ablation.csv \
		results/ablation.md results/ablation_full.json results/detection_*.json \
		results/occupancy_*.json results/tracking_*.json results/train_summary_*.json
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
