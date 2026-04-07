PYTHON ?= python
PROJECT_ROOT := $(shell pwd)

.PHONY: help install dev env grid pet diffusion-train diffusion-eval diffusion-notebook clean

help:
	@echo "NNDS make targets:"
	@echo "  make install           # Install Python deps"
	@echo "  make dev               # Install deps + set PYTHONPATH"
	@echo "  make grid              # Run traffic_analyzer grid + PET pipeline"
	@echo "  make pet               # Alias for grid (video-to-PET)"
	@echo "  make diffusion-train   # Train trajectory diffusion model"
	@echo "  make diffusion-eval    # Batch safety evaluation with saved checkpoint"
	@echo "  make diffusion-notebook# Notebook-style end-to-end diffusion eval"
	@echo "  make clean             # Remove common temporary artifacts"

install:
	$(PYTHON) -m pip install --upgrade pip
	pip install -r requirements.txt

dev: install
	@echo "Exporting PYTHONPATH for local shell sessions:"
	@echo "  export PYTHONPATH=$(PROJECT_ROOT):$$PYTHONPATH"

# === Video → Grid/PET pipeline ===
grid pet:
	PYTHONPATH=. $(PYTHON) traffic_analyzer.py

# === Diffusion training & evaluation ===

diffusion-train:
	PYTHONPATH=. $(PYTHON) traffic_diffusion/train_trajectory_diffusion.py

diffusion-eval:
	PYTHONPATH=. $(PYTHON) analysis/safety_eval_diffusion.py

diffusion-notebook:
	PYTHONPATH=. $(PYTHON) analysis/safety_eval_diffusion_notebook.py

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
