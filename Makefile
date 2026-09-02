
PYTHON ?= python
PROJECT_ROOT := $(shell pwd)
VIDEO ?= data/sample_data/traffic_video.mp4
OUT_CSV ?= outputs/petevents_bev.csv
PET_THRESHOLD ?= 2.0

.PHONY: help install dev grid pet diffusion-train diffusion-eval diffusion-notebook smoke clean weights

help:
	@echo "NNDS make targets:"
	@echo "  make install            # Install Python deps"
	@echo "  make dev                # Install deps + set PYTHONPATH"
	@echo "  make grid               # Run video-to-PET pipeline (UVH-COCO fused)"
	@echo "  make pet                # Alias for grid"
	@echo "  make diffusion-train    # Train trajectory diffusion model"
	@echo "  make diffusion-eval     # Batch safety evaluation with saved checkpoint"
	@echo "  make diffusion-notebook # Notebook-style end-to-end diffusion eval"
	@echo "  make smoke              # Run all smoke tests"
	@echo "  make clean              # Remove common temporary artifacts"
	@echo "  make weights            # Download model weights (if available)"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

dev: install
	@echo "Exporting PYTHONPATH for local shell sessions:"
	@echo "  export PYTHONPATH=$(PROJECT_ROOT):$$PYTHONPATH"

grid pet:
	PYTHONPATH=. $(PYTHON) scripts/run_pipeline.py \
		--video $(VIDEO) \
		--detector uvh-coco-fused \
		--out-csv $(OUT_CSV) \
		--pet-threshold $(PET_THRESHOLD)

diffusion-train:
	PYTHONPATH=. $(PYTHON) src/diffusion/traffic_diffusion/train_trajectory_diffusion.py

diffusion-eval:
	PYTHONPATH=. $(PYTHON) src/analysis/analysis/safety_eval_diffusion.py

diffusion-notebook:
	PYTHONPATH=. $(PYTHON) src/analysis/analysis/safety_eval_diffusion_notebook.py

smoke:
	PYTHONPATH=. $(PYTHON) -m pytest -q tests/

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache

weights:
	bash scripts/download_models.sh


.PHONY: full-validate
full-validate:
	PYTHONPATH=. $(PYTHON) scripts/validate_all.py --run-e2e --frames 300 --check-models

reproduce-final:
	@echo "Checking for required video/model files..."
	@if [ ! -f data/sample_data/traffic_video.mp4 ] || [ ! -f data/models/uvh26.pt ]; then \
		echo "Video or model files missing. Skipping full pipeline. Using existing outputs for validation."; \
	else \
		PYTHONPATH=. python src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py --detector uvh-coco-fused; \
	fi
	PYTHONPATH=. python scripts/validate_outputs.py --detections tests/fixtures/sample_detections.csv --pet outputs/giti_screened.csv --video-frames 1838
	PYTHONPATH=. python scripts/generate_results_table.py
