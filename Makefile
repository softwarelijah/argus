# Argus developer workflow.
.PHONY: help install install-dev test lint fmt demo prepare train eval export benchmark docker clean

PYTHON ?= python
VISDRONE_ROOT ?= datasets/VisDrone
WEIGHTS ?= runs/visdrone/yolov8s-1280/weights/best.pt
SOURCE ?= input.mp4

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## install the core (CPU) package
	pip install -e .

install-dev: ## install with dev + train + export extras
	pip install -e ".[dev,train,export]"

test: ## run the test suite
	pytest

lint: ## run ruff
	ruff check argus tests scripts

fmt: ## auto-fix lint issues
	ruff check --fix argus tests scripts

demo: ## run the synthetic 60-target tracking demo
	$(PYTHON) scripts/demo_synthetic.py --targets 60 --frames 300 --output demo.mp4

prepare: ## convert VisDrone to YOLO format
	$(PYTHON) scripts/prepare_visdrone.py --root $(VISDRONE_ROOT)

train: ## fine-tune YOLOv8 on VisDrone
	$(PYTHON) scripts/train.py --config configs/train.yaml

eval: ## evaluate detection mAP
	$(PYTHON) scripts/evaluate.py --weights $(WEIGHTS)

export: ## export an INT8 TensorRT engine
	$(PYTHON) scripts/export_tensorrt.py --weights $(WEIGHTS) \
		--calib-images $(VISDRONE_ROOT)/VisDrone2019-DET-val/images --precision int8

benchmark: ## benchmark FPS / latency on a video
	$(PYTHON) scripts/benchmark.py $(SOURCE) --weights $(WEIGHTS)

docker: ## build the CPU docker image
	docker build -t argus:cpu .

clean: ## remove caches and build artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
