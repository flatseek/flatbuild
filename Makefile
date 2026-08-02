PYTHON ?= python3
FLATBUILD = $(PYTHON) -m flatbuild

CONFIG ?= configs/demo.yaml
CHECKPOINT ?= outputs/demo/checkpoints/final

.PHONY: help install train evaluate resume export generate inspect benchmark demo test lint clean

help:
	@echo "Flatbuild Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make CONFIG=configs/xxx.yaml train     - Train a model from YAML config"
	@echo "  make CHECKPOINT=outputs/... evaluate   - Evaluate a trained checkpoint"
	@echo "  make CHECKPOINT=outputs/... resume     - Resume training from a checkpoint"
	@echo "  make CHECKPOINT=outputs/... export      - Export checkpoint to SafeTensors / HF"
	@echo "  make CHECKPOINT=outputs/... generate    - Generate text from a checkpoint"
	@echo "  make CHECKPOINT=outputs/... inspect     - Inspect a checkpoint"
	@echo "  make CHECKPOINT=outputs/... benchmark   - Benchmark a checkpoint"
	@echo ""
	@echo "  make install   - Install dependencies (pip install -e .)"
	@echo "  make demo      - Train the built-in demo dataset"
	@echo "  make test      - Run pytest"
	@echo "  make lint      - Run ruff"
	@echo "  make clean     - Remove outputs, logs, build artifacts"

install:
	$(PYTHON) -m pip install -e ".[dev]"

train:
	$(FLATBUILD) train $(CONFIG)

evaluate:
	$(FLATBUILD) evaluate $(CHECKPOINT)

resume:
	$(FLATBUILD) resume $(CHECKPOINT)

export:
	$(FLATBUILD) export $(CHECKPOINT)

generate:
	$(FLATBUILD) generate $(CHECKPOINT)

inspect:
	$(FLATBUILD) inspect $(CHECKPOINT)

benchmark:
	$(FLATBUILD) benchmark $(CHECKPOINT)

demo:
	$(FLATBUILD) train configs/demo.yaml

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m pip install ruff -q
	ruff check src/
	ruff format --check src/

clean:
	rm -rf outputs/
	rm -rf logs/
	rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
