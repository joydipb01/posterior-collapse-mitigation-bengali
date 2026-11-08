PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: setup setup-corpus test lint smoke corpus-smoke grid clean

setup:
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(BIN)/python -e ".[dev]"

setup-corpus:
	uv pip install --python $(BIN)/python -e ".[corpus]"

test:
	$(BIN)/python -m pytest

lint:
	$(BIN)/ruff check src tests

smoke:
	$(BIN)/python -m bcvae train --config configs/smoke.yaml --run-name smoke

corpus-smoke:
	$(BIN)/python -m bcvae build-corpus --config configs/corpus/smoke_jsonl.yaml --quiet

grid:
	$(BIN)/python -m bcvae grid --grid configs/grids/main.yaml

clean:
	rm -rf runs .pytest_cache .ruff_cache
