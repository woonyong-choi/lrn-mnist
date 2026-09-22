PYTHON := .venv/bin/python
export OPENBLAS_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1
.PHONY: setup demo test train evaluate serve bench
setup:
	uv venv --python 3.12 .venv --allow-existing
	uv pip sync --python $(PYTHON) requirements.lock

test:
	$(PYTHON) -m pytest -q
demo:
	$(PYTHON) src/application.py demo
train:
	$(PYTHON) src/application.py train
evaluate:
	$(PYTHON) src/application.py evaluate --output .artifacts/evaluation/metrics.json
serve:
	$(PYTHON) src/application.py serve
bench:
	$(PYTHON) bench.py
