.PHONY: install test notebook reproduce

install:
	pip install -e ".[dev]"

test:
	pytest

notebook:
	jupytext --to ipynb notebooks/fd01_logistic_baseline.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/fd01_logistic_baseline.ipynb

reproduce: test notebook
