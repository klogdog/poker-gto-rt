.PHONY: install serve test

install:
	python3.12 -m venv .venv
	.venv/bin/python -m pip install '.[test]'

serve:
	.venv/bin/python -m gtosolver

test:
	.venv/bin/python -m pytest -q
