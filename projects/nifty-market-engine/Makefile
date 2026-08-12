.PHONY: test lint typecheck demo

test:
	pytest

lint:
	ruff check src tests

typecheck:
	mypy src

demo:
	python -m nifty_engine.cli paper-demo
