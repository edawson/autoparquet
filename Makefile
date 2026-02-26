.PHONY: help install dev-install test lint format check clean

help:
	@echo "Available commands:"
	@echo "  install      : Install the package"
	@echo "  dev-install  : Install the package in editable mode with dev dependencies"
	@echo "  test         : Run tests using pytest"
	@echo "  lint         : Run ruff for linting"
	@echo "  format       : Run ruff for formatting"
	@echo "  check        : Run mypy for type checking"
	@echo "  clean        : Remove build artifacts and cache files"

install:
	pip install .

dev-install:
	pip install -e ".[dev,polars]"
	pre-commit install

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

check:
	mypy autoschema tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
