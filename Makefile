.PHONY: help install dev-install test lint format check clean version build upload prod

help:
	@echo "Available commands:"
	@echo "  install      : Install the package"
	@echo "  dev-install  : Install the package in editable mode with dev dependencies"
	@echo "  test         : Run tests using pytest"
	@echo "  lint         : Run ruff for linting"
	@echo "  format       : Run ruff for formatting"
	@echo "  check        : Run mypy for type checking"
	@echo "  clean        : Remove build artifacts and cache files"
	@echo "  version      : Show or update version (usage: make version VERSION=0.1.2)"
	@echo "  build        : Build the package distribution"
	@echo "  upload       : Upload to test PyPI (dry run before production)"
	@echo "  prod         : Upload to production PyPI (requires confirmation)"

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
	mypy src/autoparquet tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

version:
	@if [ -z "$(VERSION)" ]; then \
		echo "Current version:"; \
		grep -E "__version__|^version" src/autoparquet/__init__.py pyproject.toml | head -2; \
		echo ""; \
		echo "To set a new version, use: make version VERSION=0.1.2"; \
	else \
		sed -i '' 's/__version__ = "[^"]*"/__version__ = "$(VERSION)"/g' src/autoparquet/__init__.py; \
		sed -i '' 's/^version = "[^"]*"/version = "$(VERSION)"/g' pyproject.toml; \
		echo "Version updated to $(VERSION)"; \
		grep -E "__version__|^version" src/autoparquet/__init__.py pyproject.toml | head -2; \
	fi

build: clean
	python -m build
	@echo "Build complete. Files in dist/:"
	@ls -lh dist/

upload:
	@echo "Uploading to test PyPI..."
	twine upload --repository testpypi dist/*

prod:
	@echo "WARNING: About to upload to PRODUCTION PyPI"
	@echo "This cannot be undone. Make sure you've tested on test PyPI first."
	@read -p "Type 'yes' to confirm: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		twine upload dist/*; \
	else \
		echo "Upload cancelled."; \
	fi

benchmark:
	python benchmarks/bench.py
