# Contributing to AutoSchema

Thank you for your interest in contributing to AutoSchema!

## Development Setup

1. Clone the repository.
2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Running Tests

We use `pytest` for testing:
```bash
pytest
```

## Linting

We use `ruff` for linting and formatting:
```bash
ruff check .
ruff format .
```
