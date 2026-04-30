# Contributing to AutoParquet

Thank you for your interest in contributing to AutoParquet! We welcome all contributions, from bug reports and feature requests to code improvements and documentation.

## Getting Started

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/autoparquet
   cd autoparquet
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev,polars]"
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Project Structure

```
autoparquet/
├── src/autoparquet/          # Main package source code
│   ├── __init__.py          # Package exports
│   ├── cli.py               # Command-line interface
│   ├── converters.py        # DataFrame to Arrow conversion
│   ├── constants.py         # Configuration constants
│   ├── io.py                # Parquet read/write operations
│   ├── schema.py            # Schema inference and optimization
│   └── transforms.py        # Data transformations (kmer tools, etc.)
├── tests/                    # Test suite
├── benchmarks/               # Performance benchmarks
├── docs/                     # Documentation
└── pyproject.toml           # Project configuration
```

## Running Tests

We use `pytest` for testing. Run the full test suite:

```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=src/autoparquet
```

Run a specific test:
```bash
pytest tests/test_io.py::test_basic_io
```

## Code Quality

### Linting and Formatting

We use `ruff` for linting and automatic code formatting:

```bash
# Check for linting issues
ruff check .

# Automatically format code
ruff format .
```

### Type Checking

We use `mypy` for static type checking:

```bash
mypy src/autoparquet tests
```

All code should be fully type-annotated.

### Using Make Commands

For convenience, use the provided Makefile:

```bash
make help          # Show available commands
make lint          # Run ruff linting
make format        # Auto-format with ruff
make check         # Run mypy type checking
make test          # Run pytest
make clean         # Remove build artifacts
```

## Code Style Guidelines

- **Type hints**: All functions must have complete type annotations
- **Docstrings**: Use clear, descriptive docstrings for public functions
- **Comments**: Comment why, not what. The code should be self-documenting
- **Line length**: Maximum 88 characters (enforced by ruff)
- **Imports**: Organize imports in groups: stdlib, third-party, local

## Submitting Changes

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit with clear, descriptive messages:
   ```bash
   git commit -m "Add feature: clear description of what was added"
   ```

3. Run tests and linting before pushing:
   ```bash
   make test
   make lint
   make check
   ```

4. Push to your fork and create a pull request on GitHub

5. Ensure the CI checks pass and address any review feedback

## Reporting Issues

When reporting bugs or requesting features:

- Check if the issue already exists
- Include your Python version and OS
- For bugs, provide a minimal reproducible example
- For features, explain the use case and expected behavior

## Benchmarking

If you're making performance-related changes, run the benchmark suite:

```bash
python benchmarks/bench.py
```

For custom compression configurations:
```bash
python benchmarks/bench.py --compression zstd:10 snappy gzip:6
```

## Questions?

Feel free to open an issue or discussion if you have questions about contributing!
