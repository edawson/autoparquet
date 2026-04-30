# AutoParquet

AutoParquet is a Python package that wraps Parquet/Arrow to automatically generate optimized schemas for your data. It focuses on better compression through automatic bit-packing, int-packing, and dictionary encoding, while providing a convenient "header" system for storing custom metadata.

## Quick Start

### Installation

```bash
git clone https://github.com/edawson/autoparquet
cd autoparquet
pip install -e .
```

Or with dev dependencies:

```bash
pip install -e ".[dev,polars]"
```

### Basic Example

```python
import pandas as pd
import autoparquet

df = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "category": ["A", "B", "A", "B", "A"],
    "value": [1.1, 2.2, 3.3, 4.4, 5.5]
})

# Write with automatic schema optimization and a custom header
autoparquet.write_parquet(
    df, 
    "data.parquet", 
    header={"version": "1.0", "author": "Eric T. Dawson"}
)

# Read back the data and the header
df_read, header = autoparquet.read_parquet("data.parquet")
print(header)  # {'version': '1.0', 'author': 'Eric T. Dawson'}
```

### Command Line

```bash
# Convert CSV to optimized Parquet (default: zstd compression)
autoparquet csv_to_parquet data.csv

# Convert CSV to Feather with custom options
autoparquet csv_to_feather data.csv -o out.feather -c snappy

# Tab-separated input with reduced float precision
autoparquet csv_to_parquet data.tsv -d $'\t' -f float32
```

## Features

- **Automatic Schema Inference**: Downcasts integers to the smallest type that fits and optionally reduces float precision.
- **Optimized Compression**: Dictionary-encodes low-cardinality strings with the smallest index type; converts uniform-length strings to FixedSizeBinary.
- **Custom Headers**: Easily add and retrieve custom metadata (versioning, key-value pairs) in Parquet files.
- **Multi-Framework Support**: Works with Pandas, Polars, and cuDF.
- **CLI**: Convert CSV files to optimized Parquet or Feather from the command line.

## Documentation

- [Usage Guide](docs/usage.md) - Detailed examples, API reference, and advanced features
- [Contributing](CONTRIBUTING.md) - Development setup and contribution guidelines
- [Compression Roadmap](docs/compression-roadmap.md) - Future optimization techniques
- [Performance Analysis](docs/performance-analysis.md) - Casino and weather dataset optimization study

## Development

```bash
# Run tests
pytest

# Lint and format
ruff check .
ruff format .

# Type checking
mypy src/autoparquet tests

# Or use make
make test
make lint
make check
```

## Requirements

- Python 3.9+
- PyArrow 14.0.0+
- Pandas 2.0.0+ (optional)
- Polars 0.20.0+ (optional)

## License

MIT License

## Citation

```bibtex
@software{autoparquet2026,
  author = {Dawson, Eric T.},
  title = {AutoParquet: Automatic Schema Optimization for Parquet Files},
  year = {2026},
  url = {https://github.com/erictdawson/autoparquet}
}
```

---

This repository was generated using Gemini 3 Flash, based on a specification written by the author. The code was reviewed by the author for correctness and tested locally.
