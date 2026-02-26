# AutoSchema

AutoSchema is a Python package that wraps Parquet/Arrow to automatically generate optimized schemas for your data. It focuses on better compression through automatic bit-packing, int-packing, and dictionary encoding, while providing a convenient "header" system for storing custom metadata.

## AI-Generated Code with Human Review Note

```text
This repository was generated using Gemini 3 Flash, based on a specification written by the author. The code was reviewed by the author for correctness and tested locally.

Signed: Eric T. Dawson
```

## Features

- **Automatic Schema Inference**: Automatically detects the smallest possible integer and float types to save space.
- **Optimized Compression**: Uses bit-packing and dictionary encoding where appropriate.
- **Custom Headers**: Easily add and retrieve custom metadata (versioning, key-value pairs) in Parquet files.
- **Multi-Framework Support**: Works with Pandas, Polars, and cuDF.

## Installation

```bash
git clone https://github.com/edawson/autoschema
cd autoschema
pip install .
```


Coming soon: Pypi

```bash
pip install autoschema
```

## Usage

### Basic Example

```python
import pandas as pd
import autoschema

# Create a dataframe with mixed types
df = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "category": ["A", "B", "A", "B", "A"],
    "value": [1.1, 2.2, 3.3, 4.4, 5.5]
})

# Write with automatic schema optimization and a custom header
autoschema.write_parquet(
    df, 
    "data.parquet", 
    header={"version": "1.0", "author": "Eric T. Dawson"}
)

# Read back the data and the header
df_read, header = autoschema.read_parquet("data.parquet")
print(header)  # {'version': '1.0', 'author': 'Eric T. Dawson'}
```

### Automatic Schema Optimization

AutoSchema analyzes your data to find the most efficient storage format:

- **Integers**: Downcasts to the smallest possible bit-width (`uint8`, `int16`, `uint32`, etc.) based on the actual range of values in your dataset.
- **Floats**: Automatically converts `float64` to `float32` to save 50% space when extreme precision isn't required.
- **Strings/Binary**: Uses a heuristic to apply **Dictionary Encoding** to columns with low cardinality, significantly reducing file size for repetitive text data.

### Custom Headers (Metadata)

AutoSchema makes it easy to attach "headers" (key-value metadata) to your files. This is perfect for:
- Data versioning
- Tracking data lineage (source, timestamp, author)
- Storing processing parameters

```python
header = {
    "schema_version": "2.4.1",
    "captured_at": "2026-02-25",
    "environment": "production"
}
autoschema.write_parquet(df, "prod_data.parquet", header=header)
```

### Multi-Framework Support

AutoSchema is designed to be a drop-in wrapper for the most popular Python data frameworks.

#### Polars
```python
import polars as pl
import autoschema

lf = pl.DataFrame({"a": [1, 2, 3], "b": ["low", "low", "high"]})
autoschema.write_parquet(lf, "polars_data.parquet")
```

#### cuDF (NVIDIA GPU Dataframes)
```python
import cudf
import autoschema

gdf = cudf.DataFrame({"a": [1, 2, 3]})
autoschema.write_parquet(gdf, "gpu_data.parquet")
```

### Advanced Writing

You can pass any standard `pyarrow.parquet.write_table` arguments through `write_parquet`:

```python
autoschema.write_parquet(
    df, 
    "compressed.parquet", 
    compression="zstd", 
    compression_level=10,
    row_group_size=100_000
)
```
