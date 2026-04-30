# AutoParquet Usage Guide

Complete documentation for using AutoParquet's features and API.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Automatic Schema Optimization](#automatic-schema-optimization)
3. [Custom Metadata](#custom-metadata)
4. [Framework Support](#framework-support)
5. [Command Line Interface](#command-line-interface)
6. [Advanced Features](#advanced-features)
7. [API Reference](#api-reference)
8. [Performance](#performance)

## Basic Usage

### Write Parquet Files

```python
import pandas as pd
import autoparquet

df = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "category": ["A", "B", "A", "B", "A"],
    "value": [1.1, 2.2, 3.3, 4.4, 5.5]
})

# Write with automatic schema optimization and custom metadata
autoparquet.write_parquet(
    df, 
    "data.parquet", 
    header={"version": "1.0", "author": "Your Name"}
)

# Read back the data and metadata
df_read, header = autoparquet.read_parquet("data.parquet")
print(header)  # {'version': '1.0', 'author': 'Your Name'}
```

### Advanced Writing

Pass any PyArrow Parquet options through `write_parquet`:

```python
autoparquet.write_parquet(
    df, 
    "compressed.parquet", 
    compression="zstd",
    compression_level=10,
    row_group_size=100_000
)
```

## Automatic Schema Optimization

AutoParquet analyzes your data and applies intelligent optimizations:

### Integer Downcasting

Integers are automatically reduced to the smallest type that can hold their values:

```python
df = pd.DataFrame({
    "small": [0, 255],           # Downcasts to uint8
    "medium": [-32768, 32767],   # Downcasts to int16
    "large": [0, 4294967295]     # Downcasts to uint32
})

autoparquet.write_parquet(df, "downcasted.parquet")
# Saves 50-75% on integer-heavy datasets
```

### String Dictionary Encoding

Low-cardinality string columns are automatically dictionary-encoded with optimized integer indices:

```python
# A column with 22 chromosome names gets uint8 indices (1 byte each)
# A column with 500 category names gets uint16 indices (2 bytes each)
df = pd.DataFrame({
    "chromosome": ["chr1", "chr2"] * 500,
    "category": [f"cat_{i}" for i in range(500)]
})

autoparquet.write_parquet(df, "data.parquet")

```

### Float Precision Control

By default, float64 precision is preserved. Reduce to float32 to halve storage:

```python
# Keep full precision (default)
autoparquet.write_parquet(df, "data.parquet")

# Use float32 for 50% size reduction
autoparquet.write_parquet(df, "data_compact.parquet", float_type="float32")
```

## Custom Metadata

Store versioning, lineage, and processing information in Parquet metadata:

```python
import autoparquet
from datetime import datetime

header = {
    "schema_version": "2.4.1",
    "captured_at": datetime.now().isoformat(),
    "environment": "production",
    "data_source": "biomarker_assay_v3",
    "processed_by": "pipeline_v2.1",
    "row_count": "1000000",
}

autoparquet.write_parquet(df, "prod_data.parquet", header=header)

# Later, retrieve the metadata
df_read, header_read = autoparquet.read_parquet("prod_data.parquet")
print(f"Source: {header_read['data_source']}")
print(f"Processed: {header_read['processed_by']}")
```

## Framework Support

### Pandas

```python
import pandas as pd
import autoparquet

df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
autoparquet.write_parquet(df, "data.parquet")
df_read, _ = autoparquet.read_parquet("data.parquet")
```

### Polars

```python
import polars as pl
import autoparquet

df = pl.DataFrame({"a": [1, 2, 3], "b": ["low", "low", "high"]})
autoparquet.write_parquet(df, "polars_data.parquet")
df_read, _ = autoparquet.read_parquet("polars_data.parquet", engine="polars")
```

### cuDF (NVIDIA GPU DataFrames)

```python
import cudf
import autoparquet

gdf = cudf.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
autoparquet.write_parquet(gdf, "gpu_data.parquet")
df_read, _ = autoparquet.read_parquet("gpu_data.parquet")
```

### Raw Arrow Tables

```python
import pyarrow as pa
import autoparquet

table = pa.table({
    "a": [1, 2, 3],
    "b": ["x", "y", "z"]
})
autoparquet.write_parquet(table, "arrow_data.parquet")
```

## Command Line Interface

### Basic Conversion

```bash
# Convert CSV to Parquet (uses zstd compression by default)
autoparquet csv_to_parquet data.csv

# Specify output path
autoparquet csv_to_parquet data.csv -o output.parquet

# Convert to Feather format
autoparquet csv_to_feather data.csv -o data.feather
```

### Compression Options

```bash
# Use different compression algorithms
autoparquet csv_to_parquet data.csv -c snappy
autoparquet csv_to_parquet data.csv -c gzip
autoparquet csv_to_parquet data.csv -c lz4

# Specify compression level
autoparquet csv_to_parquet data.csv -c gzip -l 6
autoparquet csv_to_parquet data.csv -c zstd -l 10
```

### CSV Options

```bash
# Tab-separated values
autoparquet csv_to_parquet data.tsv -d $'\t'

# Custom delimiter
autoparquet csv_to_parquet data.csv -d ';'

# Float precision
autoparquet csv_to_parquet data.csv -f float32  # Reduce precision
autoparquet csv_to_parquet data.csv -f float64  # Keep full precision
```

## Advanced Features

### Extracting String Vocabularies

For categorical columns with many unique values, you can extract the unique values and their indices using `extract_string_vocabulary`. This is useful when you want explicit control over the vocabulary or need to share it across files.

**Example:**

```python
import pandas as pd
import autoparquet

df = pd.DataFrame({
    "chromosome": ["chr1", "chr2", "chr1", "chrX"] * 250,
    "position": range(1000),
    "coverage": [x * 1.5 for x in range(1000)]
})

table = autoparquet.to_arrow_table(df)

# Extract vocabulary and index the column
table, vocab = autoparquet.extract_string_vocabulary(table, "chromosome")

print(f"Unique chromosomes: {vocab}")
# vocab = ["chr1", "chr10", "chr11", ..., "chrY"]  # Sorted

# The column is now dictionary-encoded with efficient integer indices
autoparquet.write_parquet(table, "genome_data.parquet")
```

**Note on Automatic Optimization:**

When you call `write_parquet()`, AutoParquet's `infer_schema` already automatically dictionary-encodes low-cardinality string columns. For example:
- A column with 22 chromosome names gets dictionary-encoded with uint8 indices
- A column with 123 unique strings gets dictionary-encoded with uint8 indices
- This happens automatically without any extra parameters

So in most cases, you don't need to manually call `extract_string_vocabulary` — just use `write_parquet()` directly and the optimization happens transparently.

#### Fixed-Size Binary Optimization

For uniform-length kmers, use fixed-size binary to eliminate string overhead:

```python
from autoparquet import cast_to_fixed_binary
import pandas as pd
import autoparquet

df = pd.DataFrame({
    "kmer": ["AAAA", "CCCC", "GGGG", "TTTT"],
    "count": [10, 5, 8, 12]
})

table = autoparquet.to_arrow_table(df)
table = cast_to_fixed_binary(table, "kmer")
autoparquet.write_parquet(table, "kmers.parquet")
# Saves 4 bytes per row on the kmer column
```

#### Stable Vocabulary Mapping

Ensure kmers map to consistent IDs across multiple files:

```python
import pandas as pd
import autoparquet
import itertools
from autoparquet import map_to_vocabulary

# 1. Create a stable vocabulary (e.g., all possible 4-mers)
# This ensures "AAAA" is always ID 0, "AAAC" is always ID 1, etc.
vocabulary = ["".join(p) for p in itertools.product("ACGT", repeat=4)]

# 2. Your kmer count data (may be sparse)
df = pd.DataFrame({
    "kmer": ["AAAC", "AAGT", "TTTT"],
    "count": [10, 5, 100]
})

# 3. Convert to Arrow and map to vocabulary
table = autoparquet.to_arrow_table(df)
table = map_to_vocabulary(table, "kmer", vocabulary)
# AutoParquet automatically uses uint8 (1 byte) for IDs since 256 < 256

# 4. Store vocabulary in metadata for future reference
header = {
    "kmer_k": 4,
    "vocabulary": ",".join(vocabulary),
    "total_kmers": str(len(vocabulary))
}

autoparquet.write_parquet(table, "kmers_stable.parquet", header=header)

# Later, you can reconstruct the vocabulary from metadata
df_read, header_read = autoparquet.read_parquet("kmers_stable.parquet")
vocabulary_restored = header_read["vocabulary"].split(",")
```

### Working with Arrow Tables Directly

```python
import pyarrow as pa
import autoparquet

# Create an Arrow table manually
table = pa.table({
    "id": [1, 2, 3],
    "value": [10.5, 20.1, 15.3]
})

# Infer optimized schema
schema = autoparquet.infer_schema(table)

# Write with optimization
autoparquet.write_parquet(table, "data.parquet")
```

## API Reference

### Core Functions

#### `write_parquet(data, path, header=None, **kwargs)`

Write data to Parquet with automatic schema optimization.

**Parameters:**
- `data` (DataFrame | Table): Pandas DataFrame, Polars DataFrame, cuDF DataFrame, or PyArrow Table
- `path` (str): Output file path
- `header` (dict, optional): Custom metadata dictionary
- `**kwargs`: Additional arguments passed to `pyarrow.parquet.write_table`
  - `compression` (str): Compression algorithm ("snappy", "gzip", "zstd", "lz4", "uncompressed")
  - `compression_level` (int): Compression level (algorithm-dependent)
  - `row_group_size` (int): Rows per row group

**Returns:** None

#### `read_parquet(path, engine="pandas")`

Read a Parquet file and extract metadata.

**Parameters:**
- `path` (str): Path to Parquet file
- `engine` (str): DataFrame library ("pandas", "polars", "arrow")

**Returns:** Tuple[DataFrame | Table, dict]
- DataFrame/Table: The data
- dict: Metadata from file header

#### `infer_schema(table)`

Infer an optimized Arrow schema for the given table.

**Parameters:**
- `table` (Table): PyArrow Table

**Returns:** pa.Schema
- Optimized schema with downcasted types and encoding recommendations

#### `to_arrow_table(data)`

Convert a DataFrame to an optimized Arrow Table.

**Parameters:**
- `data` (DataFrame): Pandas, Polars, or cuDF DataFrame

**Returns:** pa.Table
- Arrow table with optimized schema

#### `cast_to_fixed_binary(table, column_name)`

Convert a string column to fixed-size binary.

**Parameters:**
- `table` (Table): PyArrow Table
- `column_name` (str): Column name with uniform-length strings

**Returns:** pa.Table

**Raises:** ValueError if strings have non-uniform lengths

#### `map_to_vocabulary(table, column_name, vocabulary)`

Map string values to dictionary-encoded IDs.

**Parameters:**
- `table` (Table): PyArrow Table
- `column_name` (str): Column to encode
- `vocabulary` (list[str]): List of valid values

**Returns:** pa.Table
- Table with column dictionary-encoded using provided vocabulary
- Values not in vocabulary become null

#### `strings_to_fixed_size_binary(table)`

Automatically convert uniform-length string columns to fixed-size binary.

**Parameters:**
- `table` (Table): PyArrow Table

**Returns:** pa.Table
- Columns with uniform-length strings converted to FixedSizeBinary

#### `extract_string_vocabulary(table, column_name)`

Extract unique string values from a column and create integer-indexed mapping.

Returns the table with the column replaced by dictionary-encoded integer indices
and a sorted list of unique values. The indices use the smallest possible type
(uint8, uint16, int32) to minimize storage.

**Parameters:**
- `table` (Table): PyArrow Table
- `column_name` (str): Column to extract vocabulary from

**Returns:** Tuple[pa.Table, list[str]]
- Tuple of (table with indexed column, sorted vocabulary list)

**Example:**
```python
table, vocab = autoparquet.extract_string_vocabulary(table, "chromosome")
# vocab = ["chr1", "chr10", "chr11", ..., "chrX", "chrY"]
# table.column("chromosome") is now indexed with uint8 (for ~22 values)

# Store vocabulary in metadata for sharing across files
header = {"chromosome_vocab": ",".join(vocab)}
autoparquet.write_parquet(table, "data.parquet", header=header)
```

## Performance

### Typical Compression Improvements

Compared to standard Parquet with zstd compression:

- **Integer-heavy datasets**: 50-75% reduction (via downcasting)
- **Low-cardinality strings**: 60-80% reduction (via dictionary encoding)
- **Uniform-length strings**: 10-20% reduction (via fixed-size binary)
- **Float-heavy datasets**: 50% reduction (via float32 precision)

### Running Benchmarks

AutoParquet includes comprehensive benchmarks across various datasets:

```bash
# Run default benchmarks (using zstd compression)
python benchmarks/bench.py

# Test multiple compression algorithms
python benchmarks/bench.py --compression zstd snappy gzip

# Test with compression levels
python benchmarks/bench.py --compression zstd:3 zstd:10 snappy
```

Benchmarks test against:
- CSV files
- Standard Parquet
- Apache Feather
- AutoParquet with various compression configs

See `benchmarks/bench.py` for implementation details.
