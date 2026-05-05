"""autoparquet: optimized Parquet schemas with minimal effort.

The public API is intentionally small. The most common entry points are
:func:`write_parquet` and :func:`read_parquet`; the rest are exposed for
callers that want finer control or want to share the schema-inference
machinery without touching files.
"""

from .converters import to_arrow_table
from .io import (
    CompressionType,
    EngineType,
    from_csv,
    from_excel,
    read_parquet,
    to_excel,
    write_parquet,
)
from .schema import infer_schema
from .transforms import (
    cast_to_fixed_binary,
    extract_string_vocabulary,
    map_to_vocabulary,
    strings_to_fixed_size_binary,
)

__version__ = "0.1.5"

__all__ = [
    # Parquet I/O
    "read_parquet",
    "write_parquet",
    "CompressionType",
    "EngineType",
    # CSV / Excel I/O
    "from_csv",
    "from_excel",
    "to_excel",
    # Schema inference & conversion
    "infer_schema",
    "to_arrow_table",
    # Transforms
    "map_to_vocabulary",
    "extract_string_vocabulary",
    "cast_to_fixed_binary",
    "strings_to_fixed_size_binary",
]
