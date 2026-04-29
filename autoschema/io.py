from enum import Enum
from typing import Any, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pv
from pyarrow.parquet import write_table

from . import constants
from .converters import to_arrow_table
from .schema import infer_schema

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

try:
    import cudf
except ImportError:
    cudf = None  # type: ignore


class CompressionType(str, Enum):
    """Supported Parquet compression types."""

    NONE = "NONE"
    SNAPPY = "SNAPPY"
    GZIP = "GZIP"
    LZO = "LZO"
    BROTLI = "BROTLI"
    LZ4 = "LZ4"
    ZSTD = "ZSTD"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value.upper() in cls._member_names_


class EngineType(str, Enum):
    """Supported DataFrame engines for reading."""

    PANDAS = "pandas"
    POLARS = "polars"
    CUDF = "cudf"
    AUTO = "auto"


def write_parquet(
    data: Any,
    path: str,
    header: Optional[dict[str, str]] = None,
    compression: Union[str, CompressionType] = constants.DEFAULT_COMPRESSION,
    compression_level: Optional[int] = constants.DEFAULT_COMPRESSION_LEVEL,
    use_parquet_dictionary_compression: bool = True,
    data_page_size: int = constants.DEFAULT_PAGE_SIZE,
    **kwargs: Any,
) -> None:
    """
    Writes data to a Parquet file with an optimized schema and custom header.
    """
    # Validate compression type
    comp_str = (
        compression.value
        if isinstance(compression, CompressionType)
        else str(compression).upper()
    )
    if not CompressionType.is_valid(comp_str):
        valid_types = ", ".join(CompressionType._member_names_)
        raise ValueError(
            f"Invalid compression type: {compression}. Valid types are: {valid_types}"
        )

    table = to_arrow_table(data)
    optimized_schema = infer_schema(table)

    # Cast table to optimized schema
    table = table.cast(optimized_schema)

    # Add header metadata
    if header:
        existing_metadata = optimized_schema.metadata or {}
        new_metadata = {**existing_metadata}
        for k, v in header.items():
            new_metadata[k.encode("utf-8")] = str(v).encode("utf-8")

        table = table.replace_schema_metadata(new_metadata)

    write_table(
        table,
        path,
        compression=comp_str,
        compression_level=compression_level,
        use_dictionary=use_parquet_dictionary_compression,
        data_page_size=data_page_size,
        row_group_size=constants.DEFAULT_ROW_GROUP_SIZE,
        version=constants.DEFAULT_PARQUET_VERSION,
        **kwargs,
    )


def read_parquet(
    path: str, engine: Union[str, EngineType] = EngineType.AUTO
) -> tuple[Any, dict[str, str]]:
    """
    Reads a Parquet file and returns the data and the custom header metadata.
    Attempts to use polars or cudf if available, falling back to pandas.

    Args:
        path: Path to the Parquet file.
        engine: The DataFrame engine to use ('pandas', 'polars', 'cudf', or 'auto').
               Defaults to 'auto', which tries polars, then cudf, then pandas.
    """
    table = pq.read_table(path)

    # Extract metadata
    metadata = table.schema.metadata or {}
    header = {k.decode("utf-8"): v.decode("utf-8") for k, v in metadata.items()}

    engine_str = engine.value if isinstance(engine, EngineType) else str(engine).lower()

    if engine_str == EngineType.POLARS or engine_str == EngineType.AUTO:
        if pl is not None:
            return pl.from_arrow(table), header
        if engine_str == EngineType.POLARS:
            raise ImportError("polars is not installed but was explicitly requested.")

    if engine_str == EngineType.CUDF or engine_str == EngineType.AUTO:
        if cudf is not None:
            return cudf.from_arrow(table), header
        if engine_str == EngineType.CUDF:
            raise ImportError("cudf is not installed but was explicitly requested.")

    if engine_str == EngineType.PANDAS or engine_str == EngineType.AUTO:
        if pd is not None:
            return table.to_pandas(), header
        if engine_str == EngineType.PANDAS:
            raise ImportError("pandas is not installed but was explicitly requested.")

    raise ImportError(
        f"No supported DataFrame library for engine '{engine_str}' is installed."
    )


def from_csv(
    path: str,
    delimiter: str = ",",
    quote_char: str = '"',
    escape_char: Optional[str] = None,
    **kwargs: Any,
) -> pa.Table:
    """
    Reads a CSV/TSV file directly into an optimized Arrow Table.
    Uses pyarrow.csv for high-performance, multi-threaded ingestion.

    Args:
        path: Path to the CSV/TSV file.
        delimiter: Field delimiter (e.g., ',' or '\t').
        quote_char: Quoting character.
        escape_char: Escape character.
        **kwargs: Additional arguments passed to pyarrow.csv.read_csv.
    """
    parse_options = pv.ParseOptions(
        delimiter=delimiter, quote_char=quote_char, escape_char=escape_char
    )

    # Use pyarrow.csv.read_csv for fast, multi-threaded reading
    table = pv.read_csv(path, parse_options=parse_options, **kwargs)

    # Apply autoschema optimization
    optimized_schema = infer_schema(table)
    return table.cast(optimized_schema)
