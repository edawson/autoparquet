from enum import Enum
from typing import Any, Optional, Union

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
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
    float_type: str = "float64",
    **kwargs: Any,
) -> None:
    """
    Writes data to a Parquet file with an optimized schema and custom header.

    Args:
        data: A pandas DataFrame, polars DataFrame, cuDF DataFrame, or Arrow Table.
        path: Output file path.
        header: Optional key-value metadata to store in the Parquet file.
        compression: Compression algorithm (ZSTD, SNAPPY, GZIP, etc.).
        compression_level: Compression level (algorithm-dependent).
        use_parquet_dictionary_compression: Let Parquet apply its own dictionary
            encoding pass on top of the schema-level optimizations.
        data_page_size: Parquet data page size in bytes.
        float_type: Target float precision ("float16", "float32", or "float64").
        **kwargs: Additional arguments forwarded to pyarrow.parquet.write_table.
    """
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

    # Codecs that don't support a compression level parameter
    if comp_str in ("SNAPPY", "LZ4"):
        compression_level = None

    table = to_arrow_table(data)
    optimized_schema = infer_schema(table, float_type=float_type)
    table = table.cast(optimized_schema, safe=False)

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
    Reads a Parquet file and returns (data, header_metadata).

    Args:
        path: Path to the Parquet file.
        engine: DataFrame engine to use ('pandas', 'polars', 'cudf', or 'auto').
                'auto' tries polars, then cudf, then pandas.
    """
    table = pq.read_table(path)

    metadata = table.schema.metadata or {}
    header = {k.decode("utf-8"): v.decode("utf-8") for k, v in metadata.items()}

    if isinstance(engine, str):
        engine = EngineType(engine.lower())
    auto = engine == EngineType.AUTO

    if engine == EngineType.POLARS or auto:
        if pl is not None:
            return pl.from_arrow(table), header
        if not auto:
            raise ImportError("polars is not installed but was explicitly requested.")

    if engine == EngineType.CUDF or auto:
        if cudf is not None:
            return cudf.from_arrow(table), header
        if not auto:
            raise ImportError("cudf is not installed but was explicitly requested.")

    if engine == EngineType.PANDAS or auto:
        if pd is not None:
            return table.to_pandas(), header
        if not auto:
            raise ImportError("pandas is not installed but was explicitly requested.")

    raise ImportError(
        f"No supported DataFrame library for engine '{engine.value}' is installed."
    )


def from_csv(
    path: str,
    delimiter: str = ",",
    quote_char: str = '"',
    escape_char: Optional[str] = None,
    float_type: str = "float64",
    **kwargs: Any,
) -> pa.Table:
    """
    Reads a CSV/TSV file directly into an optimized Arrow Table.

    Args:
        path: Path to the CSV/TSV file.
        delimiter: Field delimiter (e.g., ',' or '\\t').
        quote_char: Quoting character.
        escape_char: Escape character.
        float_type: Target float precision ("float16", "float32", or "float64").
        **kwargs: Additional arguments passed to pyarrow.csv.read_csv.
    """
    parse_options = pv.ParseOptions(
        delimiter=delimiter, quote_char=quote_char, escape_char=escape_char
    )
    table = pv.read_csv(path, parse_options=parse_options, **kwargs)

    optimized_schema = infer_schema(table, float_type=float_type)
    return table.cast(optimized_schema, safe=False)
