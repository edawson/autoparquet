from enum import Enum
from typing import Any, Optional, Union

import pyarrow.parquet as pq

from . import constants
from .converters import to_arrow_table
from .schema import infer_schema

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore


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

    pq.write_table(
        table,
        path,
        compression=comp_str,
        compression_level=compression_level,
        use_dictionary=use_parquet_dictionary_compression,
        data_page_size=data_page_size,
        **kwargs,
    )


def read_parquet(path: str) -> tuple[Any, dict[str, str]]:
    """
    Reads a Parquet file and returns the data and the custom header metadata.
    """
    if pd is None:
        raise ImportError(
            "pandas is not installed. pandas is required for read_parquet "
            "to return a DataFrame."
        )

    table = pq.read_table(path)

    # Extract metadata
    metadata = table.schema.metadata or {}
    header = {k.decode("utf-8"): v.decode("utf-8") for k, v in metadata.items()}

    return table.to_pandas(), header
