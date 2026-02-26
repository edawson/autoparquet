from typing import Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import polars as pl
except ImportError:
    pl = None


def to_arrow_table(data: Any) -> pa.Table:
    """Converts various dataframe types to a PyArrow Table."""
    if isinstance(data, pa.Table):
        return data
    if isinstance(data, pd.DataFrame):
        return pa.Table.from_pandas(data)
    if pl and isinstance(data, pl.DataFrame):
        return data.to_arrow()
    # Add support for cudf if available
    if hasattr(data, "to_arrow"):
        return data.to_arrow()

    raise ValueError(f"Unsupported data type: {type(data)}")


def write_parquet(
    data: Any, path: str, header: Optional[dict[str, str]] = None, **kwargs: Any
) -> None:
    """
    Writes data to a Parquet file with an optimized schema and custom header.
    """
    from .schema import infer_schema

    table = to_arrow_table(data)
    optimized_schema = infer_schema(table)

    # Cast table to optimized schema
    table = table.cast(optimized_schema)

    # Add header metadata
    if header:
        existing_metadata = optimized_schema.metadata or {}
        # Parquet metadata keys and values must be bytes
        new_metadata = {**existing_metadata}
        for k, v in header.items():
            new_metadata[k.encode("utf-8")] = str(v).encode("utf-8")

        table = table.replace_schema_metadata(new_metadata)

    pq.write_table(table, path, **kwargs)


def read_parquet(path: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Reads a Parquet file and returns the data and the custom header metadata.
    """
    table = pq.read_table(path)

    # Extract metadata
    metadata = table.schema.metadata or {}
    header = {k.decode("utf-8"): v.decode("utf-8") for k, v in metadata.items()}

    return table.to_pandas(), header
