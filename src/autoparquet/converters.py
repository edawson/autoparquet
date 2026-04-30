from typing import Any

import pyarrow as pa

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


def to_arrow_table(data: Any) -> pa.Table:
    """Converts various dataframe types to a PyArrow Table."""
    if isinstance(data, pa.Table):
        return data

    # Check for pandas
    if "pandas" in str(type(data)):
        if pd is None:
            raise ImportError(
                "pandas is not installed but a pandas object was passed. "
                "Please install pandas to use this functionality."
            )
        if isinstance(data, pd.DataFrame):
            return pa.Table.from_pandas(data)

    # Check for polars
    if "polars" in str(type(data)):
        if pl is None:
            raise ImportError(
                "polars is not installed but a polars object was passed. "
                "Please install polars to use this functionality."
            )
        if isinstance(data, pl.DataFrame):
            return data.to_arrow()

    # Check for objects with to_arrow (e.g. cudf)
    if hasattr(data, "to_arrow"):
        return data.to_arrow()

    raise ValueError(f"Unsupported data type: {type(data)}")
