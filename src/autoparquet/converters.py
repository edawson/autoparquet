"""Convert various dataframe types to a PyArrow Table.

We avoid hard imports of optional dependencies (pandas, polars, cudf) so the
library stays usable in environments where only some are installed.
"""

from typing import Any

import pyarrow as pa

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dep
    pd = None  # type: ignore

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dep
    pl = None  # type: ignore


def to_arrow_table(data: Any) -> pa.Table:
    """Convert a DataFrame-like object to a PyArrow Table.

    Supported inputs:
      * ``pyarrow.Table`` (returned as-is)
      * ``pandas.DataFrame`` (when pandas is installed)
      * ``polars.DataFrame`` (when polars is installed)
      * Any object with a ``to_arrow()`` method (e.g. cuDF DataFrames)

    Raises:
        ValueError: when ``data`` is not one of the supported types.
    """
    if isinstance(data, pa.Table):
        return data

    if pd is not None and isinstance(data, pd.DataFrame):
        return pa.Table.from_pandas(data)

    if pl is not None and isinstance(data, pl.DataFrame):
        return data.to_arrow()

    # Anything else with a duck-typed conversion method (e.g. cudf.DataFrame).
    to_arrow = getattr(data, "to_arrow", None)
    if callable(to_arrow):
        result = to_arrow()
        if not isinstance(result, pa.Table):
            raise ValueError(
                f"Unsupported data type: {type(data).__name__}.to_arrow() "
                f"returned {type(result).__name__}, expected pyarrow.Table"
            )
        return result

    raise ValueError(
        f"Unsupported data type: {type(data).__name__}. "
        "Expected pyarrow.Table, pandas.DataFrame, polars.DataFrame, or an "
        "object with a to_arrow() method."
    )
