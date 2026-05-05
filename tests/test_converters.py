"""Unit tests for autoparquet.converters.to_arrow_table."""

from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from autoparquet.converters import to_arrow_table


def test_to_arrow_table_passthrough() -> None:
    """An Arrow Table must be returned as-is (identity)."""
    table = pa.table({"a": [1, 2, 3]})
    result = to_arrow_table(table)
    assert result is table


def test_to_arrow_table_pandas() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    result = to_arrow_table(df)
    assert isinstance(result, pa.Table)
    assert result.num_rows == 2
    assert set(result.schema.names) == {"a", "b"}


def test_to_arrow_table_polars() -> None:
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    result = to_arrow_table(df)
    assert isinstance(result, pa.Table)
    assert result.num_rows == 2


def test_to_arrow_table_has_to_arrow_method() -> None:
    """Objects with a .to_arrow() method (e.g. cudf) must be supported."""
    expected = pa.table({"x": [10, 20]})
    mock_df = MagicMock()
    mock_df.to_arrow.return_value = expected
    # Ensure the type name doesn't contain "pandas" or "polars"
    mock_df.__class__.__name__ = "MockFrame"

    result = to_arrow_table(mock_df)
    assert result is expected
    mock_df.to_arrow.assert_called_once()


def test_to_arrow_table_unsupported() -> None:
    with pytest.raises(ValueError, match="Unsupported data type"):
        to_arrow_table([1, 2, 3])


def test_to_arrow_table_unsupported_dict() -> None:
    with pytest.raises(ValueError, match="Unsupported data type"):
        to_arrow_table({"a": [1, 2]})
