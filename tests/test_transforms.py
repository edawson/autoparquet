"""Edge-case and error-path tests for autoparquet.transforms."""

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from autoparquet.transforms import (
    cast_to_fixed_binary,
    extract_string_vocabulary,
    map_to_vocabulary,
    strings_to_fixed_size_binary,
)

# ---------------------------------------------------------------------------
# map_to_vocabulary
# ---------------------------------------------------------------------------


def test_map_to_vocabulary_column_not_found() -> None:
    table = pa.table({"a": pa.array(["x", "y"])})
    with pytest.raises(ValueError, match="Column 'missing' not found"):
        map_to_vocabulary(table, "missing", ["x", "y"])


def test_map_to_vocabulary_chunked_column() -> None:
    """Correct index mapping must hold across a multi-chunk array.

    The chunk loop in map_to_vocabulary encodes each chunk independently using
    the same vocabulary. Values not in the vocabulary must map to null in every
    chunk.
    """
    vocab = ["A", "C", "G", "T"]

    chunk1 = pa.array(["A", "C", "X"], type=pa.string())
    chunk2 = pa.array(["G", "T", "A"], type=pa.string())
    chunked = pa.chunked_array([chunk1, chunk2])
    table = pa.table({"kmer": chunked})

    mapped = map_to_vocabulary(table, "kmer", vocab)
    col = mapped.column("kmer")

    # Decode to Python to check values regardless of chunk boundaries
    decoded = pc.cast(col, pa.string()).to_pylist()
    assert decoded == ["A", "C", None, "G", "T", "A"]

    # All chunks must use the same vocabulary array
    for chunk in col.chunks:
        assert chunk.dictionary.to_pylist() == vocab


# ---------------------------------------------------------------------------
# cast_to_fixed_binary
# ---------------------------------------------------------------------------


def test_cast_to_fixed_binary_column_not_found() -> None:
    table = pa.table({"a": pa.array(["xx", "yy"])})
    with pytest.raises(ValueError, match="Column 'missing' not found"):
        cast_to_fixed_binary(table, "missing")


def test_cast_to_fixed_binary_all_null() -> None:
    col = pa.array([None, None], type=pa.string())
    table = pa.table({"seq": col})
    with pytest.raises(ValueError, match="empty or contains only nulls"):
        cast_to_fixed_binary(table, "seq")


# ---------------------------------------------------------------------------
# extract_string_vocabulary
# ---------------------------------------------------------------------------


def test_extract_string_vocabulary_column_not_found() -> None:
    table = pa.table({"a": pa.array(["x"])})
    with pytest.raises(ValueError, match="Column 'missing' not found"):
        extract_string_vocabulary(table, "missing")


# ---------------------------------------------------------------------------
# strings_to_fixed_size_binary
# ---------------------------------------------------------------------------


def test_strings_to_fixed_size_binary_binary_input() -> None:
    """binary (not string) columns with uniform length must also be converted."""
    col = pa.array([b"AAAA", b"CCCC", b"GGGG"], type=pa.binary())
    table = pa.table({"seq": col})

    result = strings_to_fixed_size_binary(table)
    assert pa.types.is_fixed_size_binary(result.schema.field("seq").type)
    assert result.schema.field("seq").type.byte_width == 4


def test_strings_to_fixed_size_binary_skips_mixed_length() -> None:
    """Columns with non-uniform lengths must not be converted."""
    col = pa.array(["A", "BB", "CCC"], type=pa.string())
    table = pa.table({"seq": col})

    result = strings_to_fixed_size_binary(table)
    assert not pa.types.is_fixed_size_binary(result.schema.field("seq").type)


def test_strings_to_fixed_size_binary_preserves_other_columns() -> None:
    """Non-string columns alongside a promoted column must pass through unchanged."""
    table = pa.table(
        {
            "kmer": pa.array(["AAAA", "CCCC"], type=pa.string()),
            "count": pa.array([10, 20], type=pa.int32()),
        }
    )
    result = strings_to_fixed_size_binary(table)

    assert pa.types.is_fixed_size_binary(result.schema.field("kmer").type)
    assert result.schema.field("count").type == pa.int32()
