import pandas as pd
import pyarrow as pa
import pytest

from autoparquet.converters import to_arrow_table
from autoparquet.schema import _smallest_index_type, infer_schema
from autoparquet.transforms import (
    cast_to_fixed_binary,
    extract_string_vocabulary,
    map_to_vocabulary,
    strings_to_fixed_size_binary,
)


def test_map_to_vocabulary() -> None:
    df = pd.DataFrame({"kmer": ["AAAA", "CCCC", "GGGG", "TTTT", "Unknown"]})
    table = to_arrow_table(df)
    vocabulary = ["AAAA", "CCCC", "GGGG", "TTTT"]

    # Map to vocabulary
    mapped_table = map_to_vocabulary(table, "kmer", vocabulary)

    # Check type is dictionary
    field = mapped_table.schema.field("kmer")
    assert pa.types.is_dictionary(field.type)

    # Check index type is uint8 (since vocabulary size is 4)
    assert field.type.index_type == pa.uint8()

    # Check values
    # "Unknown" should be null
    result = mapped_table.column("kmer").to_pylist()
    assert result == ["AAAA", "CCCC", "GGGG", "TTTT", None]


def test_map_to_vocabulary_large() -> None:
    # Vocabulary size > 255 should use uint16
    vocabulary = [str(i) for i in range(300)]
    df = pd.DataFrame({"col": ["0", "299"]})
    table = to_arrow_table(df)

    mapped_table = map_to_vocabulary(table, "col", vocabulary)
    field = mapped_table.schema.field("col")
    assert field.type.index_type == pa.uint16()


def test_cast_to_fixed_binary() -> None:
    df = pd.DataFrame({"kmer": ["AAAA", "CCCC", "GGGG"]})
    table = to_arrow_table(df)

    # Cast to fixed binary
    cast_table = cast_to_fixed_binary(table, "kmer")

    # Check type
    field = cast_table.schema.field("kmer")
    assert pa.types.is_fixed_size_binary(field.type)
    assert field.type.byte_width == 4


def test_cast_to_fixed_binary_error() -> None:
    # Non-uniform length should raise ValueError
    df = pd.DataFrame({"kmer": ["AAAA", "CCC"]})
    table = to_arrow_table(df)

    with pytest.raises(ValueError, match="requires uniform length"):
        cast_to_fixed_binary(table, "kmer")


def test_strings_to_fixed_size_binary() -> None:
    df = pd.DataFrame(
        {"kmer": ["AAAA", "CCCC"], "other": ["X", "Y"], "mixed": ["A", "BB"]}
    )
    table = to_arrow_table(df)

    optimized_table = strings_to_fixed_size_binary(table)

    assert pa.types.is_fixed_size_binary(optimized_table.schema.field("kmer").type)
    assert pa.types.is_fixed_size_binary(optimized_table.schema.field("other").type)
    assert not pa.types.is_fixed_size_binary(optimized_table.schema.field("mixed").type)


def test_infer_schema_dictionary_downcast() -> None:
    # Test that infer_schema downcasts dictionary indices
    # Create a dictionary array with int32 indices but small dictionary
    indices = pa.array([0, 1], type=pa.int32())
    dictionary = pa.array(["A", "B"], type=pa.string())
    dict_array = pa.DictionaryArray.from_arrays(indices, dictionary)
    table = pa.table({"col": dict_array})

    schema = infer_schema(table)
    assert schema.field("col").type.index_type == pa.uint8()


def test_extract_string_vocabulary() -> None:
    # Test extracting and indexing a string column
    df = pd.DataFrame({"chromosome": ["chr1", "chr2", "chr10", "chr1", "chrX", "chr2"]})
    table = to_arrow_table(df)

    # Extract vocabulary
    indexed_table, vocab = extract_string_vocabulary(table, "chromosome")

    # Check vocabulary is sorted
    assert vocab == ["chr1", "chr10", "chr2", "chrX"]

    # Check column is now dictionary-encoded
    field = indexed_table.schema.field("chromosome")
    assert pa.types.is_dictionary(field.type)

    # Check index type is uint8 (4 unique values)
    assert field.type.index_type == pa.uint8()

    # Check indices directly (accessing the underlying DictionaryArray)
    dict_array = indexed_table.column("chromosome").chunks[0]
    indices = dict_array.indices.to_pylist()
    assert indices == [0, 2, 1, 0, 3, 2]  # Indices into sorted vocab


def test_extract_string_vocabulary_many_unique() -> None:
    # Test with many unique values (> 255)
    values = [f"name_{i:04d}" for i in range(500)]
    df = pd.DataFrame({"name": values})
    table = to_arrow_table(df)

    indexed_table, vocab = extract_string_vocabulary(table, "name")

    # Should have 500 unique values
    assert len(vocab) == 500

    # Index type should be uint16 (since 500 > 255)
    field = indexed_table.schema.field("name")
    assert field.type.index_type == pa.uint16()


# ---------------------------------------------------------------------------
# _smallest_index_type boundaries
# ---------------------------------------------------------------------------


def test_smallest_index_type_boundaries() -> None:
    assert _smallest_index_type(0) == pa.uint8()
    assert _smallest_index_type(255) == pa.uint8()
    assert _smallest_index_type(256) == pa.uint16()
    assert _smallest_index_type(65535) == pa.uint16()
    assert _smallest_index_type(65536) == pa.int32()
    assert _smallest_index_type(1_000_000) == pa.int32()


# ---------------------------------------------------------------------------
# infer_schema branch coverage
# ---------------------------------------------------------------------------


def test_infer_schema_uniform_length_strings() -> None:
    """High-cardinality, uniform-length string column must become fixed_size_binary."""
    # 100 unique 4-char strings — ratio 1.0 > 0.5, total >= 100 → not low-cardinality
    values = [f"{i:04d}" for i in range(100)]
    table = pa.table({"kmer": pa.array(values, type=pa.string())})

    schema = infer_schema(table)
    assert pa.types.is_fixed_size_binary(schema.field("kmer").type)
    assert schema.field("kmer").type.byte_width == 4


def test_infer_schema_timestamp_ns_to_us_naive() -> None:
    """Tz-naive timestamp[ns] must be downcast to timestamp[us]."""
    ts = pa.array([1_000_000, 2_000_000], type=pa.timestamp("ns"))
    table = pa.table({"ts": ts})

    schema = infer_schema(table)
    assert schema.field("ts").type == pa.timestamp("us")


def test_infer_schema_timestamp_ns_to_us_aware() -> None:
    """Tz-aware timestamp[ns] must be downcast to timestamp[us, tz=...]."""
    ts = pa.array([1_000_000, 2_000_000], type=pa.timestamp("ns", tz="UTC"))
    table = pa.table({"ts": ts})

    schema = infer_schema(table)
    assert schema.field("ts").type == pa.timestamp("us", tz="UTC")


def test_infer_schema_float16() -> None:
    """float_type='float16' must produce float16 fields."""
    table = pa.table({"f": pa.array([1.1, 2.2], type=pa.float64())})
    schema = infer_schema(table, float_type="float16")
    assert schema.field("f").type == pa.float16()


def test_infer_schema_all_null_int_column() -> None:
    """An all-null integer column must not crash and must preserve its original type."""
    col = pa.array([None, None, None], type=pa.int64())
    table = pa.table({"x": col})

    schema = infer_schema(table)
    assert schema.field("x").type == pa.int64()


def test_infer_schema_negative_integers() -> None:
    """Negative integer columns must downcast to the smallest signed type."""
    table = pa.table(
        {
            "i8": pa.array([-128, 127], type=pa.int64()),
            "i16": pa.array([-32768, 32767], type=pa.int64()),
            "i32": pa.array([-2_147_483_648, 2_147_483_647], type=pa.int64()),
        }
    )
    schema = infer_schema(table)
    assert schema.field("i8").type == pa.int8()
    assert schema.field("i16").type == pa.int16()
    assert schema.field("i32").type == pa.int32()


def test_infer_schema_boolean_preserved() -> None:
    """Boolean columns must pass through unchanged."""
    table = pa.table({"flag": pa.array([True, False, True])})
    schema = infer_schema(table)
    assert schema.field("flag").type == pa.bool_()


# ---------------------------------------------------------------------------
# infer_schema parameter validation
# ---------------------------------------------------------------------------


def test_infer_schema_invalid_float_type() -> None:
    table = pa.table({"f": [1.0, 2.0]})
    with pytest.raises(ValueError, match="float_type must be one of"):
        infer_schema(table, float_type="banana")


def test_infer_schema_rejects_non_table() -> None:
    with pytest.raises(TypeError, match="expects a pyarrow.Table"):
        infer_schema(pd.DataFrame({"a": [1, 2]}))  # type: ignore[arg-type]
