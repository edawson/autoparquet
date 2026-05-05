import json
import pathlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autoparquet import read_parquet, write_parquet
from autoparquet.io import _PROMOTED_STRING_COLUMNS_KEY
from autoparquet.transforms import cast_to_fixed_binary


def test_basic_io(tmp_path: pathlib.Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
    path = str(tmp_path / "test.parquet")
    header = {"key": "value"}

    write_parquet(df, path, header=header)

    # Read back the raw Arrow table to verify schema optimization
    table = pq.read_table(path)
    assert pa.types.is_dictionary(table.schema.field("b").type)

    # Verify round-trip through read_parquet
    df_read, header_read = read_parquet(path, engine="pandas")
    assert len(df_read) == 3
    assert header_read["key"] == "value"


def test_integer_downcasting(tmp_path: pathlib.Path) -> None:
    df = pd.DataFrame(
        {"small": [0, 255], "medium": [-32768, 32767], "large": [0, 4294967295]}
    )
    path = str(tmp_path / "downcast.parquet")
    write_parquet(df, path)

    table = pq.read_table(path)
    assert table.schema.field("small").type == pa.uint8()
    assert table.schema.field("medium").type == pa.int16()
    assert table.schema.field("large").type == pa.uint32()


def test_dictionary_encoding(tmp_path: pathlib.Path) -> None:
    # High cardinality - should not be encoded
    df_high = pd.DataFrame({"col": [str(i) for i in range(1000)]})
    # Low cardinality - should be encoded
    df_low = pd.DataFrame({"col": ["A", "B"] * 500})

    path_high = str(tmp_path / "high.parquet")
    path_low = str(tmp_path / "low.parquet")

    write_parquet(df_high, path_high)
    write_parquet(df_low, path_low)

    table_high = pq.read_table(path_high)
    table_low = pq.read_table(path_low)

    assert not pa.types.is_dictionary(table_high.schema.field("col").type)
    assert pa.types.is_dictionary(table_low.schema.field("col").type)


def test_float_downcasting(tmp_path: pathlib.Path) -> None:
    df = pd.DataFrame({"f": [1.1, 2.2]}, dtype="float64")
    path = str(tmp_path / "float.parquet")
    write_parquet(df, path, float_type="float32")

    table = pq.read_table(path)
    assert table.schema.field("f").type == pa.float32()


def test_float_preserves_precision_by_default(tmp_path: pathlib.Path) -> None:
    df = pd.DataFrame({"f": [1.1, 2.2]}, dtype="float64")
    path = str(tmp_path / "float_default.parquet")
    write_parquet(df, path)

    table = pq.read_table(path)
    assert table.schema.field("f").type == pa.float64()


def test_io_with_polars(tmp_path: pathlib.Path) -> None:
    try:
        import polars as pl
    except ImportError:
        pytest.skip("polars not installed")

    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
    path = str(tmp_path / "polars.parquet")
    write_parquet(df, path)

    df_read, _ = read_parquet(path, engine="pandas")
    assert isinstance(df_read, pd.DataFrame)
    assert len(df_read) == 3


def test_io_unsupported_type() -> None:
    with pytest.raises(ValueError, match="Unsupported data type"):
        write_parquet([1, 2, 3], "test.parquet")


# ---------------------------------------------------------------------------
# Round-trip type fidelity
# ---------------------------------------------------------------------------


def test_roundtrip_low_cardinality_string(tmp_path: pathlib.Path) -> None:
    """Low-cardinality strings must come back as plain strings, not Categorical."""
    df = pd.DataFrame({"cat": ["A", "B", "A", "C"] * 50})
    path = str(tmp_path / "lowcard.parquet")
    write_parquet(df, path)

    # Verify the file contains a dictionary-encoded column (schema optimised)
    raw = pq.read_table(path)
    assert pa.types.is_dictionary(raw.schema.field("cat").type)

    # But read_parquet must decode it back to plain strings
    df_back, _ = read_parquet(path, engine="pandas")
    assert not isinstance(df_back["cat"].dtype, pd.CategoricalDtype), (
        "expected plain string, got Categorical"
    )
    assert df_back["cat"].tolist() == df["cat"].tolist()


def test_roundtrip_uniform_length_string_decoded(tmp_path: pathlib.Path) -> None:
    """High-cardinality uniform-length strings promoted to fixed_size_binary must be
    decoded back to plain strings on read."""
    # 100 unique 4-char strings, all distinct → ratio=1.0 → not low-cardinality
    # → infer_schema promotes to fixed_size_binary(4)
    values = [f"{i:04d}" for i in range(100)]
    table = pa.table({"kmer": pa.array(values, type=pa.string())})
    path = str(tmp_path / "fsb.parquet")
    write_parquet(table, path)

    # Verify it was actually promoted in the file
    raw = pq.read_table(path)
    assert pa.types.is_fixed_size_binary(raw.schema.field("kmer").type), (
        "expected fixed_size_binary promotion in file"
    )

    # read_parquet must reverse the cast to strings
    df_back, _ = read_parquet(path, engine="pandas")
    assert pd.api.types.is_string_dtype(
        df_back["kmer"]
    ) or pd.api.types.is_object_dtype(df_back["kmer"]), (
        "expected plain string dtype after decode"
    )
    assert df_back["kmer"].tolist() == values


def test_roundtrip_binary_dict_stays_bytes(tmp_path: pathlib.Path) -> None:
    """dictionary(uint8, binary) columns must NOT be decoded to string on read."""
    indices = pa.array([0, 1, 0, 1], type=pa.uint8())
    dictionary = pa.array([b"AA", b"TT"], type=pa.binary())
    dict_col = pa.DictionaryArray.from_arrays(indices, dictionary)
    table = pa.table({"seq": dict_col})
    path = str(tmp_path / "binary_dict.parquet")
    write_parquet(table, path)

    df_back, _ = read_parquet(path, engine="pandas")
    # Values should remain bytes objects, not decoded to str
    non_null = [v for v in df_back["seq"] if v is not None]
    assert all(isinstance(v, bytes) for v in non_null), (
        "binary-valued dict column should stay as bytes on read"
    )


def test_roundtrip_null_preservation(tmp_path: pathlib.Path) -> None:
    """Null values must survive a write/read round-trip unchanged."""
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"]})
    path = str(tmp_path / "nulls.parquet")
    write_parquet(df, path)

    df_back, _ = read_parquet(path, engine="pandas")
    assert df_back["a"].isna().tolist() == [False, True, False]
    assert df_back["b"].isna().tolist() == [False, True, False]


def test_roundtrip_float32_approximate_equality(tmp_path: pathlib.Path) -> None:
    """float64 written as float32 must be approximately equal on read back (documents
    the acceptable precision loss)."""
    values = [1.123456789, 2.987654321, 3.14159265358979]
    df = pd.DataFrame({"f": values})
    path = str(tmp_path / "float32.parquet")
    write_parquet(df, path, float_type="float32")

    df_back, _ = read_parquet(path, engine="pandas")
    assert np.allclose(df_back["f"].values, values, atol=1e-5), (
        "float32 round-trip values should be approximately equal to float64 originals"
    )


# ---------------------------------------------------------------------------
# Metadata sentinel for promoted string columns
# ---------------------------------------------------------------------------


def test_write_parquet_stores_string_column_metadata(tmp_path: pathlib.Path) -> None:
    """write_parquet must record promoted string columns in file metadata."""
    values = [f"{i:04d}" for i in range(100)]
    table = pa.table({"kmer": pa.array(values, type=pa.string())})
    path = str(tmp_path / "meta.parquet")
    write_parquet(table, path)

    raw_meta = pq.read_table(path).schema.metadata or {}
    key = _PROMOTED_STRING_COLUMNS_KEY.encode()
    assert key in raw_meta, "expected __autoparquet_string_columns__ in metadata"
    promoted = json.loads(raw_meta[key].decode())
    assert "kmer" in promoted


def test_write_parquet_no_metadata_for_explicit_binary(tmp_path: pathlib.Path) -> None:
    """Columns explicitly cast to fixed_size_binary by the caller must NOT appear in
    the promoted-strings metadata, since the caller made a deliberate choice."""
    df = pd.DataFrame({"kmer": ["AAAA", "CCCC", "GGGG"] * 34})
    table = pa.Table.from_pandas(df)
    table = cast_to_fixed_binary(table, "kmer")
    path = str(tmp_path / "explicit_binary.parquet")
    write_parquet(table, path)

    raw_meta = pq.read_table(path).schema.metadata or {}
    key = _PROMOTED_STRING_COLUMNS_KEY.encode()
    if key in raw_meta:
        promoted = json.loads(raw_meta[key].decode())
        assert "kmer" not in promoted, (
            "explicitly cast column should not appear in promoted-string metadata"
        )


# ---------------------------------------------------------------------------
# Invalid arguments and compression edge cases
# ---------------------------------------------------------------------------


def test_write_parquet_invalid_compression(tmp_path: pathlib.Path) -> None:
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="Invalid compression type"):
        write_parquet(df, str(tmp_path / "bad.parquet"), compression="BZIP2")


def test_write_parquet_snappy_no_level(tmp_path: pathlib.Path) -> None:
    """SNAPPY does not accept a compression level; write_parquet must not error."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    path = str(tmp_path / "snappy.parquet")
    write_parquet(df, path, compression="SNAPPY", compression_level=5)
    assert pathlib.Path(path).exists()


def test_write_parquet_column_encoding(tmp_path: pathlib.Path) -> None:
    """column_encoding parameter must be forwarded to PyArrow without error."""
    df = pd.DataFrame({"pos": list(range(100))})
    path = str(tmp_path / "encoding.parquet")
    write_parquet(df, path, column_encoding={"pos": "DELTA_BINARY_PACKED"})
    assert pathlib.Path(path).exists()


# ---------------------------------------------------------------------------
# Engine matrix
# ---------------------------------------------------------------------------


def test_read_parquet_explicit_engine_polars(tmp_path: pathlib.Path) -> None:
    pl = pytest.importorskip("polars")
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = str(tmp_path / "polars_engine.parquet")
    write_parquet(df, path)

    result, _ = read_parquet(path, engine="polars")
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 3


def test_read_parquet_invalid_engine_string(tmp_path: pathlib.Path) -> None:
    df = pd.DataFrame({"a": [1]})
    path = str(tmp_path / "inv.parquet")
    write_parquet(df, path)
    with pytest.raises(ValueError):
        read_parquet(path, engine="notanengine")


def test_read_parquet_explicit_missing_engine(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autoparquet.io

    df = pa.table({"a": [1, 2, 3]})
    path = str(tmp_path / "missing_engine.parquet")
    write_parquet(df, path)

    monkeypatch.setattr(autoparquet.io, "pl", None)
    with pytest.raises(ImportError, match="polars is not installed"):
        read_parquet(path, engine="polars")


def test_io_no_pandas(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import autoparquet.io

    monkeypatch.setattr(autoparquet.io, "pd", None)
    monkeypatch.setattr(autoparquet.io, "pl", None)
    monkeypatch.setattr(autoparquet.io, "cudf", None)

    df = pa.table({"a": [1, 2, 3]})
    path = str(tmp_path / "no_pandas.parquet")
    write_parquet(df, path)

    with pytest.raises(ImportError, match="No supported DataFrame library"):
        read_parquet(path)
