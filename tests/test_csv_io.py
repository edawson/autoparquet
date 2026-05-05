"""Integration tests for CSV reading and full CSV → Parquet → read round-trips."""

import pathlib

import pandas as pd
import pyarrow as pa
import pytest

from autoparquet.io import from_csv, read_parquet, write_parquet


@pytest.fixture()
def simple_csv(tmp_path: pathlib.Path) -> str:
    """A minimal CSV with an integer column and a low-cardinality string column."""
    path = tmp_path / "data.csv"
    path.write_text(
        "chrom,pos,strand\n"
        "chr1,100,+\n"
        "chr1,200,-\n"
        "chr2,300,+\n"
        "chr2,400,-\n"
        "chr1,500,+\n",
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture()
def tsv_with_uniform_strings(tmp_path: pathlib.Path) -> str:
    """A TSV where one column has uniform-length 4-char strings."""
    path = tmp_path / "data.tsv"
    rows = [f"kmer_{i:04d}\t{i}" for i in range(100)]
    path.write_text("kmer\tcount\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# from_csv basics
# ---------------------------------------------------------------------------


def test_from_csv_returns_arrow_table(simple_csv: str) -> None:
    table = from_csv(simple_csv)
    assert isinstance(table, pa.Table)
    assert table.num_rows == 5
    assert set(table.schema.names) == {"chrom", "pos", "strand"}


def test_from_csv_integer_downcast(simple_csv: str) -> None:
    """Integer columns must be downcast to the smallest fitting type."""
    table = from_csv(simple_csv)
    # pos values fit in uint16 (100-500)
    assert pa.types.is_integer(table.schema.field("pos").type)
    assert table.schema.field("pos").type.bit_width <= 16


def test_from_csv_low_cardinality_string_encoded(simple_csv: str) -> None:
    """Low-cardinality string columns must be dictionary-encoded."""
    table = from_csv(simple_csv)
    assert pa.types.is_dictionary(table.schema.field("strand").type), (
        "strand (+/-) is low-cardinality and must be dictionary-encoded"
    )


def test_from_csv_tab_delimiter(tsv_with_uniform_strings: str) -> None:
    """Tab-delimited files must parse correctly with delimiter='\\t'."""
    table = from_csv(tsv_with_uniform_strings, delimiter="\t")
    assert isinstance(table, pa.Table)
    assert table.num_rows == 100
    assert "kmer" in table.schema.names
    assert "count" in table.schema.names


def test_from_csv_tab_uniform_length_string_promoted(
    tsv_with_uniform_strings: str,
) -> None:
    """Uniform-length string column in a TSV must be promoted to fixed_size_binary."""
    table = from_csv(tsv_with_uniform_strings, delimiter="\t")
    # kmer_XXXX strings are 9 chars, all same length, high-cardinality (100 unique)
    assert pa.types.is_fixed_size_binary(table.schema.field("kmer").type), (
        "uniform-length kmer column must be promoted to fixed_size_binary"
    )


# ---------------------------------------------------------------------------
# Full round-trip: CSV → from_csv → write_parquet → read_parquet
# ---------------------------------------------------------------------------


def test_csv_to_parquet_roundtrip_values(
    simple_csv: str, tmp_path: pathlib.Path
) -> None:
    """Values must survive the full CSV → Parquet → DataFrame round-trip."""
    table = from_csv(simple_csv)
    parquet_path = str(tmp_path / "out.parquet")
    write_parquet(table, parquet_path)

    df, _ = read_parquet(parquet_path, engine="pandas")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (5, 3)

    # chrom and strand must come back as plain strings, not bytes or Categorical
    assert pd.api.types.is_string_dtype(df["chrom"]) or pd.api.types.is_object_dtype(
        df["chrom"]
    ), "chrom must be plain string on read-back"
    assert pd.api.types.is_string_dtype(df["strand"]) or pd.api.types.is_object_dtype(
        df["strand"]
    ), "strand must be plain string on read-back"

    assert set(df["chrom"].unique()) == {"chr1", "chr2"}
    assert set(df["strand"].unique()) == {"+", "-"}


def test_csv_to_parquet_roundtrip_float_downcast(tmp_path: pathlib.Path) -> None:
    """float32 precision requested via from_csv must survive the full round-trip."""
    csv_path = tmp_path / "floats.csv"
    csv_path.write_text(
        "name,score\nalpha,1.5\nbeta,2.5\ngamma,3.5\n", encoding="utf-8"
    )

    # float_type="float32" applied at read time
    table = from_csv(str(csv_path), float_type="float32")
    assert table.schema.field("score").type == pa.float32()

    parquet_path = str(tmp_path / "floats.parquet")
    write_parquet(table, parquet_path)

    df, _ = read_parquet(parquet_path, engine="pandas")
    assert df["score"].tolist() == pytest.approx([1.5, 2.5, 3.5], abs=1e-5)
