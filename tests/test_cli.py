"""Tests for autoparquet.cli: encoding parser, default paths, and exit codes."""

import pathlib
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from autoparquet.cli import (
    _parse_column_encoding,
    csv_to_parquet,
)

# ---------------------------------------------------------------------------
# _parse_column_encoding
# ---------------------------------------------------------------------------


def test_parse_column_encoding_none() -> None:
    assert _parse_column_encoding(None) is None


def test_parse_column_encoding_empty_string() -> None:
    assert _parse_column_encoding("") is None


def test_parse_column_encoding_json() -> None:
    result = _parse_column_encoding(
        '{"ts": "DELTA_BINARY_PACKED", "pos": "DELTA_BINARY_PACKED"}'
    )
    assert result == {"ts": "DELTA_BINARY_PACKED", "pos": "DELTA_BINARY_PACKED"}


def test_parse_column_encoding_simple() -> None:
    result = _parse_column_encoding("ts:DELTA_BINARY_PACKED,pos:DELTA_BYTE_ARRAY")
    assert result == {"ts": "DELTA_BINARY_PACKED", "pos": "DELTA_BYTE_ARRAY"}


def test_parse_column_encoding_simple_single() -> None:
    result = _parse_column_encoding("col:PLAIN")
    assert result == {"col": "PLAIN"}


def test_parse_column_encoding_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid column encoding format"):
        _parse_column_encoding("no-colon-here")


# ---------------------------------------------------------------------------
# Default output path derivation
# ---------------------------------------------------------------------------


def test_csv_to_parquet_default_output_path(tmp_path: pathlib.Path) -> None:
    """When no -o is given, the output path is derived by replacing the extension."""
    input_path = str(tmp_path / "data.csv")
    expected_output = str(tmp_path / "data.parquet")

    # Write a minimal CSV so from_csv doesn't fail
    pathlib.Path(input_path).write_text("a,b\n1,x\n2,y\n")

    with patch("autoparquet.cli.write_parquet") as mock_write:
        with patch("autoparquet.cli.from_csv") as mock_read:
            mock_read.return_value = MagicMock()
            csv_to_parquet(
                input_path=input_path,
                output_path=None,
                compression="zstd",
                compression_level=3,
                delimiter=",",
            )
        # The second positional argument to write_parquet is the output path
        _, call_args, _ = mock_write.mock_calls[0]
        assert call_args[1] == expected_output


# ---------------------------------------------------------------------------
# CLI entry point: no subcommand exits with code 1
# ---------------------------------------------------------------------------


def test_cli_no_subcommand_exits_nonzero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "autoparquet.cli"],
        capture_output=True,
    )
    assert result.returncode == 1
