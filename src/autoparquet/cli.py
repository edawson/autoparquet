import argparse
import json
import logging
import sys

import pyarrow.feather as feather

from autoparquet.io import from_csv, from_excel, read_parquet, to_excel, write_parquet
from autoparquet.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_sheet(sheet: str) -> str | int:
    """Return sheet as an integer index if possible, otherwise as a name."""
    try:
        return int(sheet)
    except ValueError:
        return sheet


def _parse_column_encoding(encoding_str: str | None) -> dict[str, str] | None:
    """
    Parse column encoding from CLI string format.

    Supports JSON or simple comma-separated format:
    - JSON: {"timestamp": "DELTA_BINARY_PACKED", "position": "DELTA_BINARY_PACKED"}
    - Simple: "timestamp:DELTA_BINARY_PACKED,position:DELTA_BINARY_PACKED"
    """
    if not encoding_str:
        return None

    try:
        # Try JSON format first
        parsed = json.loads(encoding_str)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try simple key:value format
    try:
        result: dict[str, str] = {}
        for pair in encoding_str.split(","):
            key, value = pair.strip().split(":")
            result[key.strip()] = value.strip()
        return result if result else None
    except (ValueError, AttributeError) as err:
        raise ValueError(
            f"Invalid column encoding format: {encoding_str}. "
            "Use JSON format or 'col1:ENCODING1,col2:ENCODING2'"
        ) from err


def csv_to_parquet(
    input_path: str,
    output_path: str | None,
    compression: str,
    compression_level: int | None,
    delimiter: str,
    float_type: str = "float32",
    column_encoding: str | None = None,
) -> None:
    """Converts a CSV file to an optimized Parquet file."""
    if not output_path:
        output_path = input_path.rsplit(".", 1)[0] + ".parquet"

    logger.info("reading %s ...", input_path)
    table = from_csv(path=input_path, delimiter=delimiter, float_type=float_type)

    logger.info("writing optimized Parquet → %s (%s)", output_path, compression)
    encodings = _parse_column_encoding(column_encoding)
    write_parquet(
        table,
        output_path,
        compression=compression,
        compression_level=compression_level,
        float_type=float_type,
        column_encoding=encodings,
    )
    logger.info("done")


def csv_to_feather(
    input_path: str,
    output_path: str | None,
    compression: str,
    compression_level: int | None,
    delimiter: str,
    float_type: str = "float32",
) -> None:
    """Converts a CSV file to an optimized Feather file."""
    if not output_path:
        output_path = input_path.rsplit(".", 1)[0] + ".feather"

    logger.info("reading %s ...", input_path)
    table = from_csv(path=input_path, delimiter=delimiter, float_type=float_type)

    logger.info("writing optimized Feather → %s (%s)", output_path, compression)
    feather.write_feather(
        table,
        output_path,
        compression=compression,
        compression_level=compression_level,
    )
    logger.info("done")


def excel_to_parquet(
    input_path: str,
    output_path: str | None,
    compression: str,
    compression_level: int | None,
    sheet: str = "0",
    float_type: str = "float32",
    column_encoding: str | None = None,
) -> None:
    """Converts an Excel file (XLSX/XLS) to an optimized Parquet file."""
    if not output_path:
        output_path = input_path.rsplit(".", 1)[0] + ".parquet"

    logger.info("reading %s (sheet: %s) ...", input_path, sheet)
    table = from_excel(
        path=input_path, sheet_name=_parse_sheet(sheet), float_type=float_type
    )

    logger.info("writing optimized Parquet → %s (%s)", output_path, compression)
    encodings = _parse_column_encoding(column_encoding)
    write_parquet(
        table,
        output_path,
        compression=compression,
        compression_level=compression_level,
        float_type=float_type,
        column_encoding=encodings,
    )
    logger.info("done")


def parquet_to_excel(
    input_path: str,
    output_path: str | None,
    sheet: str = "Sheet1",
) -> None:
    """Converts a Parquet file to an Excel file (XLSX)."""
    if not output_path:
        output_path = input_path.rsplit(".", 1)[0] + ".xlsx"

    logger.info("reading %s ...", input_path)
    df, _ = read_parquet(input_path, engine="pandas")

    logger.info("writing Excel → %s (sheet: %s)", output_path, sheet)
    to_excel(df, output_path, sheet_name=sheet)
    logger.info("done")


def excel_to_feather(
    input_path: str,
    output_path: str | None,
    compression: str,
    compression_level: int | None,
    sheet: str = "0",
    float_type: str = "float32",
) -> None:
    """Converts an Excel file (XLSX/XLS) to an optimized Feather file."""
    if not output_path:
        output_path = input_path.rsplit(".", 1)[0] + ".feather"

    logger.info("reading %s (sheet: %s) ...", input_path, sheet)
    table = from_excel(
        path=input_path, sheet_name=_parse_sheet(sheet), float_type=float_type
    )

    logger.info("writing optimized Feather → %s (%s)", output_path, compression)
    feather.write_feather(
        table,
        output_path,
        compression=compression,
        compression_level=compression_level,
    )
    logger.info("done")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="AutoParquet CLI: Optimize your data storage with Arrow and "
        "Parquet."
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Common arguments for CSV/Excel input
    input_parser = argparse.ArgumentParser(add_help=False)
    input_parser.add_argument("input", help="Path to the input file")
    input_parser.add_argument("-o", "--output", help="Path to the output file")
    input_parser.add_argument(
        "-c",
        "--compression",
        default="zstd",
        choices=["snappy", "gzip", "zstd", "lz4", "uncompressed"],
        help="Compression algorithm (default: zstd)",
    )
    input_parser.add_argument(
        "-l",
        "--level",
        type=int,
        help="Compression level (default depends on algorithm)",
    )
    input_parser.add_argument(
        "-f",
        "--float-type",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Float precision to use (default: float32)",
    )
    input_parser.add_argument(
        "--column-encoding",
        help="Column encoding map for Parquet. Supports JSON or simple format: "
        "'col1:DELTA_BINARY_PACKED,col2:DELTA_BYTE_ARRAY'",
    )

    # CSV-specific arguments
    csv_parser = argparse.ArgumentParser(add_help=False)
    csv_parser.add_argument(
        "-d", "--delimiter", default=",", help="CSV delimiter (default: ,)"
    )

    # Excel-specific arguments
    excel_parser = argparse.ArgumentParser(add_help=False)
    excel_parser.add_argument(
        "-s",
        "--sheet",
        default="0",
        help="Sheet name or index to read (default: 0, the first sheet)",
    )

    # Parquet to Excel arguments (simpler, no compression)
    parquet_to_excel_parser = argparse.ArgumentParser(add_help=False)
    parquet_to_excel_parser.add_argument("input", help="Path to the input Parquet file")
    parquet_to_excel_parser.add_argument(
        "-o", "--output", help="Path to the output Excel file"
    )
    parquet_to_excel_parser.add_argument(
        "-s",
        "--sheet",
        default="Sheet1",
        help="Sheet name for the Excel file (default: Sheet1)",
    )

    # csv_to_parquet subcommand
    subparsers.add_parser(
        "csv_to_parquet",
        parents=[input_parser, csv_parser],
        help="Convert CSV to optimized Parquet",
    )

    # csv_to_feather subcommand
    subparsers.add_parser(
        "csv_to_feather",
        parents=[input_parser, csv_parser],
        help="Convert CSV to optimized Feather",
    )

    # excel_to_parquet subcommand
    subparsers.add_parser(
        "excel_to_parquet",
        parents=[input_parser, excel_parser],
        help="Convert Excel (XLSX/XLS) to optimized Parquet",
    )

    # excel_to_feather subcommand
    subparsers.add_parser(
        "excel_to_feather",
        parents=[input_parser, excel_parser],
        help="Convert Excel (XLSX/XLS) to optimized Feather",
    )

    # parquet_to_excel subcommand
    subparsers.add_parser(
        "parquet_to_excel",
        parents=[parquet_to_excel_parser],
        help="Convert Parquet to Excel (XLSX)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "csv_to_parquet":
            csv_to_parquet(
                args.input,
                args.output,
                args.compression,
                args.level,
                args.delimiter,
                args.float_type,
                args.column_encoding,
            )
        elif args.command == "csv_to_feather":
            csv_to_feather(
                args.input,
                args.output,
                args.compression,
                args.level,
                args.delimiter,
                args.float_type,
            )
        elif args.command == "excel_to_parquet":
            excel_to_parquet(
                args.input,
                args.output,
                args.compression,
                args.level,
                args.sheet,
                args.float_type,
                args.column_encoding,
            )
        elif args.command == "excel_to_feather":
            excel_to_feather(
                args.input,
                args.output,
                args.compression,
                args.level,
                args.sheet,
                args.float_type,
            )
        elif args.command == "parquet_to_excel":
            parquet_to_excel(args.input, args.output, args.sheet)
    except Exception as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
