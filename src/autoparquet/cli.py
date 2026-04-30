import argparse
import sys
from typing import Optional

import pyarrow.feather as feather

from autoparquet.io import from_csv, write_parquet


def csv_to_parquet(
    input_path: str,
    output_path: Optional[str],
    compression: str,
    compression_level: Optional[int],
    delimiter: str,
    float_type: str = "float32",
) -> None:
    """Converts a CSV file to an optimized Parquet file."""
    if not output_path:
        output_path = input_path.rsplit(".", 1)[0] + ".parquet"

    print(f"Reading {input_path}...")
    table = from_csv(path=input_path, delimiter=delimiter, float_type=float_type)

    print(f"Writing optimized Parquet to {output_path} (compression: {compression})...")
    write_parquet(
        table,
        output_path,
        compression=compression,
        compression_level=compression_level,
        float_type=float_type,
    )
    print("Done!")


def csv_to_feather(
    input_path: str,
    output_path: Optional[str],
    compression: str,
    compression_level: Optional[int],
    delimiter: str,
    float_type: str = "float32",
) -> None:
    """Converts a CSV file to an optimized Feather file."""
    if not output_path:
        output_path = input_path.rsplit(".", 1)[0] + ".feather"

    print(f"Reading {input_path}...")
    table = from_csv(path=input_path, delimiter=delimiter, float_type=float_type)

    print(f"Writing optimized Feather to {output_path} (compression: {compression})...")
    feather.write_feather(
        table,
        output_path,
        compression=compression,
        compression_level=compression_level,
    )
    print("Done!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoParquet CLI: Optimize your data storage with Arrow and Parquet."
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("input", help="Path to the input CSV file")
    common_parser.add_argument("-o", "--output", help="Path to the output file")
    common_parser.add_argument(
        "-c",
        "--compression",
        default="zstd",
        choices=["snappy", "gzip", "zstd", "lz4", "uncompressed"],
        help="Compression algorithm (default: zstd)",
    )
    common_parser.add_argument(
        "-l",
        "--level",
        type=int,
        help="Compression level (default depends on algorithm)",
    )
    common_parser.add_argument(
        "-d", "--delimiter", default=",", help="CSV delimiter (default: ,)"
    )
    common_parser.add_argument(
        "-f",
        "--float-type",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Float precision to use (default: float32)",
    )

    # csv_to_parquet subcommand
    subparsers.add_parser(
        "csv_to_parquet",
        parents=[common_parser],
        help="Convert CSV to optimized Parquet",
    )

    # csv_to_feather subcommand
    subparsers.add_parser(
        "csv_to_feather",
        parents=[common_parser],
        help="Convert CSV to optimized Feather",
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
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
