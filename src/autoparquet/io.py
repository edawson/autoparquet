import json
from enum import Enum
from typing import Any

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
from pyarrow.parquet import write_table

from . import constants
from .converters import to_arrow_table
from .schema import infer_schema
from .utils.logger import get_logger

logger = get_logger(__name__)

# Metadata key used to record which columns were promoted from string to
# fixed_size_binary by infer_schema, so read_parquet can reverse the cast.
_PROMOTED_STRING_COLUMNS_KEY = "__autoparquet_string_columns__"

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

try:
    import cudf
except ImportError:
    cudf = None  # type: ignore


def _decode_dict_columns(table: pa.Table) -> pa.Table:
    """Cast dictionary(*, string/large_string) columns back to plain string.

    When Arrow dictionaries with string values are converted to pandas they
    become Categorical rather than plain object/str columns. This restores the
    expected type so round-trips are transparent to callers.

    Binary-valued dictionaries are intentionally left alone.
    """
    decoded = 0
    for i, field in enumerate(table.schema):
        if pa.types.is_dictionary(field.type) and (
            pa.types.is_string(field.type.value_type)
            or pa.types.is_large_string(field.type.value_type)
        ):
            table = table.set_column(
                i, field.name, table.column(i).cast(field.type.value_type)
            )
            decoded += 1
    if decoded:
        logger.debug("decoded %d dictionary string column(s) on read", decoded)
    return table


def _decode_promoted_string_columns(table: pa.Table) -> pa.Table:
    """Cast fixed_size_binary columns back to string using file metadata.

    infer_schema may promote uniform-length string columns to fixed_size_binary
    for storage efficiency. write_parquet records the original column names
    under _PROMOTED_STRING_COLUMNS_KEY so this function can reverse the cast.
    Columns the caller explicitly converted to binary are not listed there and
    are left untouched.
    """
    metadata = table.schema.metadata or {}
    raw = metadata.get(_PROMOTED_STRING_COLUMNS_KEY.encode())
    if not raw:
        return table

    promoted: list[str] = json.loads(raw.decode())
    if not promoted:
        return table

    decoded = 0
    for i, field in enumerate(table.schema):
        if field.name in promoted and pa.types.is_fixed_size_binary(field.type):
            table = table.set_column(i, field.name, table.column(i).cast(pa.string()))
            decoded += 1
    if decoded:
        logger.debug(
            "decoded %d promoted string column(s) from %s metadata",
            decoded,
            _PROMOTED_STRING_COLUMNS_KEY,
        )
    return table


class CompressionType(str, Enum):
    """Supported Parquet compression types."""

    NONE = "NONE"
    SNAPPY = "SNAPPY"
    GZIP = "GZIP"
    LZO = "LZO"
    BROTLI = "BROTLI"
    LZ4 = "LZ4"
    ZSTD = "ZSTD"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value.upper() in cls._member_names_


class EngineType(str, Enum):
    """Supported DataFrame engines for reading."""

    PANDAS = "pandas"
    POLARS = "polars"
    CUDF = "cudf"
    AUTO = "auto"


def write_parquet(
    data: Any,
    path: str,
    header: dict[str, str] | None = None,
    compression: str | CompressionType = constants.DEFAULT_COMPRESSION,
    compression_level: int | None = constants.DEFAULT_COMPRESSION_LEVEL,
    use_parquet_dictionary_compression: bool = True,
    data_page_size: int = constants.DEFAULT_PAGE_SIZE,
    float_type: str = "float64",
    column_encoding: dict[str, str] | None = None,
    **kwargs: Any,
) -> None:
    """
    Writes data to a Parquet file with an optimized schema and custom header.

    Args:
        data: A pandas DataFrame, polars DataFrame, cuDF DataFrame, or Arrow Table.
        path: Output file path.
        header: Optional key-value metadata to store in the Parquet file.
        compression: Compression algorithm (ZSTD, SNAPPY, GZIP, etc.).
        compression_level: Compression level (algorithm-dependent).
        use_parquet_dictionary_compression: Let Parquet apply its own dictionary
            encoding pass on top of the schema-level optimizations.
        data_page_size: Parquet data page size in bytes.
        float_type: Target float precision ("float16", "float32", or "float64").
        column_encoding: Optional mapping of column names to Parquet encodings.
            Supported encodings: "PLAIN", "RLE", "BIT_PACKED",
            "PLAIN_DICTIONARY", "RLE_DICTIONARY", "DELTA_BINARY_PACKED",
            "DELTA_LENGTH_BYTE_ARRAY", "DELTA_BYTE_ARRAY", "BYTE_STREAM_SPLIT".
            Example: {"timestamp": "DELTA_BINARY_PACKED",
            "position": "DELTA_BINARY_PACKED"}
        **kwargs: Additional arguments forwarded to pyarrow.parquet.write_table.
    """
    comp_str = (
        compression.value
        if isinstance(compression, CompressionType)
        else str(compression).upper()
    )
    if not CompressionType.is_valid(comp_str):
        valid_types = ", ".join(CompressionType._member_names_)
        raise ValueError(
            f"Invalid compression type: {compression}. Valid types are: {valid_types}"
        )

    # Codecs that don't support a compression level parameter
    if comp_str in ("SNAPPY", "LZ4") and compression_level is not None:
        logger.warning(
            "compression_level ignored for %s (not supported by this codec)", comp_str
        )
        compression_level = None

    table = to_arrow_table(data)
    original_schema = table.schema
    optimized_schema = infer_schema(table, float_type=float_type)
    table = table.cast(optimized_schema, safe=False)

    # Detect columns that infer_schema promoted from string to fixed_size_binary.
    # Store their names in file metadata so read_parquet can reverse the cast.
    promoted_string_cols = [
        original_schema.field(i).name
        for i in range(len(original_schema))
        if (
            pa.types.is_string(original_schema.field(i).type)
            or pa.types.is_large_string(original_schema.field(i).type)
        )
        and pa.types.is_fixed_size_binary(optimized_schema.field(i).type)
    ]

    existing_metadata: dict[bytes, bytes] = dict(optimized_schema.metadata or {})
    if promoted_string_cols:
        existing_metadata[_PROMOTED_STRING_COLUMNS_KEY.encode()] = json.dumps(
            promoted_string_cols
        ).encode()

    if header:
        for k, v in header.items():
            existing_metadata[k.encode("utf-8")] = str(v).encode("utf-8")

    if existing_metadata:
        table = table.replace_schema_metadata(existing_metadata)

    write_kwargs: dict[str, Any] = {
        "compression": comp_str,
        "compression_level": compression_level,
        "data_page_size": data_page_size,
        "row_group_size": constants.DEFAULT_ROW_GROUP_SIZE,
        "version": constants.DEFAULT_PARQUET_VERSION,
    }

    if column_encoding:
        # PyArrow requires use_dictionary=False when column_encoding is specified.
        write_kwargs["use_dictionary"] = False
        write_kwargs["column_encoding"] = column_encoding
    else:
        write_kwargs["use_dictionary"] = use_parquet_dictionary_compression

    write_kwargs.update(kwargs)
    write_table(table, path, **write_kwargs)
    logger.info("wrote %d rows → %s (%s)", table.num_rows, path, comp_str)


def read_parquet(
    path: str, engine: str | EngineType = EngineType.AUTO
) -> tuple[Any, dict[str, str]]:
    """
    Reads a Parquet file and returns (data, header_metadata).

    Args:
        path: Path to the Parquet file.
        engine: DataFrame engine to use ('pandas', 'polars', 'cudf', or 'auto').
                'auto' tries polars, then cudf, then pandas.
    """
    table = pq.read_table(path)

    # Decode types that were optimized at write time back to their original forms
    # so callers get strings as strings, not as Categorical or bytes.
    table = _decode_dict_columns(table)
    table = _decode_promoted_string_columns(table)

    metadata = table.schema.metadata or {}
    header = {
        k.decode("utf-8"): v.decode("utf-8")
        for k, v in metadata.items()
        if k != _PROMOTED_STRING_COLUMNS_KEY.encode()
    }

    if isinstance(engine, str):
        engine = EngineType(engine.lower())
    auto = engine == EngineType.AUTO

    logger.debug("reading %s (engine: %s)", path, engine.value)

    if engine == EngineType.POLARS or auto:
        if pl is not None:
            logger.info("read %d rows ← %s (engine: polars)", table.num_rows, path)
            return pl.from_arrow(table), header
        if not auto:
            raise ImportError("polars is not installed but was explicitly requested.")

    if engine == EngineType.CUDF or auto:
        if cudf is not None:
            logger.info("read %d rows ← %s (engine: cudf)", table.num_rows, path)
            return cudf.from_arrow(table), header
        if not auto:
            raise ImportError("cudf is not installed but was explicitly requested.")

    if engine == EngineType.PANDAS or auto:
        if pd is not None:
            logger.info("read %d rows ← %s (engine: pandas)", table.num_rows, path)
            return table.to_pandas(), header
        if not auto:
            raise ImportError("pandas is not installed but was explicitly requested.")

    raise ImportError(
        f"No supported DataFrame library for engine '{engine.value}' is installed."
    )


def from_csv(
    path: str,
    delimiter: str = ",",
    quote_char: str = '"',
    escape_char: str | None = None,
    float_type: str = "float64",
    **kwargs: Any,
) -> pa.Table:
    """
    Reads a CSV/TSV file directly into an optimized Arrow Table.

    Args:
        path: Path to the CSV/TSV file.
        delimiter: Field delimiter (e.g., ',' or '\\t').
        quote_char: Quoting character.
        escape_char: Escape character.
        float_type: Target float precision ("float16", "float32", or "float64").
        **kwargs: Additional arguments passed to pyarrow.csv.read_csv.
    """
    parse_options = pv.ParseOptions(
        delimiter=delimiter, quote_char=quote_char, escape_char=escape_char
    )
    table = pv.read_csv(path, parse_options=parse_options, **kwargs)
    optimized_schema = infer_schema(table, float_type=float_type)
    table = table.cast(optimized_schema, safe=False)
    logger.info(
        "read CSV %s → %d rows, %d columns", path, table.num_rows, table.num_columns
    )
    return table


def from_excel(
    path: str,
    sheet_name: str | int = 0,
    float_type: str = "float64",
    **kwargs: Any,
) -> pa.Table:
    """
    Reads an Excel file (XLSX/XLS/XLSB) into an optimized Arrow Table.

    Uses python-calamine for fast, Rust-based Excel reading
    (no runtime Rust dependency).

    Args:
        path: Path to the Excel file (.xlsx, .xls, .xlsb, or ODF format).
        sheet_name: Sheet name or index to read (default: 0, first sheet).
        float_type: Target float precision
            ("float16", "float32", or "float64").
        **kwargs: Additional arguments (reserved for future use).

    Returns:
        Optimized Arrow Table.
    """
    try:
        from python_calamine import CalamineWorkbook
    except ImportError as err:
        raise ImportError(
            "python-calamine is not installed but is required to read "
            "Excel files. Please install python-calamine to use this "
            "functionality: pip install python-calamine"
        ) from err

    workbook = CalamineWorkbook.from_path(path)

    # Get sheet by name or index
    if isinstance(sheet_name, int):
        sheet_names = workbook.sheet_names
        if sheet_name >= len(sheet_names):
            raise ValueError(
                f"Sheet index {sheet_name} out of range. "
                f"Workbook has {len(sheet_names)} sheets."
            )
        sheet = workbook.get_sheet_by_index(sheet_name)
    else:
        sheet = workbook.get_sheet_by_name(sheet_name)

    # Convert to pandas DataFrame, then to Arrow Table
    if pd is None:
        raise ImportError(
            "pandas is not installed but is required for Excel conversion. "
            "Please install pandas to use this functionality."
        )

    df = sheet.to_python(empty_value="", header=1)
    if isinstance(df, list):
        df = pd.DataFrame(df)

    table = pa.Table.from_pandas(df)
    optimized_schema = infer_schema(table, float_type=float_type)
    table = table.cast(optimized_schema, safe=False)
    sheet_label = sheet_name if isinstance(sheet_name, str) else f"index {sheet_name}"
    logger.info(
        "read Excel %s (sheet '%s') → %d rows, %d columns",
        path,
        sheet_label,
        table.num_rows,
        table.num_columns,
    )
    return table


def to_excel(
    data: Any,
    path: str,
    sheet_name: str = "Sheet1",
    **kwargs: Any,
) -> None:
    """
    Writes data to an Excel file (XLSX).

    Args:
        data: A pandas DataFrame, polars DataFrame, cuDF DataFrame, or Arrow Table.
        path: Output file path (.xlsx).
        sheet_name: Name of the sheet to create (default: "Sheet1").
        **kwargs: Additional arguments passed to pd.DataFrame.to_excel.
    """
    if pd is None:
        raise ImportError(
            "pandas is not installed but is required to write Excel files. "
            "Please install pandas to use this functionality."
        )

    # Convert to pandas DataFrame if needed
    if isinstance(data, pa.Table):
        df = data.to_pandas()
    elif isinstance(data, pd.DataFrame):
        df = data
    elif hasattr(data, "to_pandas"):
        df = data.to_pandas()
    else:
        raise ValueError(
            f"Unsupported data type for Excel export: {type(data)}. "
            "Expected pandas DataFrame, Arrow Table, or compatible format."
        )

    df.to_excel(path, sheet_name=sheet_name, **kwargs)
