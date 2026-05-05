import json
import os
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
except ImportError:  # pragma: no cover - optional dep
    pd = None  # type: ignore

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dep
    pl = None  # type: ignore

try:
    import cudf
except ImportError:  # pragma: no cover - optional dep
    cudf = None  # type: ignore


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------


def _validate_float_type(float_type: str) -> None:
    if float_type not in constants.VALID_FLOAT_TYPES:
        raise ValueError(
            f"float_type must be one of {constants.VALID_FLOAT_TYPES}, "
            f"got {float_type!r}"
        )


def _validate_compression_level(compression: str, level: int | None) -> None:
    """Range-check compression_level for codecs with documented bounds."""
    if level is None:
        return
    if not isinstance(level, int) or isinstance(level, bool):
        raise TypeError(
            f"compression_level must be an int or None, got {type(level).__name__}"
        )
    bounds = constants.COMPRESSION_LEVEL_RANGES.get(compression)
    if bounds and not bounds[0] <= level <= bounds[1]:
        lo, hi = bounds
        raise ValueError(
            f"compression_level={level} is out of range for {compression}: "
            f"expected {lo}..{hi}"
        )


def _validate_column_encoding(
    encoding: dict[str, str] | None, column_names: list[str]
) -> None:
    """Validate that each (column, encoding) pair is usable."""
    if not encoding:
        return
    if not isinstance(encoding, dict):
        raise TypeError(
            f"column_encoding must be a dict[str, str], got {type(encoding).__name__}"
        )

    table_columns = set(column_names)
    for col, enc in encoding.items():
        if col not in table_columns:
            raise ValueError(
                f"column_encoding references unknown column {col!r}; "
                f"table columns are {sorted(table_columns)}"
            )
        if not isinstance(enc, str) or enc.upper() not in (
            constants.VALID_COLUMN_ENCODINGS
        ):
            raise ValueError(
                f"Unsupported encoding {enc!r} for column {col!r}; "
                f"valid encodings are {constants.VALID_COLUMN_ENCODINGS}"
            )


def _ensure_writable_parent(path: str) -> None:
    """Verify the parent directory of `path` exists (or path is in cwd)."""
    parent = os.path.dirname(os.fspath(path))
    if parent and not os.path.isdir(parent):
        raise FileNotFoundError(
            f"Cannot write to {path!r}: parent directory {parent!r} does not exist"
        )


def _ensure_readable_file(path: str) -> None:
    """Verify the path exists and is a regular file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path!r}")
    if not os.path.isfile(path):
        raise IsADirectoryError(f"{path!r} is not a regular file")


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
    """Compression codecs supported by the PyArrow Parquet writer.

    Note: ``LZO`` is part of the Parquet spec but is not implemented by
    the C++ Arrow library, so it is intentionally excluded here. Attempting
    to use it would otherwise fail deep inside PyArrow with an unhelpful
    ``OSError``.
    """

    NONE = "NONE"
    SNAPPY = "SNAPPY"
    GZIP = "GZIP"
    BROTLI = "BROTLI"
    LZ4 = "LZ4"
    ZSTD = "ZSTD"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value.upper() in cls._member_names_


class EngineType(str, Enum):
    """DataFrame engines accepted by :func:`read_parquet`."""

    PANDAS = "pandas"
    POLARS = "polars"
    CUDF = "cudf"
    AUTO = "auto"

    @classmethod
    def from_value(cls, value: "str | EngineType") -> "EngineType":
        """Coerce a string or enum member to an EngineType with a clear error."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as err:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(
                f"Unknown engine {value!r}; expected one of: {valid}"
            ) from err


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
    """Write a DataFrame to a Parquet file with an optimized schema.

    Args:
        data: A pandas DataFrame, polars DataFrame, cuDF DataFrame, or Arrow Table.
        path: Output file path. The parent directory must already exist.
        header: Optional key-value metadata to store in the Parquet file. Both
            keys and values are encoded as UTF-8 byte strings.
        compression: Compression codec name or :class:`CompressionType` member.
            Supported codecs: NONE, SNAPPY, GZIP, BROTLI, LZ4, ZSTD.
        compression_level: Codec-specific compression level. Range-checked for
            ZSTD (1-22), GZIP (1-9), and BROTLI (0-11). Ignored for SNAPPY/LZ4.
        use_parquet_dictionary_compression: Apply Parquet's own dictionary
            encoding pass on top of the schema-level optimizations.
        data_page_size: Parquet data page size in bytes (must be positive).
        float_type: Target float precision: "float16", "float32", or "float64".
        column_encoding: Optional mapping of column names to Parquet encodings.
            Supported: PLAIN, BYTE_STREAM_SPLIT, DELTA_BINARY_PACKED,
            DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY. Setting this disables
            global dictionary encoding; use ``use_dictionary=[col, ...]`` (a
            list) in **kwargs to keep dict on selected columns.
        **kwargs: Additional arguments forwarded to ``pyarrow.parquet.write_table``.

    Raises:
        FileNotFoundError: if the parent directory of ``path`` does not exist.
        TypeError: for ill-typed parameters.
        ValueError: for unsupported compression codecs, invalid encodings,
            out-of-range compression levels, or unsupported ``float_type``.
    """
    # ----- Up-front parameter validation ----------------------------------
    _validate_float_type(float_type)
    if not isinstance(data_page_size, int) or data_page_size <= 0:
        raise ValueError(
            f"data_page_size must be a positive int, got {data_page_size!r}"
        )

    comp_str = (
        compression.value
        if isinstance(compression, CompressionType)
        else str(compression).upper()
    )
    if not CompressionType.is_valid(comp_str):
        valid_types = ", ".join(CompressionType._member_names_)
        raise ValueError(
            f"Invalid compression type: {compression!r}. Valid types are: {valid_types}"
        )

    # Codecs that don't accept a level: silently drop it with a debug note.
    if comp_str in ("SNAPPY", "LZ4", "NONE") and compression_level is not None:
        logger.debug(
            "compression_level ignored for %s (not supported by this codec)", comp_str
        )
        compression_level = None
    else:
        _validate_compression_level(comp_str, compression_level)

    _ensure_writable_parent(path)

    table = to_arrow_table(data)
    _validate_column_encoding(column_encoding, table.column_names)

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
    """Read a Parquet file written by :func:`write_parquet`.

    Optimizations applied at write time (low-cardinality strings stored as
    Arrow Dictionary, uniform-length strings stored as FixedSizeBinary) are
    transparently reversed so callers see plain string columns.

    Args:
        path: Path to an existing Parquet file.
        engine: Target DataFrame library: "pandas", "polars", "cudf", or
            "auto" (the default). "auto" tries polars, then cudf, then pandas
            and uses the first that is installed.

    Returns:
        A 2-tuple ``(dataframe, header_metadata)``. The header is the
        user-supplied ``header`` dict from :func:`write_parquet` (UTF-8
        decoded), with internal sentinel keys filtered out.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if ``engine`` is not a recognized engine name.
        ImportError: if a specific engine is requested but not installed, or
            if ``engine='auto'`` and no supported library is installed.
    """
    _ensure_readable_file(path)
    engine_enum = EngineType.from_value(engine)

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

    auto = engine_enum == EngineType.AUTO
    logger.debug("reading %s (engine: %s)", path, engine_enum.value)

    if engine_enum == EngineType.POLARS or auto:
        if pl is not None:
            logger.info("read %d rows ← %s (engine: polars)", table.num_rows, path)
            return pl.from_arrow(table), header
        if not auto:
            raise ImportError(
                "polars is not installed but was explicitly requested. "
                "Install with: pip install polars"
            )

    if engine_enum == EngineType.CUDF or auto:
        if cudf is not None:
            logger.info("read %d rows ← %s (engine: cudf)", table.num_rows, path)
            return cudf.from_arrow(table), header
        if not auto:
            raise ImportError(
                "cudf is not installed but was explicitly requested. "
                "Install RAPIDS cuDF for your CUDA version: https://rapids.ai/"
            )

    if engine_enum == EngineType.PANDAS or auto:
        if pd is not None:
            logger.info("read %d rows ← %s (engine: pandas)", table.num_rows, path)
            return table.to_pandas(), header
        if not auto:
            raise ImportError(
                "pandas is not installed but was explicitly requested. "
                "Install with: pip install pandas"
            )

    raise ImportError(
        "No supported DataFrame library is installed. Install one of: "
        "pandas, polars, or cudf."
    )


def from_csv(
    path: str,
    delimiter: str = ",",
    quote_char: str = '"',
    escape_char: str | None = None,
    float_type: str = "float64",
    **kwargs: Any,
) -> pa.Table:
    """Read a CSV/TSV file into an optimized Arrow Table.

    Args:
        path: Path to an existing CSV/TSV file.
        delimiter: Field delimiter (e.g. ``','`` or ``'\\t'``).
        quote_char: Quoting character.
        escape_char: Escape character.
        float_type: Target float precision: "float16", "float32", or "float64".
        **kwargs: Additional arguments passed to ``pyarrow.csv.read_csv``.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if ``float_type`` is not supported.
    """
    _ensure_readable_file(path)
    _validate_float_type(float_type)

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
    """Read an Excel sheet (XLSX/XLS/XLSB) into an optimized Arrow Table.

    Uses python-calamine for fast, Rust-based Excel reading.

    Args:
        path: Path to an existing Excel file.
        sheet_name: Sheet name or 0-based index (default: 0, first sheet).
        float_type: Target float precision: "float16", "float32", or "float64".
        **kwargs: Reserved for future use.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if ``float_type`` is unsupported, or the requested sheet
            does not exist.
        ImportError: if python-calamine or pandas is not installed.
    """
    _ensure_readable_file(path)
    _validate_float_type(float_type)

    try:
        from python_calamine import CalamineWorkbook
    except ImportError as err:
        raise ImportError(
            "python-calamine is required to read Excel files. "
            "Install with: pip install python-calamine"
        ) from err

    workbook = CalamineWorkbook.from_path(path)

    if isinstance(sheet_name, int):
        sheet_names = workbook.sheet_names
        if sheet_name < 0 or sheet_name >= len(sheet_names):
            raise ValueError(
                f"Sheet index {sheet_name} out of range; workbook has "
                f"{len(sheet_names)} sheets ({sheet_names})"
            )
        sheet = workbook.get_sheet_by_index(sheet_name)
    elif isinstance(sheet_name, str):
        if sheet_name not in workbook.sheet_names:
            raise ValueError(
                f"Sheet {sheet_name!r} not found; available sheets: "
                f"{workbook.sheet_names}"
            )
        sheet = workbook.get_sheet_by_name(sheet_name)
    else:
        raise TypeError(
            f"sheet_name must be a str or int, got {type(sheet_name).__name__}"
        )

    if pd is None:
        raise ImportError(
            "pandas is required to convert Excel sheets to Arrow Tables. "
            "Install with: pip install pandas"
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
    """Write a DataFrame to an Excel file (XLSX).

    Args:
        data: A pandas DataFrame, polars DataFrame, cuDF DataFrame, or Arrow Table.
        path: Output file path (.xlsx). Parent directory must exist.
        sheet_name: Name of the sheet to create (default: "Sheet1").
        **kwargs: Additional arguments passed to ``pandas.DataFrame.to_excel``.

    Raises:
        FileNotFoundError: if the parent directory of ``path`` does not exist.
        ImportError: if pandas is not installed.
        ValueError: if ``data`` cannot be converted to a pandas DataFrame.
    """
    if pd is None:
        raise ImportError(
            "pandas is required to write Excel files. Install with: pip install pandas"
        )

    _ensure_writable_parent(path)

    if isinstance(data, pa.Table):
        df = data.to_pandas()
    elif isinstance(data, pd.DataFrame):
        df = data
    elif hasattr(data, "to_pandas"):
        df = data.to_pandas()
    else:
        raise ValueError(
            f"Unsupported data type for Excel export: {type(data).__name__}. "
            "Expected pandas DataFrame, Arrow Table, or any object with a "
            "to_pandas() method."
        )

    df.to_excel(path, sheet_name=sheet_name, **kwargs)
