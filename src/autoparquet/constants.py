"""Constants and default values for autoparquet."""

# Default Parquet Write Settings
DEFAULT_COMPRESSION = "ZSTD"
DEFAULT_COMPRESSION_LEVEL = 3
DEFAULT_PAGE_SIZE = 1024 * 1024  # 1MB
DEFAULT_ROW_GROUP_SIZE = 1000000  # 1M rows per group
DEFAULT_PARQUET_VERSION = "2.6"

# Schema Inference Heuristics
DICTIONARY_ENCODING_UNIQUE_LIMIT = 10000
DICTIONARY_ENCODING_RATIO_LIMIT = 0.5
DICTIONARY_ENCODING_MIN_ROWS = 100

# Integer Downcasting Limits
UINT8_MAX = 255
UINT16_MAX = 65535
UINT32_MAX = 4294967295

INT8_MIN, INT8_MAX = -128, 127
INT16_MIN, INT16_MAX = -32768, 32767
INT32_MIN, INT32_MAX = -2147483648, 2147483647

# ---------------------------------------------------------------------------
# Parameter-validation reference values
# ---------------------------------------------------------------------------

# Float precisions accepted by `float_type` parameters across the public API.
VALID_FLOAT_TYPES: tuple[str, ...] = ("float16", "float32", "float64")

# Parquet column encodings that PyArrow accepts via the `column_encoding` arg.
# Note: RLE_DICTIONARY is applied automatically for Arrow Dictionary columns
# and PyArrow rejects it as an explicit value here ("already used by default").
VALID_COLUMN_ENCODINGS: tuple[str, ...] = (
    "PLAIN",
    "BYTE_STREAM_SPLIT",
    "DELTA_BINARY_PACKED",
    "DELTA_LENGTH_BYTE_ARRAY",
    "DELTA_BYTE_ARRAY",
)

# Inclusive (min, max) compression-level ranges per codec, per the Parquet
# C++ implementation. Codecs not listed either don't accept a level (SNAPPY,
# LZ4) or don't have a stable range we want to enforce.
COMPRESSION_LEVEL_RANGES: dict[str, tuple[int, int]] = {
    "GZIP": (1, 9),
    "BROTLI": (0, 11),
    "ZSTD": (1, 22),
}
