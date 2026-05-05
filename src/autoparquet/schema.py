import pyarrow as pa
import pyarrow.compute as pc

from . import constants
from .utils.logger import get_logger

logger = get_logger(__name__)


def _smallest_index_type(n: int) -> pa.DataType:
    """Returns the smallest unsigned integer type that can index `n` values."""
    if n <= constants.UINT8_MAX:
        return pa.uint8()
    if n <= constants.UINT16_MAX:
        return pa.uint16()
    return pa.int32()


def infer_schema(table: pa.Table, float_type: str = "float64") -> pa.Schema:
    """Infer an optimized Arrow schema for the given table.

    Optimizations applied per column:
      - Integers: downcast to the smallest type that fits the data range.
      - Floats: optionally reduce precision (float64 -> float32 or float16).
      - Strings/Binary: dictionary-encode low-cardinality columns with the
        smallest possible index type; for high-cardinality uniform-length
        columns, use FixedSizeBinary.
      - Existing dictionaries: downcast index type if oversized.
      - Timestamps: downcast nanoseconds to microseconds for compatibility.

    Args:
        table: The Arrow table to analyze.
        float_type: Target float precision ("float16", "float32", or "float64").
                    Defaults to "float64" (lossless). Use "float32" to halve
                    float storage when full precision is not needed.

    Raises:
        TypeError: if ``table`` is not a pyarrow.Table.
        ValueError: if ``float_type`` is not one of the supported precisions.
    """
    if not isinstance(table, pa.Table):
        raise TypeError(
            f"infer_schema expects a pyarrow.Table, got {type(table).__name__}"
        )
    if float_type not in constants.VALID_FLOAT_TYPES:
        raise ValueError(
            f"float_type must be one of {constants.VALID_FLOAT_TYPES}, "
            f"got {float_type!r}"
        )

    fields = []

    for i in range(table.num_columns):
        column = table.column(i)
        name = table.schema.names[i]
        dtype = column.type
        new_type = dtype

        # 1. Integer optimization: find the smallest type that fits.
        if pa.types.is_integer(dtype):
            min_val = pc.min(column).as_py()  # type: ignore[attr-defined]
            max_val = pc.max(column).as_py()  # type: ignore[attr-defined]

            if min_val is not None and max_val is not None:
                if min_val >= 0:
                    if max_val <= constants.UINT8_MAX:
                        new_type = pa.uint8()
                    elif max_val <= constants.UINT16_MAX:
                        new_type = pa.uint16()
                    elif max_val <= constants.UINT32_MAX:
                        new_type = pa.uint32()
                else:
                    if min_val >= constants.INT8_MIN and max_val <= constants.INT8_MAX:
                        new_type = pa.int8()
                    elif (
                        min_val >= constants.INT16_MIN
                        and max_val <= constants.INT16_MAX
                    ):
                        new_type = pa.int16()
                    elif (
                        min_val >= constants.INT32_MIN
                        and max_val <= constants.INT32_MAX
                    ):
                        new_type = pa.int32()

        # 2. Float optimization: reduce precision when requested.
        elif pa.types.is_floating(dtype):
            if float_type == "float16":
                new_type = pa.float16()
            elif float_type == "float32":
                new_type = pa.float32()
            # float64 is a no-op (keep original)

        # 3. String/Binary optimization.
        elif (
            pa.types.is_string(dtype)
            or pa.types.is_binary(dtype)
            or pa.types.is_large_string(dtype)
            or pa.types.is_large_binary(dtype)
        ):
            unique_count = len(column.unique())
            total_count = len(column)

            is_low_cardinality = (
                unique_count < constants.DICTIONARY_ENCODING_UNIQUE_LIMIT
                and (
                    total_count < constants.DICTIONARY_ENCODING_MIN_ROWS
                    or unique_count / total_count
                    < constants.DICTIONARY_ENCODING_RATIO_LIMIT
                )
            )

            if is_low_cardinality:
                # Dictionary-encode with the smallest possible index type.
                # Parquet will RLE-encode the indices automatically.
                index_type = _smallest_index_type(unique_count)
                value_type = (
                    pa.string()
                    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype)
                    else pa.binary()
                )
                new_type = pa.dictionary(index_type, value_type)
            else:
                # For high-cardinality columns, check for uniform length.
                # FixedSizeBinary removes the 4-byte offset overhead per row.
                if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
                    lengths = pc.utf8_length(column)  # type: ignore[attr-defined]
                else:
                    lengths = pc.binary_length(column)  # type: ignore[attr-defined]

                min_len = pc.min(lengths).as_py()  # type: ignore[attr-defined]
                max_len = pc.max(lengths).as_py()  # type: ignore[attr-defined]

                if min_len is not None and min_len == max_len and min_len > 0:
                    new_type = pa.binary(min_len)

        # 4. Existing dictionary: downcast the index type if oversized.
        elif pa.types.is_dictionary(dtype):
            dict_size = len(column.chunk(0).dictionary) if column.num_chunks > 0 else 0
            index_type = _smallest_index_type(dict_size)
            if dtype.index_type != index_type:
                new_type = pa.dictionary(index_type, dtype.value_type)

        # 5. Boolean: Arrow already bit-packs booleans; no change needed.
        elif pa.types.is_boolean(dtype):
            pass

        # 6. Timestamp: downcast ns to us for broader compatibility.
        elif pa.types.is_timestamp(dtype):
            if dtype.unit == "ns":
                new_type = pa.timestamp("us", tz=dtype.tz)

        if new_type != dtype:
            logger.debug("column '%s': %s → %s", name, dtype, new_type)

        fields.append(pa.field(name, new_type, nullable=table.schema.field(i).nullable))

    return pa.schema(fields)
