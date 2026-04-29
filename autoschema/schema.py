import pyarrow as pa
import pyarrow.compute as pc
from . import constants


def infer_schema(table: pa.Table) -> pa.Schema:
    """
    Infers an optimized Arrow schema for the given table.
    - Downcasts integers to the smallest possible bit-width.
    - Downcasts floats if possible (e.g., float64 to float32).
    - Dictionary encodes strings if they have low cardinality.
    """
    fields = []

    for i in range(table.num_columns):
        column = table.column(i)
        name = table.schema.names[i]
        dtype = column.type

        new_type = dtype

        # Integer optimization
        if pa.types.is_integer(dtype):
            min_val = pc.min(column).as_py()
            max_val = pc.max(column).as_py()

            if min_val is not None and max_val is not None:
                if min_val >= 0:
                    if max_val <= constants.UINT8_MAX:
                        new_type = pa.uint8()
                    elif max_val <= constants.UINT16_MAX:
                        new_type = pa.uint16()
                    elif max_val <= constants.UINT32_MAX:
                        new_type = pa.uint32()
                else:
                    if (
                        min_val >= constants.INT8_MIN
                        and max_val <= constants.INT8_MAX
                    ):
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

        # Dictionary optimization (Downcast indices)
        elif pa.types.is_dictionary(dtype):
            dict_size = (
                len(column.chunk(0).dictionary) if column.num_chunks > 0 else 0
            )

            if dict_size <= constants.UINT8_MAX:
                index_type = pa.uint8()
            elif dict_size <= constants.UINT16_MAX:
                index_type = pa.uint16()
            else:
                index_type = pa.int32()

            if dtype.index_type != index_type:
                new_type = pa.dictionary(index_type, dtype.value_type)

        # Float optimization
        elif pa.types.is_floating(dtype):
            if dtype == pa.float64():
                new_type = pa.float32()

        # String optimization (Dictionary encoding)
        elif (
            pa.types.is_string(dtype)
            or pa.types.is_binary(dtype)
            or pa.types.is_large_string(dtype)
            or pa.types.is_large_binary(dtype)
        ):
            unique_count = len(column.unique())
            total_count = len(column)

            value_type = dtype
            if pa.types.is_large_string(dtype):
                value_type = pa.string()
            elif pa.types.is_large_binary(dtype):
                value_type = pa.binary()

            if (
                unique_count < constants.DICTIONARY_ENCODING_UNIQUE_LIMIT
                and (
                    total_count < constants.DICTIONARY_ENCODING_MIN_ROWS
                    or unique_count / total_count
                    < constants.DICTIONARY_ENCODING_RATIO_LIMIT
                )
            ):
                new_type = pa.dictionary(pa.int32(), value_type)

        # Boolean optimization (Ensure bit-packed)
        elif pa.types.is_boolean(dtype):
            new_type = pa.bool_()

        # Timestamp optimization (Downcast precision if possible)
        elif pa.types.is_timestamp(dtype):
            # If it's nanoseconds, consider if we can downcast to microseconds or milliseconds
            # for better compatibility and potentially smaller storage in some contexts.
            # For Parquet, it doesn't change much but helps with R/other engine compatibility.
            if dtype.unit == "ns":
                new_type = pa.timestamp("us", tz=dtype.tz)

        fields.append(
            pa.field(name, new_type, nullable=table.schema.field(i).nullable)
        )

    return pa.schema(fields)
