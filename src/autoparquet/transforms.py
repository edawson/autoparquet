from typing import Tuple

import pyarrow as pa
import pyarrow.compute as pc

from .schema import _smallest_index_type


def map_to_vocabulary(
    table: pa.Table, column_name: str, vocabulary: list[str]
) -> pa.Table:
    """
    Maps a column to a fixed vocabulary to ensure stable integer IDs.

    This ensures that a given string always maps to the same ID (its index in the
    vocabulary) across different files, which is critical for joining sparse
    datasets like kmers.

    - IDs correspond to the index in the vocabulary list.
    - Values not present in the vocabulary will be mapped to null.
    - The underlying storage uses a DictionaryArray with the smallest possible
      integer index type (uint8, uint16, or int32) to minimize memory usage.
    """
    col_idx = table.schema.get_field_index(column_name)
    if col_idx == -1:
        raise ValueError(f"Column '{column_name}' not found in table.")

    column = table.column(col_idx)

    vocab_array = pa.array(vocabulary, type=pa.string())
    index_type = _smallest_index_type(len(vocabulary))

    def encode_chunk(chunk: pa.Array) -> pa.DictionaryArray:
        # Map chunk values to vocabulary indices
        indices = pc.index_in(chunk, value_set=vocab_array)
        indices = indices.cast(index_type)
        return pa.DictionaryArray.from_arrays(indices, vocab_array)

    new_chunks = [encode_chunk(chunk) for chunk in column.chunks]
    new_column = pa.chunked_array(new_chunks)

    return table.set_column(col_idx, column_name, new_column)


def cast_to_fixed_binary(table: pa.Table, column_name: str) -> pa.Table:
    """
    Casts a string/binary column to a FixedSizeBinary type.

    This is the most efficient storage format for fixed-length sequences like kmers,
    as it eliminates the 4-byte-per-row offset overhead required by standard strings.

    - All entries in the column must have the same length.
    - Raises ValueError if lengths are non-uniform or the column is empty.
    """
    col_idx = table.schema.get_field_index(column_name)
    if col_idx == -1:
        raise ValueError(f"Column '{column_name}' not found in table.")

    column = table.column(col_idx)

    # Validate uniform length
    lengths = pc.binary_length(column)
    min_len_scalar = pc.min(lengths)
    max_len_scalar = pc.max(lengths)

    min_len = min_len_scalar.as_py() if min_len_scalar.is_valid else None
    max_len = max_len_scalar.as_py() if max_len_scalar.is_valid else None

    if min_len is None:
        raise ValueError(f"Column '{column_name}' is empty or contains only nulls.")

    if min_len != max_len:
        raise ValueError(
            f"Column '{column_name}' requires uniform length for FixedSizeBinary "
            f"(found min={min_len}, max={max_len})."
        )

    new_type = pa.binary(min_len)
    new_column = column.cast(new_type)

    return table.set_column(col_idx, column_name, new_column)


def strings_to_fixed_size_binary(table: pa.Table) -> pa.Table:
    """
    Detects string/binary columns with uniform length and converts them to FixedSizeBinary.

    This is particularly efficient for kmers and other fixed-length sequences.
    """
    new_fields = []
    new_columns = []

    for i in range(table.num_columns):
        column = table.column(i)
        field = table.schema.field(i)
        dtype = field.type

        if (
            pa.types.is_string(dtype)
            or pa.types.is_binary(dtype)
            or pa.types.is_large_string(dtype)
            or pa.types.is_large_binary(dtype)
        ):
            # Check for uniform length
            lengths = pc.binary_length(column)
            min_len_scalar = pc.min(lengths)
            max_len_scalar = pc.max(lengths)

            min_len = min_len_scalar.as_py() if min_len_scalar.is_valid else None
            max_len = max_len_scalar.as_py() if max_len_scalar.is_valid else None

            if min_len is not None and min_len == max_len and min_len > 0:
                new_type = pa.binary(min_len)
                new_columns.append(column.cast(new_type))
                new_fields.append(
                    pa.field(field.name, new_type, nullable=field.nullable)
                )
                continue

        new_columns.append(column)
        new_fields.append(field)

    return pa.Table.from_arrays(new_columns, schema=pa.schema(new_fields))


def extract_string_vocabulary(
    table: pa.Table, column_name: str
) -> Tuple[pa.Table, list[str]]:
    """
    Extract unique string values from a column and create an integer-indexed mapping.

    Returns the table with the column replaced by integer indices (dictionary-encoded
    with the smallest possible index type) and a list of unique values in sorted order.

    This is useful for:
    - Storing column vocabularies for sharing across files
    - Reducing storage for categorical columns with many unique values
    - Creating stable, reproducible mappings across datasets

    Example:
        table, vocab = extract_string_vocabulary(table, "chromosome")
        # vocab = ["chr1", "chr10", "chr11", ..., "chrX", "chrY"]
        # table.column("chromosome") is now indexed with uint8 indices
    """
    col_idx = table.schema.get_field_index(column_name)
    if col_idx == -1:
        raise ValueError(f"Column '{column_name}' not found in table.")

    column = table.column(col_idx)

    # Get unique values, sorted
    unique_vals = pc.unique(column)
    sorted_vals = pc.sort_indices(unique_vals)
    vocabulary = pc.take(unique_vals, sorted_vals).to_pylist()

    # Map column to indices using the vocabulary
    mapped_table = map_to_vocabulary(table, column_name, vocabulary)

    return mapped_table, vocabulary
