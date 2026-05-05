"""
Benchmarks comparing CSV, standard Parquet, Feather, and AutoParquet
across six synthetic datasets at three scale tiers (small, medium, large).

Usage:
    python benchmarks/bench.py
    python benchmarks/bench.py --compression zstd snappy gzip
    python benchmarks/bench.py --compression zstd:3 zstd:10 snappy
"""

import argparse
import os
import random
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.feather as feather
import pyarrow.parquet as pq

import autoparquet


class BenchFn(Protocol):
    """Signature of every per-format benchmark function.

    Defined as a Protocol (not a Callable alias) so mypy sees the
    `dataset_name` keyword name at the call site.
    """

    def __call__(
        self, table: pa.Table, path: str, dataset_name: str | None = None
    ) -> tuple[float, float, int]: ...


# Signature of every dataset generator: takes a single scale arg (row count or
# similar) and returns a pandas DataFrame.
GeneratorFn = Callable[[int], pd.DataFrame]

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

SBS96_CONTEXTS = [
    f"{five}{ref}[{ref}>{alt}]{three}"
    for ref in "CT"
    for alt in "ACGT"
    if alt != ref
    for five in "ACGT"
    for three in "ACGT"
]

SIGNATURE_NAMES = [f"SBS{i}" for i in range(1, 31)]

CASINO_GAMES = [
    "blackjack",
    "roulette",
    "poker",
    "craps",
    "baccarat",
    "slots",
]

WEATHER_STATIONS = [
    "KORD",
    "KJFK",
    "KLAX",
    "KATL",
    "KDEN",
    "KSFO",
    "KIAH",
    "KMIA",
    "KBOS",
    "KPHX",
]


def _random_names(n: int) -> list[str]:
    """Generate n plausible sample/player names."""
    prefixes = ["Sample", "Patient", "Donor", "Subject"]
    return [f"{random.choice(prefixes)}_{i:06d}" for i in range(n)]


def make_bed(n_rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    starts = rng.integers(0, 250_000_000, size=n_rows)
    lengths = rng.integers(100, 10_000, size=n_rows)
    return pd.DataFrame(
        {
            "chrom": rng.choice(CHROMOSOMES, size=n_rows),
            "start": starts,
            "end": starts + lengths,
            "name": [f"region_{i}" for i in range(n_rows)],
            "score": rng.integers(0, 1000, size=n_rows),
            "strand": rng.choice(["+", "-"], size=n_rows),
        }
    )


def make_sig_matrix(n_samples: int) -> pd.DataFrame:
    """Wide mutational-signature matrix: samples x SBS96 contexts."""
    rng = np.random.default_rng(7)
    names = _random_names(n_samples)
    counts = rng.poisson(lam=5, size=(n_samples, len(SBS96_CONTEXTS)))
    df = pd.DataFrame(counts, columns=SBS96_CONTEXTS)
    df.insert(0, "SampleName", names)
    return df


def make_sig_tidy(n_samples: int) -> pd.DataFrame:
    """Long-form: (SampleName, Feature, Count)."""
    wide = make_sig_matrix(n_samples)
    return wide.melt(id_vars="SampleName", var_name="Feature", value_name="Count")


def make_sig_freq(n_samples: int) -> pd.DataFrame:
    """(SampleName, Signature, Feature, Frequency)."""
    rng = np.random.default_rng(13)
    names = _random_names(n_samples)
    rows = []
    for name in names:
        for sig in SIGNATURE_NAMES:
            for feat in SBS96_CONTEXTS:
                freq = round(rng.exponential(0.01), 6)
                rows.append((name, sig, feat, freq))
    cols = ["SampleName", "Signature", "Feature", "Frequency"]
    return pd.DataFrame(rows, columns=cols)


def make_casino(n_rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    players = _random_names(max(50, n_rows // 100))
    tiers = ["bronze", "silver", "gold", "platinum"]
    return pd.DataFrame(
        {
            "player": rng.choice(players, size=n_rows),
            "game": rng.choice(CASINO_GAMES, size=n_rows),
            "table_id": rng.integers(1, 200, size=n_rows),
            "bet_amount": np.round(rng.uniform(5, 5000, n_rows), 2),
            "payout": np.round(rng.uniform(0, 15000, n_rows), 2),
            "hand_duration_s": np.round(rng.uniform(10, 600, n_rows), 1),
            "is_winner": rng.choice([True, False], size=n_rows),
            "loyalty_tier": rng.choice(tiers, size=n_rows),
        }
    )


def make_weather(n_rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    base = pd.Timestamp("2020-01-01")
    timestamps = [base + pd.Timedelta(minutes=int(m)) for m in range(n_rows)]
    conditions = [
        "clear",
        "cloudy",
        "rain",
        "snow",
        "fog",
        "thunderstorm",
    ]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "station": rng.choice(WEATHER_STATIONS, size=n_rows),
            "temperature_c": np.round(rng.normal(20, 12, n_rows), 1),
            "humidity_pct": np.round(rng.uniform(10, 100, n_rows), 1),
            "wind_speed_kph": np.round(rng.exponential(15, n_rows), 1),
            "pressure_hpa": np.round(rng.normal(1013, 10, n_rows), 1),
            "condition": rng.choice(conditions, size=n_rows),
        }
    )


# ---------------------------------------------------------------------------
# Dataset definitions: (name, generator, {scale: arg})
# ---------------------------------------------------------------------------

DATASETS: list[tuple[str, GeneratorFn, dict[str, int]]] = [
    (
        "BED intervals",
        make_bed,
        {
            "small": 1_000,
            "medium": 100_000,
            "large": 5_000_000,
        },
    ),
    (
        "Sig matrix (wide)",
        make_sig_matrix,
        {
            "small": 10,
            "medium": 500,
            "large": 5_000,
        },
    ),
    (
        "Sig tidy (long)",
        make_sig_tidy,
        {
            "small": 10,
            "medium": 500,
            "large": 5_000,
        },
    ),
    (
        "Sig freq decomp",
        make_sig_freq,
        {
            "small": 2,
            "medium": 20,
            "large": 100,
        },
    ),
    (
        "Casino ledger",
        make_casino,
        {
            "small": 1_000,
            "medium": 100_000,
            "large": 5_000_000,
        },
    ),
    (
        "Weather series",
        make_weather,
        {
            "small": 1_000,
            "medium": 100_000,
            "large": 2_000_000,
        },
    ),
]


# ---------------------------------------------------------------------------
# Compression config parsing
# ---------------------------------------------------------------------------


@dataclass
class CompressionConfig:
    """A compression algorithm with an optional level."""

    algo: str
    level: int | None

    @property
    def label(self) -> str:
        if self.level is not None:
            return f"AP {self.algo}:{self.level}"
        return f"AP {self.algo}"

    @classmethod
    def parse(cls, spec: str) -> "CompressionConfig":
        """Parse 'zstd', 'zstd:10', 'snappy', etc."""
        if ":" in spec:
            algo, lvl = spec.split(":", 1)
            return cls(algo=algo.lower(), level=int(lvl))
        return cls(algo=spec.lower(), level=None)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


@dataclass
class Result:
    dataset: str
    scale: str
    n_rows: int
    n_cols: int
    format: str
    write_s: float
    read_s: float
    file_bytes: int


# Hand-tuned per-dataset optimizations.
#
# Design rules (no compression-level tweaks):
#   * Default `float_type="float32"` is safe for these datasets — the source data
#     is rounded to 1–2 decimal places or is integer counts, so float32 is
#     lossless in practice and halves the storage of every floating-point column.
#   * `use_byte_stream_split=[cols]` for high-cardinality floats. BSS works
#     alongside dictionary encoding (Parquet picks dict when the column has few
#     unique values, BSS otherwise) so we don't sacrifice dict-friendly columns.
#   * `column_encoding={col: enc}` for columns where the default loses badly —
#     most notably DELTA_BINARY_PACKED for monotonic timestamps. Setting this
#     forces PyArrow's writer to disable per-column dict, so we pair it with
#     `use_dictionary=[list]` to re-enable dict on the columns that need it.
#   * `row_group_size` is enlarged for datasets with a single large
#     dict-encoded column (e.g. casino "player" at 50k unique entries). By
#     default the dictionary is stored once per row group; fewer/bigger row
#     groups means the dictionary is repeated fewer times.
OPTIMIZED_SETTINGS: dict[str, dict] = {
    "BED intervals": {
        "float_type": "float32",
    },
    "Sig matrix (wide)": {
        "float_type": "float32",
    },
    "Sig tidy (long)": {
        "float_type": "float32",
    },
    "Sig freq decomp": {
        "float_type": "float32",
        "use_byte_stream_split": ["Frequency"],
    },
    "Casino ledger": {
        "float_type": "float32",
        "use_byte_stream_split": ["bet_amount", "payout"],
        # 5 default row groups → 5 copies of the 50k-entry player dictionary;
        # one large group cuts ~3 MB at no measurable cost in write time.
        "row_group_size": 5_000_000,
    },
    "Weather series": {
        "float_type": "float32",
        # Sequential 1-minute timestamps → DELTA_BINARY_PACKED collapses to a
        # few KB versus the ~10 MB PLAIN encoding the default produces.
        "column_encoding": {"timestamp": "DELTA_BINARY_PACKED"},
        # Re-enable dict on every other column. With column_encoding set,
        # PyArrow's writer disables dict globally; passing a list overrides
        # that and gives per-column dict only for the listed columns.
        "use_dictionary": [
            "station",
            "condition",
            "temperature_c",
            "humidity_pct",
            "wind_speed_kph",
            "pressure_hpa",
        ],
    },
}


def _file_size(path: str) -> int:
    return os.path.getsize(path)


def bench_csv(
    table: pa.Table, path: str, dataset_name: str | None = None
) -> tuple[float, float, int]:
    t0 = time.perf_counter()
    pacsv.write_csv(table, path)
    write_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pacsv.read_csv(path)
    read_s = time.perf_counter() - t0

    return write_s, read_s, _file_size(path)


def bench_parquet_standard(
    table: pa.Table, path: str, dataset_name: str | None = None
) -> tuple[float, float, int]:
    t0 = time.perf_counter()
    pq.write_table(table, path, compression="zstd")
    write_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pq.read_table(path)
    read_s = time.perf_counter() - t0

    return write_s, read_s, _file_size(path)


def bench_feather(
    table: pa.Table, path: str, dataset_name: str | None = None
) -> tuple[float, float, int]:
    t0 = time.perf_counter()
    feather.write_feather(table, path, compression="zstd")
    write_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    feather.read_feather(path)
    read_s = time.perf_counter() - t0

    return write_s, read_s, _file_size(path)


_LEVELLESS_CODECS = {"snappy", "lz4"}


def make_bench_autoparquet(
    cfg: CompressionConfig,
) -> BenchFn:
    """Return a benchmark function for AutoParquet with specific compression."""

    def _bench(
        table: pa.Table, path: str, dataset_name: str | None = None
    ) -> tuple[float, float, int]:
        kwargs: dict = {"compression": cfg.algo}
        if cfg.algo in _LEVELLESS_CODECS:
            kwargs["compression_level"] = None
        elif cfg.level is not None:
            kwargs["compression_level"] = cfg.level

        t0 = time.perf_counter()
        autoparquet.write_parquet(table, path, **kwargs)
        write_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        pq.read_table(path)
        read_s = time.perf_counter() - t0

        return write_s, read_s, _file_size(path)

    return _bench


def make_bench_autoparquet_optimized(
    float_type: str = "float32",
) -> BenchFn:
    """Build a benchmark fn that applies dataset-specific OPTIMIZED_SETTINGS.

    The `float_type` argument overrides the per-dataset default, so callers can
    compare e.g. "float32 + tuned encodings" vs "float16 + tuned encodings".
    """

    def _bench(
        table: pa.Table, path: str, dataset_name: str | None = None
    ) -> tuple[float, float, int]:
        per_dataset = OPTIMIZED_SETTINGS.get(dataset_name, {}) if dataset_name else {}
        settings = dict(per_dataset)
        settings["float_type"] = float_type
        kwargs = {"compression": "zstd", **settings}

        t0 = time.perf_counter()
        autoparquet.write_parquet(table, path, **kwargs)
        write_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        pq.read_table(path)
        read_s = time.perf_counter() - t0

        return write_s, read_s, _file_size(path)

    return _bench


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _ratio(ref: int, target: int) -> str:
    if target <= 0:
        return "  n/a"
    r = ref / target
    return f"{r:5.1f}:1"


_EMPTY = Result("", "", 0, 0, "", 0, 0, 0)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark AutoParquet against CSV, Parquet, and Feather."
    )
    p.add_argument(
        "-c",
        "--compression",
        nargs="+",
        default=["zstd"],
        metavar="ALGO[:LEVEL]",
        help=(
            "One or more AutoParquet compression configs. "
            "Examples: zstd  zstd:10  snappy  gzip:6  "
            "(default: zstd)"
        ),
    )
    p.add_argument(
        "--with-optimized",
        action="store_true",
        help=(
            "Also run the hand-tuned per-dataset 'Optimized' and 'Opt+fp16' "
            "variants and print a comparison table showing how much they "
            "shrink each dataset relative to the AutoParquet baseline."
        ),
    )
    return p


def run() -> None:
    args = build_cli().parse_args()
    configs = [CompressionConfig.parse(s) for s in args.compression]

    # Build the list of formats to benchmark.
    # Baselines are always CSV, standard Parquet, and Feather.
    # Then one AutoParquet variant per --compression config.
    baselines: list[tuple[str, str, BenchFn]] = [
        ("CSV", ".csv", bench_csv),
        ("Parquet (std)", ".parquet", bench_parquet_standard),
        ("Feather", ".feather", bench_feather),
    ]
    ap_formats: list[tuple[str, str, BenchFn]] = []
    for i, cfg in enumerate(configs):
        ext = f".ap{i}.parquet"
        ap_formats.append((cfg.label, ext, make_bench_autoparquet(cfg)))
    ap_labels = [label for label, _, _ in ap_formats]

    # Hand-tuned optimized variants are only built when --with-optimized is set.
    # When skipped they don't run, don't show in the file-size table, and don't
    # produce the comparison table at the end.
    opt_formats: list[tuple[str, str, BenchFn]] = []
    opt_labels: list[str] = []
    if args.with_optimized:
        opt_formats = [
            (
                "AP Optimized",
                ".opt.ap.parquet",
                make_bench_autoparquet_optimized("float32"),
            ),
            (
                "AP Opt+fp16",
                ".opt16.ap.parquet",
                make_bench_autoparquet_optimized("float16"),
            ),
        ]
        opt_labels = [label for label, _, _ in opt_formats]

    all_formats = baselines + ap_formats + opt_formats

    results: list[Result] = []

    for ds_name, generator, scales in DATASETS:
        for scale, arg in scales.items():
            print(f"\n{'=' * 60}")
            print(f"  {ds_name}  [{scale}]  (arg={arg:,})")
            print(f"{'=' * 60}")

            df = generator(arg)
            table = pa.Table.from_pandas(df)
            n_rows, n_cols = len(df), len(df.columns)
            print(f"  {n_rows:>12,} rows x {n_cols} cols")

            with tempfile.TemporaryDirectory() as tmp:
                for fmt_name, ext, bench_fn in all_formats:
                    path = os.path.join(tmp, f"data{ext}")
                    w, r, sz = bench_fn(table, path, dataset_name=ds_name)
                    results.append(
                        Result(
                            dataset=ds_name,
                            scale=scale,
                            n_rows=n_rows,
                            n_cols=n_cols,
                            format=fmt_name,
                            write_s=w,
                            read_s=r,
                            file_bytes=sz,
                        )
                    )
                    print(
                        f"  {fmt_name:<20s}  "
                        f"size={_human_bytes(sz):>10s}  "
                        f"write={w:.3f}s  "
                        f"read={r:.3f}s"
                    )

    # Group results by (dataset, scale)
    by_key: dict[tuple[str, str], dict[str, Result]] = {}
    for result in results:
        by_key.setdefault((result.dataset, result.scale), {})[result.format] = result

    # ------------------------------------------------------------------
    # Summary: file sizes
    #
    # Columns: baselines (CSV / Parquet / Feather) + each user-requested AP
    # variant + (only when --with-optimized) the hand-tuned variants.
    # ------------------------------------------------------------------
    size_labels = ap_labels + opt_labels
    size_hdr = "".join(f" {lbl:>13s}" for lbl in size_labels)
    w = 62 + 14 * len(size_labels)
    sep = "=" * w
    dash = "-" * w

    print(f"\n\n{sep}")
    print("SUMMARY: file sizes")
    print(sep)
    print(
        f"{'Dataset':<22s} {'Scale':<7s} {'Rows':>10s}"
        f"  {'CSV':>9s} {'Parquet':>9s}" + size_hdr
    )
    print(dash)

    for (ds, sc), fmts in by_key.items():
        csv_r = fmts.get("CSV", _EMPTY)
        size_cols = "".join(
            f" {_human_bytes(fmts.get(lbl, _EMPTY).file_bytes):>13s}"
            for lbl in size_labels
        )
        print(
            f"{ds:<22s} {sc:<7s} {csv_r.n_rows:>10,}"
            f"  {_human_bytes(fmts.get('CSV', _EMPTY).file_bytes):>9s}"
            f" {_human_bytes(fmts.get('Parquet (std)', _EMPTY).file_bytes):>9s}"
            + size_cols
        )

    # ------------------------------------------------------------------
    # Summary: compression ratios (vs CSV, vs Parquet, vs Feather)
    # One table per user-requested AP compression config.
    # ------------------------------------------------------------------
    for lbl in ap_labels:
        print(f"\n{sep}")
        print(f"SUMMARY: {lbl} compression ratios (higher = better)")
        print(sep)
        print(
            f"{'Dataset':<22s} {'Scale':<7s}"
            f"  {'Size':>9s}"
            f" {'vs CSV':>7s}"
            f" {'vs Pqt':>7s}"
            f" {'vs Fth':>7s}"
        )
        print(dash)

        for (ds, sc), fmts in by_key.items():
            ap_sz = fmts.get(lbl, _EMPTY).file_bytes
            csv_sz = fmts.get("CSV", _EMPTY).file_bytes
            pqt_sz = fmts.get("Parquet (std)", _EMPTY).file_bytes
            fth_sz = fmts.get("Feather", _EMPTY).file_bytes
            print(
                f"{ds:<22s} {sc:<7s}"
                f"  {_human_bytes(ap_sz):>9s}"
                f" {_ratio(csv_sz, ap_sz):>7s}"
                f" {_ratio(pqt_sz, ap_sz):>7s}"
                f" {_ratio(fth_sz, ap_sz):>7s}"
            )

    # ------------------------------------------------------------------
    # Summary: optimized vs. AutoParquet baseline (only with --with-optimized)
    #
    # A single table comparing the AutoParquet baseline against the hand-tuned
    # variants. The "baseline" is the first --compression config (the default
    # `AP zstd` unless the user passed others). Each optimized variant gets a
    # size column and a "vs base" ratio column showing the savings.
    # ------------------------------------------------------------------
    if opt_labels:
        baseline_label = ap_labels[0]
        cmp_w = 38 + 11 + 14 * len(opt_labels) * 2
        cmp_sep = "=" * cmp_w
        cmp_dash = "-" * cmp_w

        # Build the header row dynamically: each opt variant gets two columns
        # (size, ratio vs baseline).
        opt_size_hdr = "".join(f" {lbl:>13s} {'vs base':>10s}" for lbl in opt_labels)

        print(f"\n{cmp_sep}")
        print(
            f"SUMMARY: optimized vs. AutoParquet baseline ({baseline_label})"
            "  (higher ratio = better)"
        )
        print(cmp_sep)
        print(f"{'Dataset':<22s} {'Scale':<7s} {'Baseline':>10s}" + opt_size_hdr)
        print(cmp_dash)

        for (ds, sc), fmts in by_key.items():
            base_sz = fmts.get(baseline_label, _EMPTY).file_bytes
            opt_cells = "".join(
                f" {_human_bytes(fmts.get(lbl, _EMPTY).file_bytes):>13s}"
                f" {_ratio(base_sz, fmts.get(lbl, _EMPTY).file_bytes):>10s}"
                for lbl in opt_labels
            )
            print(f"{ds:<22s} {sc:<7s} {_human_bytes(base_sz):>10s}" + opt_cells)


if __name__ == "__main__":
    run()
