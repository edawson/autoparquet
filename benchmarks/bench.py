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
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.feather as feather
import pyarrow.parquet as pq

import autoparquet

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
    "blackjack", "roulette", "poker",
    "craps", "baccarat", "slots",
]

WEATHER_STATIONS = [
    "KORD", "KJFK", "KLAX", "KATL", "KDEN",
    "KSFO", "KIAH", "KMIA", "KBOS", "KPHX",
]


def _random_names(n: int) -> list[str]:
    """Generate n plausible sample/player names."""
    prefixes = ["Sample", "Patient", "Donor", "Subject"]
    return [f"{random.choice(prefixes)}_{i:06d}" for i in range(n)]


def make_bed(n_rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    starts = rng.integers(0, 250_000_000, size=n_rows)
    lengths = rng.integers(100, 10_000, size=n_rows)
    return pd.DataFrame({
        "chrom": rng.choice(CHROMOSOMES, size=n_rows),
        "start": starts,
        "end": starts + lengths,
        "name": [f"region_{i}" for i in range(n_rows)],
        "score": rng.integers(0, 1000, size=n_rows),
        "strand": rng.choice(["+", "-"], size=n_rows),
    })


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
    return wide.melt(
        id_vars="SampleName", var_name="Feature", value_name="Count"
    )


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
    return pd.DataFrame({
        "player": rng.choice(players, size=n_rows),
        "game": rng.choice(CASINO_GAMES, size=n_rows),
        "table_id": rng.integers(1, 200, size=n_rows),
        "bet_amount": np.round(rng.uniform(5, 5000, n_rows), 2),
        "payout": np.round(rng.uniform(0, 15000, n_rows), 2),
        "hand_duration_s": np.round(rng.uniform(10, 600, n_rows), 1),
        "is_winner": rng.choice([True, False], size=n_rows),
        "loyalty_tier": rng.choice(tiers, size=n_rows),
    })


def make_weather(n_rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    base = pd.Timestamp("2020-01-01")
    timestamps = [
        base + pd.Timedelta(minutes=int(m)) for m in range(n_rows)
    ]
    conditions = [
        "clear", "cloudy", "rain", "snow", "fog", "thunderstorm",
    ]
    return pd.DataFrame({
        "timestamp": timestamps,
        "station": rng.choice(WEATHER_STATIONS, size=n_rows),
        "temperature_c": np.round(rng.normal(20, 12, n_rows), 1),
        "humidity_pct": np.round(rng.uniform(10, 100, n_rows), 1),
        "wind_speed_kph": np.round(rng.exponential(15, n_rows), 1),
        "pressure_hpa": np.round(rng.normal(1013, 10, n_rows), 1),
        "condition": rng.choice(conditions, size=n_rows),
    })


# ---------------------------------------------------------------------------
# Dataset definitions: (name, generator, {scale: arg})
# ---------------------------------------------------------------------------

DATASETS: list[tuple[str, callable, dict[str, int]]] = [
    ("BED intervals", make_bed, {
        "small": 1_000, "medium": 100_000, "large": 5_000_000,
    }),
    ("Sig matrix (wide)", make_sig_matrix, {
        "small": 10, "medium": 500, "large": 5_000,
    }),
    ("Sig tidy (long)", make_sig_tidy, {
        "small": 10, "medium": 500, "large": 5_000,
    }),
    ("Sig freq decomp", make_sig_freq, {
        "small": 2, "medium": 20, "large": 100,
    }),
    ("Casino ledger", make_casino, {
        "small": 1_000, "medium": 100_000, "large": 5_000_000,
    }),
    ("Weather series", make_weather, {
        "small": 1_000, "medium": 100_000, "large": 2_000_000,
    }),
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


def _file_size(path: str) -> int:
    return os.path.getsize(path)


def bench_csv(
    table: pa.Table, path: str
) -> tuple[float, float, int]:
    t0 = time.perf_counter()
    pacsv.write_csv(table, path)
    write_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pacsv.read_csv(path)
    read_s = time.perf_counter() - t0

    return write_s, read_s, _file_size(path)


def bench_parquet_standard(
    table: pa.Table, path: str
) -> tuple[float, float, int]:
    t0 = time.perf_counter()
    pq.write_table(table, path, compression="zstd")
    write_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pq.read_table(path)
    read_s = time.perf_counter() - t0

    return write_s, read_s, _file_size(path)


def bench_feather(
    table: pa.Table, path: str
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
) -> callable:
    """Return a benchmark function for AutoParquet with specific compression."""
    def _bench(table: pa.Table, path: str) -> tuple[float, float, int]:
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


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


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
        "-c", "--compression",
        nargs="+",
        default=["zstd"],
        metavar="ALGO[:LEVEL]",
        help=(
            "One or more AutoParquet compression configs. "
            "Examples: zstd  zstd:10  snappy  gzip:6  "
            "(default: zstd)"
        ),
    )
    return p


def run() -> None:
    args = build_cli().parse_args()
    configs = [CompressionConfig.parse(s) for s in args.compression]

    # Build the list of formats to benchmark.
    # Baselines are always CSV, standard Parquet, and Feather.
    # Then one ArrowPack variant per compression config.
    baselines: list[tuple[str, str, callable]] = [
        ("CSV", ".csv", bench_csv),
        ("Parquet (std)", ".parquet", bench_parquet_standard),
        ("Feather", ".feather", bench_feather),
    ]
    ap_formats: list[tuple[str, str, callable]] = []
    for i, cfg in enumerate(configs):
        ext = f".ap{i}.parquet"
        ap_formats.append((cfg.label, ext, make_bench_autoparquet(cfg)))

    all_formats = baselines + ap_formats
    ap_labels = [label for label, _, _ in ap_formats]

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
                    w, r, sz = bench_fn(table, path)
                    results.append(Result(
                        dataset=ds_name, scale=scale,
                        n_rows=n_rows, n_cols=n_cols,
                        format=fmt_name,
                        write_s=w, read_s=r, file_bytes=sz,
                    ))
                    print(
                        f"  {fmt_name:<20s}  "
                        f"size={_human_bytes(sz):>10s}  "
                        f"write={w:.3f}s  "
                        f"read={r:.3f}s"
                    )

    # Group results by (dataset, scale)
    by_key: dict[tuple[str, str], dict[str, Result]] = {}
    for r in results:
        by_key.setdefault((r.dataset, r.scale), {})[r.format] = r

    # ------------------------------------------------------------------
    # Summary: file sizes
    # ------------------------------------------------------------------
    ap_hdr = "".join(f" {lbl:>12s}" for lbl in ap_labels)
    w = 62 + 13 * len(ap_labels)
    sep = "=" * w
    dash = "-" * w

    print(f"\n\n{sep}")
    print("SUMMARY: file sizes")
    print(sep)
    print(
        f"{'Dataset':<22s} {'Scale':<7s} {'Rows':>10s}"
        f"  {'CSV':>9s} {'Parquet':>9s}"
        + ap_hdr
    )
    print(dash)

    for (ds, sc), fmts in by_key.items():
        csv_r = fmts.get("CSV", _EMPTY)
        ap_cols = "".join(
            f" {_human_bytes(fmts.get(lbl, _EMPTY).file_bytes):>12s}"
            for lbl in ap_labels
        )
        print(
            f"{ds:<22s} {sc:<7s} {csv_r.n_rows:>10,}"
            f"  {_human_bytes(fmts.get('CSV', _EMPTY).file_bytes):>9s}"
            f" {_human_bytes(fmts.get('Parquet (std)', _EMPTY).file_bytes):>9s}"
            + ap_cols
        )

    # ------------------------------------------------------------------
    # Summary: compression ratios (vs CSV, vs Parquet, vs Feather)
    # ------------------------------------------------------------------
    # One table per ArrowPack variant
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


if __name__ == "__main__":
    run()
