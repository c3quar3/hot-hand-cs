"""Build read-only interval-distribution histograms from frozen state matrices.

The figures summarize the time support of the existing primary association
models. They use only ``d_tick`` and ``match_id`` from the persisted E-A/E-B
state matrices, never outcomes or residual values. Re-running skips rendering
when the source hashes, display specification, script, and exports match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "unified_evidence_two_partition_confirmation_v1" / "full"
FIGURE_DIR = PROJECT_ROOT / "docs" / "稿件" / "figures" / "time_interval_distribution_precheck_v1"
SOURCE_CSV = FIGURE_DIR / "source_data.csv"
SOURCE_JSON = FIGURE_DIR / "source_data.json"
MANIFEST_PATH = FIGURE_DIR / "build_manifest.json"

PARTITIONS = ("E-A", "E-B")
SECONDS_PER_TICK = 1.0 / 64.0
REFERENCE_SECONDS = 46.45
LOG_BIN_MIN_SECONDS = SECONDS_PER_TICK
LOG_BIN_MAX_SECONDS = 1200.0
LOG_BIN_COUNT = 32
RASTER_DPI = 600
WIDTH_MM = 89.0
HEIGHT_MM = 62.23
FIGURE_WIDTH_IN = WIDTH_MM / 25.4
FIGURE_HEIGHT_IN = HEIGHT_MM / 25.4

COLORS = {
    "E-A": "#0F4D92",
    "E-B": "#42949E",
    "grid": "#D9D9D9",
    "quantile": "#767676",
    "reference": "#222222",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_path(partition: str) -> Path:
    return RESULT_ROOT / partition / "state_matrix.parquet"


def output_base(partition: str) -> Path:
    suffix = partition.lower().replace("-", "")
    return FIGURE_DIR / f"figure_time_interval_distribution_{suffix}_precheck_v1"


def output_paths() -> list[Path]:
    paths = [SOURCE_CSV, SOURCE_JSON]
    for partition in PARTITIONS:
        base = output_base(partition)
        paths.extend(base.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png", ".tiff"))
    return paths


def load_intervals(partition: str) -> pd.DataFrame:
    path = state_path(partition)
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen state matrix: {path}")
    frame = pd.read_parquet(path, columns=["match_id", "d_tick"])
    before_count = len(frame)
    if before_count == 0:
        raise ValueError(f"{partition} has no primary-analysis rows")
    if frame["d_tick"].isna().any() or not np.isfinite(frame["d_tick"]).all() or (frame["d_tick"] <= 0).any():
        raise ValueError(f"{partition} contains a missing, non-finite, or non-positive d_tick")
    frame = frame.copy()
    frame["seconds"] = frame["d_tick"].astype(float) * SECONDS_PER_TICK
    if frame["seconds"].min() < LOG_BIN_MIN_SECONDS or frame["seconds"].max() > LOG_BIN_MAX_SECONDS:
        raise ValueError(f"{partition} has an interval outside the fixed display support")
    if len(frame) != before_count:
        raise AssertionError("No rows may be excluded from the interval display")
    return frame


def summarize(partition: str, frame: pd.DataFrame, bins: np.ndarray) -> tuple[pd.DataFrame, dict]:
    seconds = frame["seconds"].to_numpy()
    counts, _ = np.histogram(seconds, bins=bins)
    table = pd.DataFrame(
        {
            "partition": partition,
            "bin_left_seconds": bins[:-1],
            "bin_right_seconds": bins[1:],
            "bin_center_geometric_seconds": np.sqrt(bins[:-1] * bins[1:]),
            "record_count": counts,
            "record_percent": counts / len(seconds) * 100.0,
        }
    )
    quantiles = {f"p{int(q * 100):02d}": float(np.quantile(seconds, q)) for q in (0.10, 0.50, 0.90)}
    metadata = {
        "records": int(len(frame)),
        "matches": int(frame["match_id"].nunique()),
        "minimum_seconds": float(seconds.min()),
        "maximum_seconds": float(seconds.max()),
        "mean_seconds": float(seconds.mean()),
        "median_seconds": float(np.median(seconds)),
        "quantiles_seconds": quantiles,
    }
    return table, metadata


def render(partition: str, table: pd.DataFrame, metadata: dict, y_max: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), layout="constrained")
    left = table["bin_left_seconds"].to_numpy()
    width = table["bin_right_seconds"].to_numpy() - left
    ax.bar(
        left,
        table["record_percent"].to_numpy(),
        width=width,
        align="edge",
        color=COLORS[partition],
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
    quantiles = metadata["quantiles_seconds"]
    markers = [
        (quantiles["p10"], COLORS["quantile"], "--", 0.8),
        (REFERENCE_SECONDS, COLORS["reference"], "-", 1.0),
        (quantiles["p50"], COLORS["quantile"], "--", 0.8),
        (quantiles["p90"], COLORS["quantile"], "--", 0.8),
    ]
    for x_value, color, style, line_width in markers:
        ax.axvline(x_value, color=color, linestyle=style, linewidth=line_width, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(LOG_BIN_MIN_SECONDS, LOG_BIN_MAX_SECONDS)
    ax.set_ylim(0, y_max)
    ax.set_xticks([0.1, 1, 10, 100, 1000])
    ax.set_xticklabels(["0.1", "1", "10", "100", "1,000"])
    ax.set_yticks(np.arange(0, y_max + 0.01, 2))
    ax.set_xlabel("Interval since prior eligible event (s)", labelpad=4)
    ax.set_ylabel("Participant-event records (%)", labelpad=4)
    ax.set_title(partition, loc="left", pad=4, fontweight="bold")
    fig.suptitle("Dashed: P10/P50/P90     Solid: D1-A reference (46.45 s)", fontsize=6.0, y=0.995)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, zorder=0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="both", width=0.8, length=3)

    base = output_base(partition)
    for suffix, options in {
        ".pdf": {},
        ".svg": {},
        ".png": {"dpi": RASTER_DPI},
        ".tiff": {"dpi": RASTER_DPI},
    }.items():
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight", pad_inches=0.03, **options)
    plt.close(fig)


def current_manifest(source_hashes: dict[str, str], y_max: float) -> dict:
    return {
        "figure_id": "time_interval_distribution_precheck_v1",
        "backend": "python-matplotlib",
        "script_sha256": sha256(Path(__file__)),
        "state_matrix_sha256": source_hashes,
        "definition": "strict prior eligible participant-event interval in seconds",
        "seconds_per_tick": SECONDS_PER_TICK,
        "display_support_seconds": [LOG_BIN_MIN_SECONDS, LOG_BIN_MAX_SECONDS],
        "binning": {"scale": "log10 seconds", "bin_count": LOG_BIN_COUNT},
        "reference_seconds": REFERENCE_SECONDS,
        "shared_y_max_percent": y_max,
        "outputs": [path.name for path in output_paths()],
    }


def is_current(manifest: dict) -> bool:
    if not MANIFEST_PATH.exists() or not all(path.exists() for path in output_paths()):
        return False
    previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return all(previous.get(key) == value for key, value in manifest.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Render even when all recorded inputs are unchanged.")
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    bins = np.logspace(math.log10(LOG_BIN_MIN_SECONDS), math.log10(LOG_BIN_MAX_SECONDS), LOG_BIN_COUNT + 1)
    frames = {partition: load_intervals(partition) for partition in PARTITIONS}
    summaries = {partition: summarize(partition, frames[partition], bins) for partition in PARTITIONS}
    source_hashes = {partition: sha256(state_path(partition)) for partition in PARTITIONS}
    y_max = math.ceil(max(table["record_percent"].max() for table, _ in summaries.values()) * 1.15 / 2.0) * 2.0
    manifest = current_manifest(source_hashes, y_max)
    if not args.force and is_current(manifest):
        print("Figure inputs and outputs are unchanged; skipped rendering.")
        return

    source_table = pd.concat([summaries[partition][0] for partition in PARTITIONS], ignore_index=True)
    source_table.to_csv(SOURCE_CSV, index=False, float_format="%.12g")
    source_metadata = {
        "figure_id": manifest["figure_id"],
        "status": "descriptive precheck only; not a new hypothesis test or decision rule",
        "analysis_unit": "eligible participant-event record",
        "interval_definition": manifest["definition"],
        "source_state_matrix_sha256": source_hashes,
        "display": {
            "shared_log10_seconds_bins": [LOG_BIN_MIN_SECONDS, LOG_BIN_MAX_SECONDS, LOG_BIN_COUNT],
            "reference_seconds": REFERENCE_SECONDS,
            "quantile_markers": ["P10", "P50", "P90"],
            "shared_y_max_percent": y_max,
        },
        "partitions": {partition: summaries[partition][1] for partition in PARTITIONS},
    }
    SOURCE_JSON.write_text(json.dumps(source_metadata, indent=2) + "\n", encoding="utf-8")
    for partition in PARTITIONS:
        render(partition, summaries[partition][0], summaries[partition][1], y_max)
    manifest["built_at_utc"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered interval-distribution figures in {FIGURE_DIR}.")


if __name__ == "__main__":
    main()
