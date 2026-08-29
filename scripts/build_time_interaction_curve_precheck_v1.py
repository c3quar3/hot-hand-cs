"""Build descriptive state-slope versus time curves from the fixed interaction model.

This script reads the released full interaction checkpoint and frozen state
matrices. It does not refit a model, access outcomes, calculate p values, or
create a new decision rule. Curves are the fixed-model implication
beta + kappa * z(d), with delta-method two-way CRV1 confidence bands.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "results" / "unified_evidence_two_partition_time_interaction_v1" / "checkpoint_03_full_report.json"
STATE_ROOT = PROJECT_ROOT / "results" / "unified_evidence_two_partition_confirmation_v1" / "full"
FIGURE_DIR = PROJECT_ROOT / "docs" / "稿件" / "figures" / "time_interaction_curve_precheck_v1"
SOURCE_CSV = FIGURE_DIR / "source_data.csv"
SOURCE_JSON = FIGURE_DIR / "source_data.json"
MANIFEST_PATH = FIGURE_DIR / "build_manifest.json"
BASE_PATH = FIGURE_DIR / "figure_time_interaction_curve_precheck_v1"

PARTITIONS = ("E-A", "E-B")
MU_Z = 11.538069409706317
TICKS_PER_SECOND = 64.0
REFERENCE_SECONDS = 46.45
NORMAL_95 = 1.959963984540054
CURVE_POINTS = 401
RASTER_DPI = 600
WIDTH_MM = 183.0
HEIGHT_MM = 78.0

COLORS = {
    "E-A": "#0F4D92",
    "E-B": "#42949E",
    "interval": "#767676",
    "reference": "#222222",
    "zero": "#767676",
    "grid": "#D9D9D9",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_path(partition: str) -> Path:
    return STATE_ROOT / partition / "state_matrix.parquet"


def output_paths() -> list[Path]:
    return [
        SOURCE_CSV,
        SOURCE_JSON,
        BASE_PATH.with_suffix(".pdf"),
        BASE_PATH.with_suffix(".svg"),
        BASE_PATH.with_suffix(".png"),
        BASE_PATH.with_suffix(".tiff"),
    ]


def load_support(partition: str) -> dict[str, float]:
    path = state_path(partition)
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen state matrix: {path}")
    ticks = pd.read_parquet(path, columns=["d_tick"])["d_tick"].astype(float).to_numpy()
    if len(ticks) == 0 or not np.isfinite(ticks).all() or (ticks <= 0).any():
        raise ValueError(f"{partition} has invalid strict-prior interval values")
    # d_tick is strictly positive before the log-time transformation below.
    seconds = ticks / TICKS_PER_SECOND
    return {
        "records": int(len(seconds)),
        "p01_seconds": float(np.quantile(seconds, 0.01)),
        "p10_seconds": float(np.quantile(seconds, 0.10)),
        "p50_seconds": float(np.quantile(seconds, 0.50)),
        "p90_seconds": float(np.quantile(seconds, 0.90)),
        "p99_seconds": float(np.quantile(seconds, 0.99)),
    }


def derive_curve(partition: str, checkpoint: dict, support: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    result = checkpoint["partitions"][partition]
    beta = float(result["coefficients"]["beta_state"]["estimate"])
    kappa = float(result["coefficients"]["kappa_state_time"]["estimate"])
    covariance = np.asarray(result["covariance_matrix"], dtype=float)
    variance_beta = float(covariance[2, 2])
    covariance_beta_kappa = float(covariance[2, 3])
    variance_kappa = float(covariance[3, 3])
    seconds = np.logspace(math.log10(support["p01_seconds"]), math.log10(support["p99_seconds"]), CURVE_POINTS)
    z = np.log2(1.0 + TICKS_PER_SECOND * seconds) - MU_Z
    slope = beta + kappa * z
    variance = variance_beta + 2.0 * z * covariance_beta_kappa + np.square(z) * variance_kappa
    if (variance < 0).any():
        raise ValueError(f"{partition} produced a negative delta-method variance")
    standard_error = np.sqrt(variance)
    curve = pd.DataFrame(
        {
            "partition": partition,
            "seconds": seconds,
            "z": z,
            "state_slope": slope,
            "standard_error": standard_error,
            "ci95_low": slope - NORMAL_95 * standard_error,
            "ci95_high": slope + NORMAL_95 * standard_error,
        }
    )
    metadata = {
        "beta": beta,
        "kappa": kappa,
        "var_beta": variance_beta,
        "cov_beta_kappa": covariance_beta_kappa,
        "var_kappa": variance_kappa,
        "support": support,
    }
    return curve, metadata


def y_limits(curves: dict[str, pd.DataFrame]) -> tuple[float, float]:
    lower = min(curve["ci95_low"].min() for curve in curves.values())
    upper = max(curve["ci95_high"].max() for curve in curves.values())
    step = 0.02
    return math.floor((lower - 0.005) / step) * step, math.ceil((upper + 0.005) / step) * step


def render(curves: dict[str, pd.DataFrame], metadata: dict[str, dict], limits: tuple[float, float]) -> None:
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
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.19, top=0.75, wspace=0.12)
    for ax, partition in zip(axes, PARTITIONS):
        curve = curves[partition]
        support = metadata[partition]["support"]
        color = COLORS[partition]
        ax.fill_between(
            curve["seconds"].to_numpy(),
            curve["ci95_low"].to_numpy(),
            curve["ci95_high"].to_numpy(),
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
        ax.plot(curve["seconds"], curve["state_slope"], color=color, linewidth=1.8, zorder=3)
        ax.axhline(0, color=COLORS["zero"], linewidth=0.8, zorder=0)
        for marker in (support["p10_seconds"], support["p50_seconds"], support["p90_seconds"]):
            ax.axvline(marker, color=COLORS["interval"], linestyle="--", linewidth=0.8, zorder=2)
        ax.axvline(REFERENCE_SECONDS, color=COLORS["reference"], linewidth=1.0, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(0.1, 600)
        ax.set_ylim(*limits)
        ax.set_xticks([0.1, 1, 10, 100])
        ax.set_xticklabels(["0.1", "1", "10", "100"])
        ax.set_yticks(np.arange(limits[0], limits[1] + 0.001, 0.04))
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, zorder=0)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="both", width=0.8, length=3)
        ax.set_title(partition, loc="left", fontweight="bold", pad=3)
        ax.set_xlabel("Interval since prior eligible event (s)", labelpad=4)
    axes[0].set_ylabel("Model-implied state slope\n(residual units per D1-A reference SD)", labelpad=5)
    legend_handles = [
        Line2D([0], [0], color=COLORS["E-A"], linewidth=1.8, label="Model-implied slope"),
        Patch(facecolor=COLORS["E-A"], alpha=0.18, edgecolor="none", label="Delta-method 95% CI"),
        Line2D([0], [0], color=COLORS["interval"], linestyle="--", linewidth=0.8, label="P10/P50/P90"),
        Line2D([0], [0], color=COLORS["reference"], linewidth=1.0, label="D1-A reference (46.45 s)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=6,
        handlelength=1.8,
        columnspacing=1.4,
    )
    for suffix, options in {
        ".pdf": {},
        ".svg": {},
        ".png": {"dpi": RASTER_DPI},
        ".tiff": {"dpi": RASTER_DPI},
    }.items():
        fig.savefig(BASE_PATH.with_suffix(suffix), bbox_inches="tight", pad_inches=0.03, **options)
    plt.close(fig)


def current_manifest(source_hashes: dict[str, str], limits: tuple[float, float]) -> dict:
    return {
        "figure_id": "time_interaction_curve_precheck_v1",
        "backend": "python-matplotlib",
        "script_sha256": sha256(Path(__file__)),
        "time_interaction_checkpoint_sha256": source_hashes["checkpoint"],
        "state_matrix_sha256": {partition: source_hashes[partition] for partition in PARTITIONS},
        "curve": "beta + kappa * (log2(1 + 64 * seconds) - mu_z)",
        "mu_z": MU_Z,
        "domain": "within-partition P1 to P99 of observed strict-prior intervals",
        "confidence_band": "delta-method two-way CRV1 normal 95% interval",
        "reference_seconds": REFERENCE_SECONDS,
        "shared_y_limits": list(limits),
        "curve_points_per_partition": CURVE_POINTS,
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

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing fixed interaction checkpoint: {CHECKPOINT_PATH}")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    supports = {partition: load_support(partition) for partition in PARTITIONS}
    derived = {partition: derive_curve(partition, checkpoint, supports[partition]) for partition in PARTITIONS}
    curves = {partition: derived[partition][0] for partition in PARTITIONS}
    metadata = {partition: derived[partition][1] for partition in PARTITIONS}
    limits = y_limits(curves)
    source_hashes = {"checkpoint": sha256(CHECKPOINT_PATH)}
    source_hashes.update({partition: sha256(state_path(partition)) for partition in PARTITIONS})
    manifest = current_manifest(source_hashes, limits)
    if not args.force and is_current(manifest):
        print("Figure inputs and outputs are unchanged; skipped rendering.")
        return

    pd.concat([curves[partition] for partition in PARTITIONS], ignore_index=True).to_csv(
        SOURCE_CSV, index=False, float_format="%.12g"
    )
    source_metadata = {
        "figure_id": manifest["figure_id"],
        "status": "descriptive fixed-model implication only; not a new hypothesis test, subgroup analysis, or decision rule",
        "curve": manifest["curve"],
        "mu_z": MU_Z,
        "confidence_band": manifest["confidence_band"],
        "source_sha256": source_hashes,
        "partitions": metadata,
    }
    SOURCE_JSON.write_text(json.dumps(source_metadata, indent=2) + "\n", encoding="utf-8")
    render(curves, metadata, limits)
    manifest["built_at_utc"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered time-interaction curve figure in {FIGURE_DIR}.")


if __name__ == "__main__":
    main()
