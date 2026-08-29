#!/usr/bin/env python
"""Build the versioned E-A and E-B residual-loss interval plots.

The script reads only the frozen fixed-model comparison release. It verifies
the released comparison file against its SHA-256 ledger, writes per-figure
source-data CSV files, and skips an unchanged completed build unless forced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "unified_evidence_two_partition_loss_comparison_v2"
COMPARISON_PATH = RESULT_DIR / "comparison.json"
HASH_LEDGER_PATH = RESULT_DIR / "file_sha256.json"
FIGURE_ROOT = PROJECT_ROOT / "docs" / "稿件" / "figures"
RASTER_DPI = 600
X_LIMITS = (0.0, 0.46)
X_TICKS = (0.0, 0.1, 0.2, 0.3, 0.4)

COMPARISONS = (
    (
        "B1_vs_M1",
        "Time -> Lag-1",
        "#767676",
    ),
    (
        "B1_vs_M3",
        "Time -> 60-s",
        "#3775BA",
    ),
    (
        "M1_vs_M3",
        "Lag-1 -> 60-s",
        "#0F4D92",
    ),
)

FIGURES = {
    "E-A": {
        "figure_id": "figure_04_ea_loss_reduction_v1",
        "records": 62846,
        "label": "E-A",
    },
    "E-B": {
        "figure_id": "figure_05_eb_loss_reduction_v1",
        "records": 62386,
        "label": "E-B",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_comparisons() -> dict:
    if not COMPARISON_PATH.exists() or not HASH_LEDGER_PATH.exists():
        raise FileNotFoundError("The frozen fixed-model comparison release is incomplete.")

    ledger = json.loads(HASH_LEDGER_PATH.read_text(encoding="utf-8"))
    expected_hash = ledger.get(COMPARISON_PATH.name)
    actual_hash = sha256(COMPARISON_PATH)
    if expected_hash != actual_hash:
        raise RuntimeError("comparison.json does not match its frozen SHA-256 ledger.")

    released = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    expected_partitions = set(FIGURES)
    if set(released.get("comparisons", {})) != expected_partitions:
        raise ValueError("The frozen comparison release has an unexpected partition set.")

    rows_by_partition: dict[str, list[dict]] = {}
    for partition in FIGURES:
        comparison_block = released["comparisons"][partition]
        if set(comparison_block) != {key for key, _, _ in COMPARISONS}:
            raise ValueError(f"The frozen comparison release has an unexpected comparison set for {partition}.")

        rows: list[dict] = []
        for key, label, color in COMPARISONS:
            record = comparison_block[key]
            estimate = 100.0 * float(record["relative_gain"])
            ci_low = 100.0 * float(record["ci95_low"])
            ci_high = 100.0 * float(record["ci95_high"])
            if not (ci_low <= estimate <= ci_high):
                raise ValueError(f"The released interval does not contain its estimate for {partition} {key}.")
            rows.append(
                {
                    "partition": partition,
                    "comparison_id": key,
                    "comparison": label,
                    "baseline": record["baseline"],
                    "candidate": record["candidate"],
                    "relative_loss_reduction_percent": estimate,
                    "ci95_low_percent": ci_low,
                    "ci95_high_percent": ci_high,
                    "holm_p_value": float(record["p_value_holm"]),
                    "color": color,
                }
            )
        rows_by_partition[partition] = rows

    return {
        "rows_by_partition": rows_by_partition,
        "comparison_sha256": actual_hash,
        "input_lock_sha256": released["input_lock_sha256"],
        "inference": released["inference"],
    }


def paths_for(partition: str) -> dict[str, Path]:
    spec = FIGURES[partition]
    figure_dir = FIGURE_ROOT / spec["figure_id"]
    base = figure_dir / spec["figure_id"]
    return {
        "directory": figure_dir,
        "base": base,
        "source_csv": figure_dir / "source_data.csv",
        "source_json": figure_dir / "source_data.json",
        "manifest": figure_dir / "build_manifest.json",
        "pdf": base.with_suffix(".pdf"),
        "svg": base.with_suffix(".svg"),
        "png": base.with_suffix(".png"),
        "tiff": base.with_suffix(".tiff"),
    }


def write_source_data(partition: str, rows: list[dict], provenance: dict) -> dict[str, Path]:
    paths = paths_for(partition)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    columns = (
        "partition",
        "comparison_id",
        "comparison",
        "baseline",
        "candidate",
        "relative_loss_reduction_percent",
        "ci95_low_percent",
        "ci95_high_percent",
        "holm_p_value",
    )
    with paths["source_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})

    source_json = {
        "figure_id": FIGURES[partition]["figure_id"],
        "partition": partition,
        "comparison_records": FIGURES[partition]["records"],
        "source_result": "results/unified_evidence_two_partition_loss_comparison_v2/comparison.json",
        "source_comparison_sha256": provenance["comparison_sha256"],
        "input_lock_sha256": provenance["input_lock_sha256"],
        "inference": provenance["inference"],
        "rows": [{key: value for key, value in row.items() if key != "color"} for row in rows],
    }
    paths["source_json"].write_text(json.dumps(source_json, indent=2) + "\n", encoding="utf-8")
    return paths


def render(partition: str, rows: list[dict], paths: dict[str, Path]) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(3.504, 2.18), layout="constrained")
    y_positions = list(range(len(rows) - 1, -1, -1))

    for y, row in zip(y_positions, rows):
        ax.plot(
            [row["ci95_low_percent"], row["ci95_high_percent"]],
            [y, y],
            color=row["color"],
            linewidth=1.8,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            row["relative_loss_reduction_percent"],
            y,
            marker="o",
            markersize=5.8,
            markerfacecolor=row["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
        label = (
            f"{row['relative_loss_reduction_percent']:.3f}% "
            f"[{row['ci95_low_percent']:.3f}, {row['ci95_high_percent']:.3f}]"
        )
        ax.text(
            0.325,
            y,
            label,
            ha="left",
            va="center",
            fontsize=6.5,
            color="#272727",
        )

    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(-0.65, 2.55)
    ax.set_yticks(y_positions, [row["comparison"] for row in rows])
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xticks(X_TICKS, [f"{tick:.1f}%" for tick in X_TICKS])
    ax.set_xlabel("Relative reduction in match-equal residual squared loss", labelpad=5)
    ax.text(
        0.0,
        2.34,
        FIGURES[partition]["label"],
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#272727",
    )

    for suffix, options in {
        ".pdf": {},
        ".svg": {},
        ".png": {"dpi": RASTER_DPI},
        ".tiff": {"dpi": RASTER_DPI},
    }.items():
        fig.savefig(paths["base"].with_suffix(suffix), bbox_inches="tight", pad_inches=0.03, **options)
    plt.close(fig)


def manifest_for(partition: str, paths: dict[str, Path], provenance: dict) -> dict:
    return {
        "figure_id": FIGURES[partition]["figure_id"],
        "partition": partition,
        "backend": "python-matplotlib",
        "script_sha256": sha256(Path(__file__)),
        "source_comparison_sha256": provenance["comparison_sha256"],
        "source_data_csv_sha256": sha256(paths["source_csv"]),
        "source_data_json_sha256": sha256(paths["source_json"]),
        "outputs": [paths[key].name for key in ("pdf", "svg", "png", "tiff")],
    }


def is_current(paths: dict[str, Path], current: dict) -> bool:
    output_keys = ("pdf", "svg", "png", "tiff")
    if not paths["manifest"].exists() or not all(paths[key].exists() for key in output_keys):
        return False
    previous = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if any(previous.get(key) != value for key, value in current.items()):
        return False
    return previous.get("output_sha256") == {key: sha256(paths[key]) for key in output_keys}


def build(partition: str, rows: list[dict], provenance: dict, force: bool, dry_run: bool) -> str:
    paths = write_source_data(partition, rows, provenance)
    manifest = manifest_for(partition, paths, provenance)
    if not force and is_current(paths, manifest):
        return f"{partition}: inputs and outputs are unchanged; skipped rendering."
    if dry_run:
        return f"{partition}: frozen input and source-data validation passed."

    render(partition, rows, paths)
    output_keys = ("pdf", "svg", "png", "tiff")
    manifest["output_sha256"] = {key: sha256(paths[key]) for key in output_keys}
    manifest["built_at_utc"] = datetime.now(timezone.utc).isoformat()
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return f"{partition}: rendered {FIGURES[partition]['figure_id']}."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("E-A", "E-B", "all"), default="all")
    parser.add_argument("--force", action="store_true", help="Render even when the completed build is unchanged.")
    parser.add_argument("--dry-run", action="store_true", help="Validate frozen input and write source data without rendering.")
    args = parser.parse_args()

    provenance = load_frozen_comparisons()
    partitions = tuple(FIGURES) if args.partition == "all" else (args.partition,)
    for partition in partitions:
        print(build(partition, provenance["rows_by_partition"][partition], provenance, args.force, args.dry_run))


if __name__ == "__main__":
    main()
