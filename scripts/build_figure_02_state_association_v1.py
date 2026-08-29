#!/usr/bin/env python
"""Build the versioned primary-association forest plot for the manuscript.

The source file contains only values released in the frozen final-association
summary. Re-running retains matching exports, so a completed build can resume
without redrawing unless the source or script changes or --force is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "docs" / "稿件" / "figures" / "figure_02_state_association_v1"
SOURCE_PATH = FIGURE_DIR / "source_data.json"
BASE_PATH = FIGURE_DIR / "figure_02_state_association_v1"
MANIFEST_PATH = FIGURE_DIR / "build_manifest.json"
OUTPUTS = [
    BASE_PATH.with_suffix(".pdf"),
    BASE_PATH.with_suffix(".svg"),
    BASE_PATH.with_suffix(".png"),
    BASE_PATH.with_suffix(".tiff"),
]
RASTER_DPI = 600

COLORS = {
    "grid": "#D9D9D9",
    "zero": "#767676",
    "E-A": "#0F4D92",
    "E-B": "#42949E",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(data: dict) -> None:
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

    rows = data["estimates"]
    y_positions = list(range(len(rows) - 1, -1, -1))
    labels = [row["partition"] for row in rows]

    fig, ax = plt.subplots(figsize=(3.504, 1.97), layout="constrained")

    ax.axvline(0, color=COLORS["zero"], linestyle="--", linewidth=0.9, zorder=0)
    for y, row in zip(y_positions, rows):
        color = COLORS[row["partition"]]
        ax.plot(
            [row["ci95_low"], row["ci95_high"]],
            [y, y],
            color=color,
            linewidth=1.8,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            row["estimate"],
            y,
            marker="o",
            markersize=5.8,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )

    ax.set_xlim(-0.015, 0.080)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks(y_positions, labels)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xticks([0.00, 0.02, 0.04, 0.06, 0.08])
    ax.set_xlabel("β", labelpad=5)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, zorder=0)

    for suffix, options in {
        ".pdf": {},
        ".svg": {},
        ".png": {"dpi": RASTER_DPI},
        ".tiff": {"dpi": RASTER_DPI},
    }.items():
        fig.savefig(BASE_PATH.with_suffix(suffix), bbox_inches="tight", pad_inches=0.03, **options)
    plt.close(fig)


def current_manifest() -> dict:
    return {
        "figure_id": "figure_02_state_association_v1",
        "script_sha256": sha256(Path(__file__)),
        "source_data_sha256": sha256(SOURCE_PATH),
        "outputs": [path.name for path in OUTPUTS],
        "backend": "python-matplotlib",
    }


def is_current(manifest: dict) -> bool:
    if not MANIFEST_PATH.exists() or not all(path.exists() for path in OUTPUTS):
        return False
    previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return all(previous.get(key) == value for key, value in manifest.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Render even when inputs are unchanged.")
    args = parser.parse_args()

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing source data: {SOURCE_PATH}")
    data = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    manifest = current_manifest()
    if not args.force and is_current(manifest):
        print("Figure inputs and outputs are unchanged; skipped rendering.")
        return

    render(data)
    manifest["built_at_utc"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {BASE_PATH.name}.")


if __name__ == "__main__":
    main()
