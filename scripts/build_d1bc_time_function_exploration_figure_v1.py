"""Render the D1-B/C exploratory flexible state-slope functions."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "d1bc_time_function_exploration_v1"
FULL_PATH = RESULT_ROOT / "checkpoint_03_full.json"
CURVE_PATH = RESULT_ROOT / "curve_source_data.csv"
FIGURE_DIR = ROOT / "docs" / "稿件" / "figures" / "d1bc_time_function_exploration_v1"
BASE_PATH = FIGURE_DIR / "figure_d1bc_time_function_exploration_v1"
SOURCE_CSV = FIGURE_DIR / "source_data.csv"
SOURCE_JSON = FIGURE_DIR / "source_data.json"
MANIFEST_PATH = FIGURE_DIR / "build_manifest.json"
WIDTH_MM = 183.0
HEIGHT_MM = 82.0
RASTER_DPI = 600
COLORS = {"D1-B": "#1766A3", "D1-C": "#C45A23", "knot": "#6E6E6E", "zero": "#363636", "grid": "#DADADA"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outputs() -> list[Path]:
    return [SOURCE_CSV, SOURCE_JSON, *(BASE_PATH.with_suffix(suffix) for suffix in (".svg", ".pdf", ".png", ".tiff"))]


def manifest(inputs: dict[str, str]) -> dict:
    return {
        "figure_id": "d1bc_time_function_exploration_v1",
        "backend": "python-matplotlib",
        "script_sha256": sha256(Path(__file__)),
        "inputs": inputs,
        "source_curve": "predeclared D1-A-knot piecewise-linear state slope; two-way CRV1 pointwise delta-method 95% intervals",
        "domain": "D1-B/C shared P1-P99 support; main panel P1-P90 and tail panel P90-P99",
        "outputs": [path.name for path in outputs()],
    }


def is_current(expected: dict) -> bool:
    if not MANIFEST_PATH.exists() or not all(path.exists() for path in outputs()):
        return False
    previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return all(previous.get(key) == value for key, value in expected.items())


def draw_panel(ax: plt.Axes, curves: pd.DataFrame, low: float, high: float, log_x: bool, ylim: tuple[float, float], title: str, knots: list[float], label: str) -> None:
    subset = curves.loc[(curves["seconds"] >= low) & (curves["seconds"] <= high)]
    for dataset in ("D1-B", "D1-C"):
        frame = subset.loc[subset["dataset"] == dataset]
        ax.fill_between(frame["seconds"].to_numpy(), frame["ci95_low"].to_numpy(), frame["ci95_high"].to_numpy(), color=COLORS[dataset], alpha=0.16, linewidth=0, zorder=1)
        ax.plot(frame["seconds"], frame["state_slope"], color=COLORS[dataset], linewidth=1.55, zorder=3)
    ax.axhline(0, color=COLORS["zero"], linewidth=0.8, zorder=0)
    for knot in knots:
        if low <= knot <= high:
            ax.axvline(knot, color=COLORS["knot"], linestyle="--", linewidth=0.75, zorder=2)
    if log_x:
        ax.set_xscale("log")
        ax.set_xticks([0.25, 1, 10, 100])
        ax.set_xticklabels(["0.25", "1", "10", "100"])
    ax.set_xlim(low, high)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=0.8, length=3)
    ax.set_title(title, loc="left", fontweight="bold", pad=3)
    ax.text(-0.13, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="bottom")
    ax.set_xlabel("Interval since prior eligible event (s)", labelpad=4)


def render(curves: pd.DataFrame, full: dict) -> None:
    reference = full["reference"]
    support = full["common_curve_support"]
    knots = reference["knots_seconds"]
    core_high = knots[2]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"], "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.8, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), gridspec_kw={"width_ratios": [1.52, 1.0]})
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.19, top=0.72, wspace=0.30)
    draw_panel(axes[0], curves, support["low_seconds"], core_high, True, (-0.13, 0.16), "Core support (P1-P90)", knots, "a")
    draw_panel(axes[1], curves, core_high, support["high_seconds"], False, (-4.0, 4.0), "Upper tail (P90-P99)", knots, "b")
    axes[0].set_ylabel("State slope\n(residual units per D1-A reference SD)", labelpad=5)
    axes[1].set_yticks([-4, -2, 0, 2, 4])
    handles = [Line2D([0], [0], color=COLORS["D1-B"], linewidth=1.55, label="D1-B"), Line2D([0], [0], color=COLORS["D1-C"], linewidth=1.55, label="D1-C"), Patch(facecolor="#808080", alpha=0.16, edgecolor="none", label="Pointwise 95% CI"), Line2D([0], [0], color=COLORS["knot"], linestyle="--", linewidth=0.75, label="D1-A P10/P50/P90 knots")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False, fontsize=6, handlelength=1.8, columnspacing=1.3)
    for suffix, options in {".svg": {}, ".pdf": {}, ".png": {"dpi": RASTER_DPI}, ".tiff": {"dpi": RASTER_DPI}}.items():
        fig.savefig(BASE_PATH.with_suffix(suffix), bbox_inches="tight", pad_inches=0.03, **options)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not FULL_PATH.exists() or not CURVE_PATH.exists():
        raise FileNotFoundError("Missing completed D1-B/C exploratory analysis outputs")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    inputs = {"full_checkpoint_sha256": sha256(FULL_PATH), "curve_source_sha256": sha256(CURVE_PATH)}
    expected = manifest(inputs)
    if not args.force and is_current(expected):
        print("Figure inputs and outputs are unchanged; skipped rendering.")
        return
    full = json.loads(FULL_PATH.read_text(encoding="utf-8"))
    curves = pd.read_csv(CURVE_PATH)
    required = {"dataset", "seconds", "state_slope", "ci95_low", "ci95_high"}
    if not required.issubset(curves.columns) or set(curves["dataset"]) != {"D1-B", "D1-C"}:
        raise ValueError("Unexpected exploratory curve source schema")
    if not np.isfinite(curves[["seconds", "state_slope", "ci95_low", "ci95_high"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite curve values")
    curves.to_csv(SOURCE_CSV, index=False, float_format="%.12g")
    SOURCE_JSON.write_text(json.dumps({"figure_id": expected["figure_id"], "status": "exploratory D1-B/C display; pointwise intervals and Wald quantities do not modify the frozen primary model or confirmation adjudication", "inputs": inputs, "reference_knots_seconds": full["reference"]["knots_seconds"], "common_curve_support": full["common_curve_support"]}, indent=2) + "\n", encoding="utf-8")
    render(curves, full)
    expected["built_at_utc"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered exploratory time-function figure in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
