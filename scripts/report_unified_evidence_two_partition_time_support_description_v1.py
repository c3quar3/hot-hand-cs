"""Render a descriptive time-support figure from frozen E-A/E-B state matrices.

This versioned reporter uses only strict-prior interval records and completed
fixed-model coefficients. It neither reads outcomes nor refits a model.
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
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION_ROOT = PROJECT_ROOT / "results" / "unified_evidence_two_partition_confirmation_v1" / "full"
INTERACTION_ROOT = PROJECT_ROOT / "results" / "unified_evidence_two_partition_time_interaction_v1"
RESULT_DIR = PROJECT_ROOT / "results" / "unified_evidence_two_partition_time_support_description_v1"
FIGURE_DIR = PROJECT_ROOT / "docs" / "稿件" / "figures" / "time_support_negative_slope_coverage_v1"

INPUT_LOCK_PATH = RESULT_DIR / "input_lock.json"
SUMMARY_PATH = RESULT_DIR / "checkpoint_01_summary.json"
RESULT_MANIFEST_PATH = RESULT_DIR / "manifest.json"
SOURCE_CSV = FIGURE_DIR / "source_data.csv"
SOURCE_JSON = FIGURE_DIR / "source_data.json"
FIGURE_MANIFEST_PATH = FIGURE_DIR / "build_manifest.json"
OUTPUT_BASE = FIGURE_DIR / "figure_time_support_negative_slope_coverage_v1"

PARTITIONS = ("E-A", "E-B")
SECONDS_PER_TICK = 1.0 / 64.0
REFERENCE_SECONDS = 46.45
LOG_BIN_MIN_SECONDS = SECONDS_PER_TICK
LOG_BIN_MAX_SECONDS = 1200.0
LOG_BIN_COUNT = 32
RASTER_DPI = 600
FIGURE_WIDTH_IN = 89.0 / 25.4
FIGURE_HEIGHT_IN = 108.0 / 25.4
COLORS = {"E-A": "#0F4D92", "E-B": "#42949E", "grid": "#D9D9D9", "zero": "#B44C2C", "reference": "#222222"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def state_path(partition: str) -> Path:
    return CONFIRMATION_ROOT / partition / "state_matrix.parquet"


def audit_path(partition: str) -> Path:
    return CONFIRMATION_ROOT / partition / "state_audit.json"


def output_paths() -> list[Path]:
    return [SOURCE_CSV, SOURCE_JSON, OUTPUT_BASE.with_suffix(".pdf"), OUTPUT_BASE.with_suffix(".svg"), OUTPUT_BASE.with_suffix(".png"), OUTPUT_BASE.with_suffix(".tiff")]


def expected_input_lock() -> dict:
    interaction_lock = INTERACTION_ROOT / "checkpoint_01_input_lock.json"
    interaction_report = INTERACTION_ROOT / "checkpoint_03_full_report.json"
    return {
        "artifact_id": "unified_evidence_two_partition_time_support_description_v1",
        "classification": "user_authorized_descriptive_display_of_existing_fixed_time_model",
        "scope": "read_only_d_tick_and_match_id_from_existing_state_matrices; no_outcomes_residuals_state_values_or_refit",
        "analysis_unit": "strict-prior eligible participant-event record",
        "inputs": {
            partition: {
                "state_matrix_path": str(state_path(partition)),
                "state_matrix_sha256": sha256(state_path(partition)),
                "state_audit_path": str(audit_path(partition)),
                "state_audit_sha256": sha256(audit_path(partition)),
            }
            for partition in PARTITIONS
        }
        | {
            "time_interaction_input_lock_path": str(interaction_lock),
            "time_interaction_input_lock_sha256": sha256(interaction_lock),
            "time_interaction_report_path": str(interaction_report),
            "time_interaction_report_sha256": sha256(interaction_report),
            "reporting_script_path": str(Path(__file__)),
            "reporting_script_sha256": sha256(Path(__file__)),
        },
        "definition": {
            "seconds_per_tick": SECONDS_PER_TICK,
            "reference_seconds": REFERENCE_SECONDS,
            "log_bin_support_seconds": [LOG_BIN_MIN_SECONDS, LOG_BIN_MAX_SECONDS],
            "log_bin_count": LOG_BIN_COUNT,
            "negative_slope_rule": "beta + kappa * (log2(1 + d_tick) - mu_z) < 0",
            "zero_crossing_rule": "2 ** (mu_z - beta / kappa) - 1",
        },
        "reporting_boundary": "descriptive_only_no_new_p_values_confidence_intervals_multiplicity_adjustment_or_decision_rule",
    }


def load_coefficients() -> tuple[float, dict[str, dict[str, float]]]:
    interaction_lock = read_json(INTERACTION_ROOT / "checkpoint_01_input_lock.json")
    report = read_json(INTERACTION_ROOT / "checkpoint_03_full_report.json")
    mu_z = float(interaction_lock["reference"]["mu_z"])
    coefficients: dict[str, dict[str, float]] = {}
    for partition in PARTITIONS:
        source = report["partitions"][partition]["coefficients"]
        beta = float(source["beta_state"]["estimate"])
        kappa = float(source["kappa_state_time"]["estimate"])
        if not (np.isfinite(beta) and np.isfinite(kappa) and kappa > 0):
            raise ValueError(f"{partition} has invalid fixed interaction coefficients")
        coefficients[partition] = {"beta": beta, "kappa": kappa}
    return mu_z, coefficients


def load_partition(partition: str, expected_hash: str) -> pd.DataFrame:
    audit = read_json(audit_path(partition))
    actual_hash = sha256(state_path(partition))
    if actual_hash != expected_hash or actual_hash != audit["state_matrix_sha256"]:
        raise ValueError(f"{partition} state-matrix hash does not match the locked audit")
    frame = pd.read_parquet(state_path(partition), columns=["match_id", "d_tick"])
    expected_rows = int(audit["metrics"]["analysis_rows"])
    expected_matches = int(audit["matches"])
    if len(frame) != expected_rows or frame["match_id"].nunique() != expected_matches:
        raise ValueError(f"{partition} records do not match its completed state audit")
    values = frame["d_tick"].to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{partition} has invalid strict-prior intervals")
    frame = frame.copy()
    frame["seconds"] = values * SECONDS_PER_TICK
    if frame["seconds"].min() < LOG_BIN_MIN_SECONDS or frame["seconds"].max() > LOG_BIN_MAX_SECONDS:
        raise ValueError(f"{partition} interval lies outside the fixed display support")
    return frame


def summarize_partition(partition: str, frame: pd.DataFrame, mu_z: float, coefficients: dict[str, float], bins: np.ndarray) -> tuple[pd.DataFrame, dict]:
    d_tick = frame["d_tick"].to_numpy(dtype=float)
    seconds = frame["seconds"].to_numpy(dtype=float)
    beta, kappa = coefficients["beta"], coefficients["kappa"]
    crossing_tick = float(2.0 ** (mu_z - beta / kappa) - 1.0)
    crossing_seconds = crossing_tick * SECONDS_PER_TICK
    if not (np.isfinite(crossing_seconds) and crossing_seconds > 0):
        raise ValueError(f"{partition} has an invalid zero-crossing point")
    negative = d_tick < crossing_tick
    counts, _ = np.histogram(seconds, bins=bins)
    if int(counts.sum()) != len(frame):
        raise AssertionError(f"{partition} histogram omitted records")
    table = pd.DataFrame(
        {
            "partition": partition,
            "bin_left_seconds": bins[:-1],
            "bin_right_seconds": bins[1:],
            "bin_center_geometric_seconds": np.sqrt(bins[:-1] * bins[1:]),
            "record_count": counts,
            "record_percent": counts / len(frame) * 100.0,
        }
    )
    summary = {
        "records": int(len(frame)),
        "matches": int(frame["match_id"].nunique()),
        "beta": beta,
        "kappa": kappa,
        "zero_crossing_d_tick": crossing_tick,
        "zero_crossing_seconds": crossing_seconds,
        "negative_slope_record_count": int(negative.sum()),
        "negative_slope_record_percent": float(negative.mean() * 100.0),
        "minimum_seconds": float(seconds.min()),
        "maximum_seconds": float(seconds.max()),
        "histogram_count_sum": int(counts.sum()),
    }
    return table, summary


def render(tables: dict[str, pd.DataFrame], summaries: dict[str, dict], y_max: float) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"], "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.8, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), layout="constrained")
    for index, (axis, partition) in enumerate(zip(axes, PARTITIONS)):
        table, summary = tables[partition], summaries[partition]
        left = table["bin_left_seconds"].to_numpy()
        axis.bar(left, table["record_percent"].to_numpy(), width=table["bin_right_seconds"].to_numpy() - left, align="edge", color=COLORS[partition], edgecolor="white", linewidth=0.3, zorder=2)
        axis.axvline(summary["zero_crossing_seconds"], color=COLORS["zero"], linewidth=1.0, zorder=4)
        axis.axvline(REFERENCE_SECONDS, color=COLORS["reference"], linewidth=1.0, zorder=4)
        axis.set_xscale("log")
        axis.set_xlim(LOG_BIN_MIN_SECONDS, LOG_BIN_MAX_SECONDS)
        axis.set_ylim(0, y_max)
        axis.set_yticks(np.arange(0, y_max + 0.01, 2))
        axis.set_ylabel("Records (%)", labelpad=4)
        axis.grid(axis="y", color=COLORS["grid"], linewidth=0.55, zorder=0)
        axis.spines["right"].set_visible(False)
        axis.spines["top"].set_visible(False)
        axis.tick_params(axis="both", width=0.8, length=3)
        axis.text(-0.16, 1.03, chr(ord("a") + index), transform=axis.transAxes, fontweight="bold", va="bottom")
        axis.set_title(f"{partition}   n={summary['records']:,}   zero crossing={summary['zero_crossing_seconds']:.2f} s   negative slope={summary['negative_slope_record_percent']:.1f}%", loc="left", pad=4, fontweight="bold")
    axes[-1].set_xticks([0.1, 1, 10, 100, 1000])
    axes[-1].set_xticklabels(["0.1", "1", "10", "100", "1,000"])
    axes[-1].set_xlabel("Strict-prior interval (s)", labelpad=4)
    legend = [Line2D([0], [0], color=COLORS["zero"], lw=1.0, label="Zero model-implied state slope"), Line2D([0], [0], color=COLORS["reference"], lw=1.0, label="Fixed reference (46.45 s)")]
    fig.legend(handles=legend, loc="lower center", ncol=1, frameon=False, fontsize=6.2, bbox_to_anchor=(0.5, -0.015))
    for suffix, options in {".pdf": {}, ".svg": {}, ".png": {"dpi": RASTER_DPI}, ".tiff": {"dpi": RASTER_DPI}}.items():
        fig.savefig(OUTPUT_BASE.with_suffix(suffix), bbox_inches="tight", pad_inches=0.03, **options)
    plt.close(fig)


def core_manifest(input_lock: dict, summaries: dict[str, dict], y_max: float) -> dict:
    return {"figure_id": "time_support_negative_slope_coverage_v1", "backend": "python-matplotlib", "input_lock_sha256": hashlib.sha256(json.dumps(input_lock, sort_keys=True).encode("utf-8")).hexdigest(), "shared_y_max_percent": y_max, "outputs": [path.name for path in output_paths()], "partitions": summaries}


def is_current(manifest: dict) -> bool:
    required = [INPUT_LOCK_PATH, SUMMARY_PATH, RESULT_MANIFEST_PATH, FIGURE_MANIFEST_PATH, *output_paths()]
    if not all(path.exists() for path in required):
        return False
    existing = read_json(FIGURE_MANIFEST_PATH)
    return all(existing.get(key) == value for key, value in manifest.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild even when all locked inputs and exports match.")
    args = parser.parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    input_lock = expected_input_lock()
    mu_z, coefficients = load_coefficients()
    bins = np.logspace(math.log10(LOG_BIN_MIN_SECONDS), math.log10(LOG_BIN_MAX_SECONDS), LOG_BIN_COUNT + 1)
    # Keep exact tick-derived boundary values inside the closed display support.
    bins[0] = LOG_BIN_MIN_SECONDS
    bins[-1] = LOG_BIN_MAX_SECONDS
    frames = {partition: load_partition(partition, input_lock["inputs"][partition]["state_matrix_sha256"]) for partition in PARTITIONS}
    pairs = {partition: summarize_partition(partition, frames[partition], mu_z, coefficients[partition], bins) for partition in PARTITIONS}
    tables = {partition: pairs[partition][0] for partition in PARTITIONS}
    summaries = {partition: pairs[partition][1] for partition in PARTITIONS}
    y_max = math.ceil(max(table["record_percent"].max() for table in tables.values()) * 1.15 / 2.0) * 2.0
    manifest = core_manifest(input_lock, summaries, y_max)
    if not args.force and is_current(manifest):
        print("Time-support figure inputs and outputs are unchanged; skipped rendering.")
        return
    write_json(INPUT_LOCK_PATH, input_lock)
    source_table = pd.concat([tables[partition] for partition in PARTITIONS], ignore_index=True)
    source_table.to_csv(SOURCE_CSV, index=False, float_format="%.12g")
    source_metadata = {"figure_id": manifest["figure_id"], "status": "descriptive only; no new hypothesis test, confidence interval, p value, or decision rule", "analysis_unit": input_lock["analysis_unit"], "mu_z": mu_z, "reference_seconds": REFERENCE_SECONDS, "partitions": summaries, "input_lock": input_lock}
    write_json(SOURCE_JSON, source_metadata)
    render(tables, summaries, y_max)
    summary = {"stage": "unified_evidence_two_partition_time_support_description_v1", "status": "completed_descriptive_only", "gates": {"T0_locked_inputs": "pass", "T1_interval_and_histogram_integrity": "pass", "T2_fixed_function_calculation": "pass", "T3_reporting_boundary": "assessed"}, "reference": {"mu_z": mu_z, "seconds": REFERENCE_SECONDS}, "partitions": summaries}
    write_json(SUMMARY_PATH, summary)
    manifest["built_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(FIGURE_MANIFEST_PATH, manifest)
    write_json(RESULT_MANIFEST_PATH, {"artifact_id": input_lock["artifact_id"], "status": "completed_descriptive_only", "input_lock_sha256": manifest["input_lock_sha256"], "files": {path.name: sha256(path) for path in [INPUT_LOCK_PATH, SUMMARY_PATH, SOURCE_CSV, SOURCE_JSON, *output_paths()[2:]]}})
    result_manifest = read_json(RESULT_MANIFEST_PATH)
    result_manifest["input_lock_file_sha256"] = sha256(INPUT_LOCK_PATH)
    write_json(RESULT_MANIFEST_PATH, result_manifest)
    print(f"Rendered descriptive time-support figure in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
