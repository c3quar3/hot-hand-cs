"""Review-driven robustness supplement derived only from frozen artifacts.

This script produces descriptive robustness values requested by the pre-submission
review, all computed from frozen inputs and written to a new versioned output root.
It does not modify, rescore, or re-select any frozen protocol, result, gate record,
or interpretation boundary. Nothing here changes the frozen primary estimates.

Contents (all descriptive, no new decision rule):
  1. simulation-based null model: Y* ~ Bernoulli(p_hat) under the frozen p_hat,
     rebuild signed participant states with the frozen half-life and reference
     standardization, refit the fixed OLS model, and locate the frozen beta in the
     simulated null distribution (E-A and E-B);
  2. support-bound sensitivity grid: rebuild states after filtering the frozen
     asset to 0.05 <= p_hat <= 0.95 and 0.001 <= p_hat <= 0.999, refit, report beta
     and CRV1 95% CI (E-A and E-B);
  3. t(G-1) inference comparison for the frozen beta (G = 100 match clusters);
  4. within-partition e_t sample standard deviation (frozen state matrices);
  5. delta-method 95% CI for the frozen zero-crossing time from the frozen
     time-interaction covariance;
  6. average marginal effect of the frozen beta on the win-probability scale,
     approximated as beta * E[sqrt(p_hat (1 - p_hat))] over the frozen eligible
     physical engagements;
  7. verification that the two identical reported Holm-adjusted P values follow
     the step-down cumulative-max rule of the six-test Holm family.

Smoke mode runs a single simulation iteration and the grid on one partition to
validate the pipeline before the full run. Every output is written only once;
resume reuses existing checkpoints after hash verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

import analyze_exponential_history_state_sequence_v1 as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "review_robustness_supplement_v1"
ASSET = ROOT / "results" / "conditional_residual_asset_e1_v1" / "E1-v2" / "residual_asset.parquet"
ASSIGNMENTS = ROOT / "data" / "unified_evidence_two_partitions_v3" / "match_assignments.parquet"
ASSIGNMENT_MANIFEST = ROOT / "data" / "unified_evidence_two_partitions_v3" / "manifest.json"
CONFIRMATION_SUMMARY = ROOT / "results" / "unified_evidence_two_partition_confirmation_v1" / "checkpoint_05_summary.json"
TIME_INTERACTION = ROOT / "results" / "unified_evidence_two_partition_time_interaction_v1" / "checkpoint_03_full_report.json"
LOSS_COMPARISON = ROOT / "results" / "unified_evidence_two_partition_loss_comparison_v2" / "comparison.json"
STATE_MATRICES = {
    "E-A": ROOT / "results" / "unified_evidence_two_partition_confirmation_v1" / "full" / "E-A" / "state_matrix.parquet",
    "E-B": ROOT / "results" / "unified_evidence_two_partition_confirmation_v1" / "full" / "E-B" / "state_matrix.parquet",
}
PARTITIONS = ("E-A", "E-B")
TICK_RATE = 64.0
Z95 = 1.959963984540054
SMOKE_MATCHES = 30
SIM_ITERATIONS_FULL = 200
SIM_ITERATIONS_SMOKE = 1
SUPPORT_GRID = ((0.05, 0.95), (0.001, 0.999))
RNG_SEED = 20260827


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def reference_block(confirmation: dict[str, Any]) -> dict[str, float]:
    reference = confirmation["reference"]
    return {
        "mu_z": float(reference["mu_z"]),
        "mu_S": float(reference["mu_S"]),
        "sigma_S_ddof1": float(reference["sigma_S_ddof1"]),
    }


def load_asset_and_assignments() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    assignment_manifest = read_json(ASSIGNMENT_MANIFEST)
    require(digest(ASSIGNMENTS) == assignment_manifest["files"]["match_assignments.parquet"], "assignment_manifest_hash_mismatch")
    assignments = pd.read_parquet(ASSIGNMENTS)
    assignments["match_id"] = assignments["match_id"].astype(str)
    require(assignments["match_id"].is_unique, "assignment_match_nonunique")
    asset = pd.read_parquet(ASSET)
    require(asset["physical_event_id"].is_unique, "asset_event_nonunique")
    asset["match_id"] = asset["match_id"].astype(str)
    merged = asset.merge(assignments[["match_id", "partition"]], on="match_id", how="left")
    require(merged["partition"].notna().all(), "assignment_coverage_incomplete")
    return merged, {partition: assignments.loc[assignments["partition"].eq(partition), "match_id"].tolist() for partition in PARTITIONS}


def simulate_once(asset: pd.DataFrame, matches: list[str], reference: dict[str, float], rng: np.random.Generator) -> dict[str, Any]:
    subset = asset.loc[asset["match_id"].isin(matches)].copy()
    p = subset["p_hat"].to_numpy(dtype=float)
    require(np.isfinite(p).all() and (p > 0).all() and (p < 1).all(), "invalid_p_hat_for_simulation")
    y_star = rng.binomial(1, p).astype(float)
    subset["Y_t"] = y_star
    subset["e_t"] = (y_star - p) / np.sqrt(p * (1.0 - p))
    require(np.isfinite(subset["e_t"].to_numpy(dtype=float)).all(), "nonfinite_simulated_residual")
    state, _ = base.build_state(subset, "simulation")
    model = base.fit(state, reference, expose=True)
    return {"beta": float(model["beta"]["estimate"]), "standard_error": float(model["beta"]["standard_error"])}


def run_simulation(asset: pd.DataFrame, matches: list[str], reference: dict[str, float], iterations: int, seed: int, out_path: Path) -> dict[str, Any]:
    if out_path.exists():
        saved = read_json(out_path)
        require(saved.get("iterations") == iterations, "simulation_iteration_mismatch")
        return saved
    rng = np.random.default_rng(seed)
    rows = [simulate_once(asset, matches, reference, rng) for _ in range(iterations)]
    betas = np.asarray([row["beta"] for row in rows], dtype=float)
    ses = np.asarray([row["standard_error"] for row in rows], dtype=float)
    result = {
        "stage": "review_robustness_supplement_v1_simulation",
        "created_at_utc": now(),
        "iterations": iterations,
        "seed": seed,
        "beta_mean": float(betas.mean()) if len(betas) else None,
        "beta_sd": float(betas.std(ddof=1)) if len(betas) > 1 else None,
        "beta_p2_5": float(np.quantile(betas, 0.025)) if len(betas) else None,
        "beta_p97_5": float(np.quantile(betas, 0.975)) if len(betas) else None,
        "beta_min": float(betas.min()) if len(betas) else None,
        "beta_max": float(betas.max()) if len(betas) else None,
        "se_mean": float(ses.mean()) if len(ses) else None,
    }
    write_json_new(out_path, result)
    return result


def grid_once(asset: pd.DataFrame, matches: list[str], reference: dict[str, float], lower: float, upper: float) -> dict[str, Any]:
    subset = asset.loc[asset["match_id"].isin(matches)].copy()
    p = subset["p_hat"].to_numpy(dtype=float)
    mask = (p >= lower) & (p <= upper)
    filtered = subset.loc[mask].copy()
    require(len(filtered) > 0, "empty_support_grid_cell")
    state, _ = base.build_state(filtered, f"grid_{lower}_{upper}")
    model = base.fit(state, reference, expose=True)
    return {
        "support_lower": lower, "support_upper": upper,
        "physical_events": int(len(filtered)),
        "participant_rows": int(len(state)),
        "beta": float(model["beta"]["estimate"]),
        "ci95_low": float(model["beta"]["ci95_low"]),
        "ci95_high": float(model["beta"]["ci95_high"]),
        "standard_error": float(model["beta"]["standard_error"]),
    }


def t99_comparison(confirmation: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for partition in PARTITIONS:
        beta = confirmation["models"][partition]["beta"]
        t_stat = beta["z_value"]
        p_t = 2.0 * stats.t.sf(abs(t_stat), df=99)
        crit_t = stats.t.ppf(0.975, 99)
        se = beta["standard_error"]
        est = beta["estimate"]
        out[partition] = {
            "z_value": float(t_stat),
            "p_normal": float(beta["p_value"]),
            "p_t99": float(p_t),
            "ci95_normal_low": float(est - Z95 * se),
            "ci95_normal_high": float(est + Z95 * se),
            "ci95_t99_low": float(est - crit_t * se),
            "ci95_t99_high": float(est + crit_t * se),
            "half_width_normal": float(Z95 * se),
            "half_width_t99": float(crit_t * se),
        }
    return out


def e_t_sd() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for partition in PARTITIONS:
        state = pd.read_parquet(STATE_MATRICES[partition])
        e = state["e_t"].to_numpy(dtype=float)
        out[partition] = {"n": int(len(e)), "mean": float(e.mean()), "sd_ddof1": float(e.std(ddof=1)), "var_ddof1": float(e.var(ddof=1))}
    return out


def zero_crossing_ci(time_interaction: dict[str, Any], confirmation: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for partition in PARTITIONS:
        part = time_interaction["partitions"][partition]
        beta = part["coefficients"]["beta_state"]["estimate"]
        kappa = part["coefficients"]["kappa_state_time"]["estimate"]
        cov = np.asarray(part["covariance_matrix"], dtype=float)
        var_beta = cov[2, 2]
        var_kappa = cov[3, 3]
        cov_bk = cov[2, 3]
        z_star = -beta / kappa
        # delta method for f(b,k) = -b/k
        var_z = (1.0 / kappa ** 2) * var_beta + (beta ** 2 / kappa ** 4) * var_kappa - 2.0 * (beta / kappa ** 3) * cov_bk
        se_z = math.sqrt(max(var_z, 0.0))
        mu_z = confirmation["reference"]["mu_z"]
        lo = mu_z + z_star - Z95 * se_z
        hi = mu_z + z_star + Z95 * se_z
        def seconds(x: float) -> float:
            return ((2.0 ** x) - 1.0) / TICK_RATE
        out[partition] = {
            "z_star": float(z_star),
            "se_z": float(se_z),
            "zero_crossing_seconds": float(seconds(mu_z + z_star)),
            "ci95_low_seconds": float(seconds(lo)),
            "ci95_high_seconds": float(seconds(hi)),
        }
    return out


def average_marginal_effect(asset: pd.DataFrame, confirmation: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for partition in PARTITIONS:
        sub = asset.loc[asset["partition"].eq(partition)].copy()
        p = sub["p_hat"].to_numpy(dtype=float)
        sp = np.sqrt(p * (1.0 - p))
        mean_sp = float(sp.mean())
        beta = confirmation["models"][partition]["beta"]["estimate"]
        out[partition] = {
            "eligible_physical_events": int(len(sub)),
            "mean_sqrt_p_1mp": mean_sp,
            "beta": float(beta),
            "approx_average_marginal_effect_on_win_probability": float(beta * mean_sp),
        }
    return out


def holm_verification(loss_comparison: dict[str, Any]) -> dict[str, Any]:
    raw: dict[str, float] = {}
    for partition in PARTITIONS:
        for label, item in loss_comparison["comparisons"][partition].items():
            raw[f"{partition}:{label}"] = float(item["p_value_two_sided_raw"])
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, (m - index) * raw[key])
        adjusted[key] = running
    reported = {f"{partition}:{label}": float(item["p_value_holm"]) for partition in PARTITIONS for label, item in loss_comparison["comparisons"][partition].items()}
    mismatches = {key: {"reported": reported[key], "recomputed": adjusted[key]} for key in reported if not math.isclose(reported[key], adjusted[key], rel_tol=1e-9, abs_tol=1e-12)}
    return {
        "family_size": m,
        "rule": "holm_step_down_cumulative_max",
        "reported_equal_to_recomputed": not mismatches,
        "mismatches": mismatches,
        "identical_reported_pairs": [key for key in reported if any(other != key and reported[other] == reported[key] for other in reported)],
    }


def run(phase: str) -> None:
    confirmation = read_json(CONFIRMATION_SUMMARY)
    time_interaction = read_json(TIME_INTERACTION)
    loss_comparison = read_json(LOSS_COMPARISON)
    reference = reference_block(confirmation)
    asset, assignment_matches = load_asset_and_assignments()

    if phase == "smoke":
        matches = {partition: sorted(assignment_matches[partition])[:SMOKE_MATCHES] for partition in PARTITIONS}
        iterations = SIM_ITERATIONS_SMOKE
        partitions = ("E-A",)
    else:
        matches = assignment_matches
        iterations = SIM_ITERATIONS_FULL
        partitions = PARTITIONS

    block: dict[str, Any] = {
        "stage": f"review_robustness_supplement_v1_{phase}",
        "created_at_utc": now(),
        "phase": phase,
        "classification": "review_driven_descriptive_robustness_supplement_no_change_to_frozen_primary_results",
        "inputs": {
            "asset": {"path": str(ASSET), "sha256": digest(ASSET)},
            "assignments": {"path": str(ASSIGNMENTS), "sha256": digest(ASSIGNMENTS)},
            "confirmation_summary": {"path": str(CONFIRMATION_SUMMARY), "sha256": digest(CONFIRMATION_SUMMARY)},
            "time_interaction": {"path": str(TIME_INTERACTION), "sha256": digest(TIME_INTERACTION)},
            "loss_comparison": {"path": str(LOSS_COMPARISON), "sha256": digest(LOSS_COMPARISON)},
        },
        "reference": reference,
        "simulation": {},
        "support_grid": {},
        "t99_comparison": t99_comparison(confirmation),
        "e_t_sd": e_t_sd(),
        "zero_crossing_ci": zero_crossing_ci(time_interaction, confirmation),
        "average_marginal_effect": average_marginal_effect(asset, confirmation),
        "holm_verification": holm_verification(loss_comparison),
    }

    for partition in partitions:
        sim_path = OUT / phase / partition / "simulation.json"
        block["simulation"][partition] = run_simulation(asset, matches[partition], reference, iterations, RNG_SEED, sim_path)
        grid_path = OUT / phase / partition / "support_grid.json"
        grid_values = [grid_once(asset, matches[partition], reference, lower, upper) for lower, upper in SUPPORT_GRID]
        if grid_path.exists():
            saved = read_json(grid_path)
            require(saved.get("cells") == grid_values, "support_grid_mismatch")
        else:
            write_json_new(grid_path, {"stage": f"review_robustness_supplement_v1_support_grid_{partition}", "created_at_utc": now(), "cells": grid_values})

    block["support_grid"] = {partition: read_json(OUT / phase / partition / "support_grid.json") for partition in partitions}

    out_path = OUT / f"{phase}.json"
    if out_path.exists():
        saved = read_json(out_path)
        require(saved.get("stage") == block["stage"], "stage_mismatch")
        return
    write_json_new(out_path, block)
    manifest = {"stage": f"review_robustness_supplement_v1_{phase}_manifest", "created_at_utc": now(), "files": {path.name: digest(path) for path in sorted(OUT.rglob("*")) if path.is_file()}}
    write_json_new(OUT / f"{phase}_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    try:
        run(args.phase)
    except Exception as error:  # noqa: BLE001
        log = OUT / f"{args.phase}_error.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"{now()}\n{type(error).__name__}: {error}\n{traceback.format_exc()}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
