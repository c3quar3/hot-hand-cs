"""Focal-direction-only sensitivity for the primary association estimate.

R1-M1-26.1 (Blocking Yes) requires, when the role-swap antisymmetry deviation
is non-negligible, either a redefinition of the opponent records or a
focal-direction-only sensitivity analysis. This script implements the latter:
it rebuilds the participant-event sequences using ONLY the canonical focal
direction of each physical engagement (the opponent participant-event records
are dropped), reconstructs the 60-s exponentially weighted history state and
the continuous-time term, and refits the primary association model with the
same two-way CRV1 clustering, in E-A and E-B.

It reads only frozen inputs (the E1-v2 residual asset with Y_t, p_hat, e_t and
the partition assignments, plus the confirmation summary for the frozen
reference values) and writes to a new versioned output root. It is a
descriptive sensitivity check, not a new confirmatory test, and does not modify
any frozen result.
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


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "review_focal_direction_sensitivity_v1"
ASSET = ROOT / "results" / "conditional_residual_asset_e1_v1" / "E1-v2" / "residual_asset.parquet"
ASSIGNMENTS = ROOT / "data" / "unified_evidence_two_partitions_v3" / "match_assignments.parquet"
ASSIGNMENT_MANIFEST = ROOT / "data" / "unified_evidence_two_partitions_v3" / "manifest.json"
CONFIRMATION_SUMMARY = ROOT / "results" / "unified_evidence_two_partition_confirmation_v1" / "checkpoint_05_summary.json"
PARTITIONS = ("E-A", "E-B")
SMOKE_MATCHES = 30
Z95 = 1.959963984540054
TICK_RATE = 64.0
HALF_LIFE_SECONDS = 60.0


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


def build_focal_state(asset: pd.DataFrame, matches: list[str]) -> pd.DataFrame:
    """Build the 60-s history state using only the canonical focal direction."""
    subset = asset.loc[asset["match_id"].isin(matches)].copy()
    frame = subset[["physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id", "e_t"]].copy()
    frame = frame.rename(columns={"focal_player_id": "player_id"})
    frame["player_id"] = frame["player_id"].astype(str)
    frame = frame.sort_values(["match_id", "map_id", "player_id", "event_tick", "physical_event_id"], kind="stable").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for (match_id, map_id, player_id), group in frame.groupby(["match_id", "map_id", "player_id"], sort=False):
        running = 0.0
        prior_tick: float | None = None
        history_ticks: list[float] = []
        history_e: list[float] = []
        for tick, batch in group.groupby("event_tick", sort=False):
            tick_value = float(tick)
            if prior_tick is None:
                pre_state = 0.0
            else:
                q = 2.0 ** (-((tick_value - prior_tick) / TICK_RATE) / HALF_LIFE_SECONDS)
                pre_state = q * running
            for row in batch.itertuples(index=False):
                rows.append({
                    "physical_event_id": str(row.physical_event_id), "match_id": str(match_id), "map_id": str(map_id),
                    "player_id": str(player_id), "event_tick": tick_value, "e_t": float(row.e_t),
                    "S_h60": pre_state, "strict_prior_event_count": len(history_e), "prior_tick": prior_tick,
                })
            running = pre_state + float(batch["e_t"].sum())
            history_ticks.extend([tick_value] * len(batch))
            history_e.extend(batch["e_t"].astype(float).tolist())
            prior_tick = tick_value
    state = pd.DataFrame(rows)
    state["analysis_eligible"] = state["strict_prior_event_count"].ge(1)
    state = state.loc[state["analysis_eligible"]].copy()
    state["d_tick"] = state["event_tick"] - state["prior_tick"]
    require((state["d_tick"] >= 1).all(), "nonpositive_history_interval")
    return state.sort_values(["match_id", "map_id", "player_id", "event_tick", "physical_event_id"], kind="stable").reset_index(drop=True)


def fit_crv1(frame: pd.DataFrame, reference: dict[str, float]) -> dict[str, Any]:
    n = len(frame)
    require(n > 4, "too_few_rows")
    z = np.log2(1.0 + frame["d_tick"].to_numpy(dtype=float)) - float(reference["mu_z"])
    s = (frame["S_h60"].to_numpy(dtype=float) - float(reference["mu_S"])) / float(reference["sigma_S_ddof1"])
    design = np.column_stack([np.ones(n), z, s, s * z])
    names = ["alpha", "a_time", "beta_state", "kappa_state_time"]
    k = design.shape[1]
    require(np.linalg.matrix_rank(design) == k, "rank_deficient")
    inverse = np.linalg.inv(design.T @ design)
    response = frame["e_t"].to_numpy(dtype=float)
    coefficient = inverse @ design.T @ response
    residual = response - design @ coefficient
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)

    def meat(groups: pd.Series) -> tuple[np.ndarray, int]:
        _, codes = np.unique(groups.astype(str).to_numpy(), return_inverse=True)
        summed = np.zeros((int(codes.max()) + 1, k))
        np.add.at(summed, codes, design * residual[:, None])
        return summed.T @ summed, len(summed)

    match_meat, matches = meat(frame["match_id"])
    player_meat, players = meat(frame["player_id"])
    inter_meat, intersections = meat(frame["match_id"].astype(str) + "\x1f" + frame["player_id"].astype(str))
    correction = lambda groups: (groups / (groups - 1)) * ((n - 1) / (n - k))
    covariance = inverse @ (correction(matches) * match_meat + correction(players) * player_meat - correction(intersections) * inter_meat) @ inverse
    covariance = (covariance + covariance.T) / 2
    result: dict[str, Any] = {"n": int(n), "match_clusters": int(matches), "player_clusters": int(players), "intersection_clusters": int(intersections), "max_leverage": float(leverage.max())}
    for index, name in enumerate(names):
        est = float(coefficient[index])
        se = math.sqrt(float(covariance[index, index]))
        result[name] = {"estimate": est, "standard_error": se, "ci95_low": est - Z95 * se, "ci95_high": est + Z95 * se}
    result["beta"] = result["beta_state"]
    result["kappa"] = result["kappa_state_time"]
    return result


def run(phase: str) -> None:
    assignment_manifest = read_json(ASSIGNMENT_MANIFEST)
    require(digest(ASSIGNMENTS) == assignment_manifest["files"]["match_assignments.parquet"], "assignment_hash_mismatch")
    assignments = pd.read_parquet(ASSIGNMENTS)
    assignments["match_id"] = assignments["match_id"].astype(str)
    require(assignments["match_id"].is_unique, "assignment_nonunique")
    asset = pd.read_parquet(ASSET)
    require(asset["physical_event_id"].is_unique, "asset_event_nonunique")
    asset["match_id"] = asset["match_id"].astype(str)
    merged = asset.merge(assignments[["match_id", "partition"]], on="match_id", how="left")
    require(merged["partition"].notna().all(), "assignment_coverage_incomplete")
    confirmation = read_json(CONFIRMATION_SUMMARY)
    reference = confirmation["reference"]
    matches = {part: sorted(assignments.loc[assignments["partition"].eq(part), "match_id"].tolist()) for part in PARTITIONS}
    if phase == "smoke":
        matches = {part: matches[part][:SMOKE_MATCHES] for part in PARTITIONS}

    block: dict[str, Any] = {
        "stage": f"review_focal_direction_sensitivity_v1_{phase}",
        "created_at_utc": now(),
        "phase": phase,
        "classification": "review_driven_descriptive_focal_direction_only_sensitivity_no_change_to_frozen_results",
        "inputs": {
            "asset": {"path": str(ASSET), "sha256": digest(ASSET)},
            "assignments": {"path": str(ASSIGNMENTS), "sha256": digest(ASSIGNMENTS)},
            "confirmation_summary": {"path": str(CONFIRMATION_SUMMARY), "sha256": digest(CONFIRMATION_SUMMARY)},
        },
        "reference": {key: float(reference[key]) for key in ("mu_z", "mu_S", "sigma_S_ddof1")},
        "partitions": {},
    }
    for part in PARTITIONS:
        state = build_focal_state(merged, matches[part])
        model = fit_crv1(state, reference)
        block["partitions"][part] = {
            "participant_rows_focal_direction_only": int(len(state)),
            "physical_engagements_full_sample": int(len(merged[merged["partition"] == part])),
            "model": model,
        }
    out_path = OUT / f"{phase}.json"
    if out_path.exists():
        saved = read_json(out_path)
        require(saved.get("stage") == block["stage"], "stage_mismatch")
        return
    write_json_new(out_path, block)
    manifest = {"stage": f"review_focal_direction_sensitivity_v1_{phase}_manifest", "created_at_utc": now(),
                "files": {path.name: digest(path) for path in sorted(OUT.rglob("*")) if path.is_file()}}
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
