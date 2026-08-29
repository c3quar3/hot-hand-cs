"""Review-driven raw-outcome contrast derived only from frozen artifacts.

This script computes the raw-outcome-type estimands requested by the round-2
review (R2-M1-new-1) as descriptive contrasts on the same E-A and E-B
partitions: a lag-1 raw-outcome state association and a run-conditioned success
association on raw wins and losses, using the same continuous-time handling
(z_t) and the same two-way CRV1 clustering as the primary residual analysis.
Its purpose is to show whether a raw-outcome operationalization gives a
different sign or time structure than the conditional-residual estimand.

It reads only frozen inputs (the E1-v2 residual asset with Y_t, p_hat, e_t and
the partition assignments) and writes to a new versioned output root. It does
not modify, rescore, or re-select any frozen protocol, result, gate record, or
interpretation boundary, and it is descriptive, not a new confirmatory test or
decision rule.

Models (all fixed-form, no re-estimation of the frozen probability model):
  A. lag-1 raw outcome:  Y_t ~ alpha + a z_t + beta H1* + kappa H1* z_t,
     where H1 is the standardized raw outcome (Y) of the strictly preceding
     eligible participant-event of the same player within the same map.
  B. run-conditioned success: Y_t ~ alpha + a z_t + beta R* + kappa R* z_t,
     where R is the standardized number of consecutive prior raw wins
     (win streak length) of the same player within the same map.
Reference centring for z_t uses the frozen mu_z; the raw states H1 and R are
standardized within each partition for this descriptive contrast only.
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
OUT = ROOT / "results" / "review_raw_outcome_contrast_v1"
ASSET = ROOT / "results" / "conditional_residual_asset_e1_v1" / "E1-v2" / "residual_asset.parquet"
ASSIGNMENTS = ROOT / "data" / "unified_evidence_two_partitions_v3" / "match_assignments.parquet"
ASSIGNMENT_MANIFEST = ROOT / "data" / "unified_evidence_two_partitions_v3" / "manifest.json"
CONFIRMATION_SUMMARY = ROOT / "results" / "unified_evidence_two_partition_confirmation_v1" / "checkpoint_05_summary.json"
PARTITIONS = ("E-A", "E-B")
SMOKE_MATCHES = 30
Z95 = 1.959963984540054


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


def load() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    assignment_manifest = read_json(ASSIGNMENT_MANIFEST)
    require(digest(ASSIGNMENTS) == assignment_manifest["files"]["match_assignments.parquet"], "assignment_manifest_hash_mismatch")
    assignments = pd.read_parquet(ASSIGNMENTS)
    assignments["match_id"] = assignments["match_id"].astype(str)
    require(assignments["match_id"].is_unique, "assignment_match_nonunique")
    asset = pd.read_parquet(ASSET)
    require(asset["physical_event_id"].is_unique, "asset_event_nonunique")
    require(set(asset["Y_t"].unique()) <= {0, 1}, "Y_t_not_binary")
    asset["match_id"] = asset["match_id"].astype(str)
    merged = asset.merge(assignments[["match_id", "partition"]], on="match_id", how="left")
    require(merged["partition"].notna().all(), "assignment_coverage_incomplete")
    confirmation = read_json(CONFIRMATION_SUMMARY)
    matches = {partition: sorted(assignments.loc[assignments["partition"].eq(partition), "match_id"].tolist()) for partition in PARTITIONS}
    return merged, matches, confirmation


def build_ledger(asset: pd.DataFrame, matches: list[str]) -> pd.DataFrame:
    """Build the two signed participant perspectives with raw outcomes."""
    subset = asset.loc[asset["match_id"].isin(matches)].copy()
    base = subset[["physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id", "opponent_player_id", "Y_t"]].copy()
    focal = base[["physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id", "Y_t"]].rename(columns={"focal_player_id": "player_id"})
    opponent = base[["physical_event_id", "match_id", "map_id", "event_tick", "opponent_player_id", "Y_t"]].rename(columns={"opponent_player_id": "player_id"})
    opponent["Y_t"] = 1 - opponent["Y_t"]
    ledger = pd.concat([focal, opponent], ignore_index=True)
    ledger["player_id"] = ledger["player_id"].astype(str)
    require(not ledger.duplicated(["physical_event_id", "player_id"]).any(), "duplicate_participant_physical_event")
    return ledger.sort_values(["match_id", "map_id", "player_id", "event_tick", "physical_event_id"], kind="stable").reset_index(drop=True)


def build_raw_states(ledger: pd.DataFrame) -> pd.DataFrame:
    """Compute lag-1 raw outcome and win-streak length per player-map over time."""
    rows: list[dict[str, Any]] = []
    for (match_id, map_id, player_id), group in ledger.groupby(["match_id", "map_id", "player_id"], sort=False):
        prior_y: float | None = None
        prior_tick: float | None = None
        streak = 0
        for tick, batch in group.groupby("event_tick", sort=False):
            tick_value = float(tick)
            if prior_tick is not None:
                pass  # interval handled below via d_tick
            for row in batch.itertuples(index=False):
                y = float(row.Y_t)
                rows.append({
                    "physical_event_id": str(row.physical_event_id), "match_id": str(match_id), "map_id": str(map_id),
                    "player_id": str(player_id), "event_tick": tick_value, "Y_t": y,
                    "lag1_raw": prior_y, "streak_prior": streak, "prior_tick": prior_tick,
                })
                streak = streak + 1 if y == 1.0 else 0
            prior_y = float(batch["Y_t"].iloc[-1])
            prior_tick = tick_value
    state = pd.DataFrame(rows)
    state["has_lag1"] = state["lag1_raw"].notna()
    return state


def fit_model(frame: pd.DataFrame, state_name: str, mu_z: float) -> dict[str, Any]:
    """OLS with two-way CRV1 (match, player); returns beta and kappa estimates."""
    use = frame.loc[frame["has_lag1"]].copy()
    use = use.loc[use["lag1_raw"].notna()].copy()
    n = len(use)
    require(n > 4, "too_few_rows")
    z = np.log2(1.0 + (use["event_tick"].to_numpy(dtype=float) - use["prior_tick"].to_numpy(dtype=float))) - mu_z
    state = use[state_name].to_numpy(dtype=float)
    state = (state - state.mean()) / (state.std(ddof=1) + 1e-12)
    design = np.column_stack([np.ones(n), z, state, state * z])
    k = design.shape[1]
    require(np.linalg.matrix_rank(design) == k, "rank_deficient")
    inverse = np.linalg.inv(design.T @ design)
    y = use["Y_t"].to_numpy(dtype=float)
    coefficient = inverse @ design.T @ y
    residual = y - design @ coefficient
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)

    def meat(groups: pd.Series) -> tuple[np.ndarray, int]:
        _, codes = np.unique(groups.astype(str).to_numpy(), return_inverse=True)
        summed = np.zeros((int(codes.max()) + 1, k))
        np.add.at(summed, codes, design * residual[:, None])
        return summed.T @ summed, len(summed)

    match_meat, matches = meat(use["match_id"])
    player_meat, players = meat(use["player_id"])
    inter_meat, intersections = meat(use["match_id"].astype(str) + "\x1f" + use["player_id"].astype(str))
    correction = lambda groups: (groups / (groups - 1)) * ((n - 1) / (n - k))
    covariance = inverse @ (correction(matches) * match_meat + correction(players) * player_meat - correction(intersections) * inter_meat) @ inverse
    covariance = (covariance + covariance.T) / 2
    names = ["alpha", "a_time", "beta_state", "kappa_state_time"]
    result: dict[str, Any] = {"n": int(n), "match_clusters": int(matches), "player_clusters": int(players), "intersection_clusters": int(intersections), "max_leverage": float(leverage.max())}
    for index, name in enumerate(names):
        est = float(coefficient[index])
        se = math.sqrt(float(covariance[index, index]))
        result[name] = {"estimate": est, "standard_error": se, "ci95_low": est - Z95 * se, "ci95_high": est + Z95 * se}
    result["beta"] = result["beta_state"]
    result["kappa"] = result["kappa_state_time"]
    return result


def run(phase: str) -> None:
    asset, matches, confirmation = load()
    mu_z = float(confirmation["reference"]["mu_z"])
    if phase == "smoke":
        match_sets = {partition: matches[partition][:SMOKE_MATCHES] for partition in PARTITIONS}
    else:
        match_sets = matches

    block: dict[str, Any] = {
        "stage": f"review_raw_outcome_contrast_v1_{phase}",
        "created_at_utc": now(),
        "phase": phase,
        "classification": "review_driven_descriptive_raw_outcome_contrast_no_change_to_frozen_primary_results",
        "inputs": {
            "asset": {"path": str(ASSET), "sha256": digest(ASSET)},
            "assignments": {"path": str(ASSIGNMENTS), "sha256": digest(ASSIGNMENTS)},
            "confirmation_summary": {"path": str(CONFIRMATION_SUMMARY), "sha256": digest(CONFIRMATION_SUMMARY)},
        },
        "mu_z": mu_z,
        "partitions": {},
    }

    for partition in PARTITIONS:
        ledger = build_ledger(asset, match_sets[partition])
        states = build_raw_states(ledger)
        # lag-1 raw outcome model: Y_t ~ z + lag1*
        lag1_rows = states.loc[states["has_lag1"]].copy()
        lag1_rows["state_var"] = lag1_rows["lag1_raw"].astype(float)
        lag1_model = fit_model(lag1_rows, "state_var", mu_z)
        # run-conditioned success model: Y_t ~ z + streak* (streak from prior wins)
        streak_rows = states.loc[states["has_lag1"]].copy()
        streak_rows["state_var"] = streak_rows["streak_prior"].astype(float)
        streak_model = fit_model(streak_rows, "state_var", mu_z)
        block["partitions"][partition] = {
            "physical_events": int(len(ledger) // 2),
            "participant_rows": int(len(states)),
            "lag1_rows": int(len(lag1_rows)),
            "lag1_raw_outcome_model": lag1_model,
            "run_conditioned_success_model": streak_model,
        }

    out_path = OUT / f"{phase}.json"
    if out_path.exists():
        saved = read_json(out_path)
        require(saved.get("stage") == block["stage"], "stage_mismatch")
        return
    write_json_new(out_path, block)
    manifest = {"stage": f"review_raw_outcome_contrast_v1_{phase}_manifest", "created_at_utc": now(), "files": {path.name: digest(path) for path in sorted(OUT.rglob("*")) if path.is_file()}}
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
