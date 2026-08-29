"""Review-driven conditioning-increment contrast derived only from frozen artifacts.

This script addresses the round-17 significance concern (R2-M1-17-1) by
demonstrating, in a descriptive way, whether conditioning on the recorded
pre-engagement information set changes the association reading in a specific
subgroup of the observed interval support. It stratifies the primary-analysis
rows by the strict-prior eligible interval (below and above the prespecified
46.45-s reference interval) and, within each stratum, estimates both a
raw-outcome lag-1 association (Y_t regressed on the lag-1 raw outcome state)
and the conditional-residual 60-s-history association (e_t regressed on the
60-s history state), using the same continuous-time centring (mu_z) and the
same two-way CRV1 clustering as the primary analysis.

The contrast is descriptive and derived only from frozen inputs (the E1-v2
residual asset, the partition assignments, and the confirmation summary). It
does not modify, rescore, or re-select any frozen protocol, result, gate
record, or interpretation boundary, and it is not a new confirmatory test or
decision rule.

Stratum definition: rows with d_tick below the reference (46.45 s) form the
short-interval stratum; rows at or above form the long-interval stratum. The
two models within each stratum are the raw lag-1 model and the residual 60-s
model, mirroring the models in Table 4 and Table 2 respectively, restricted
to the stratum rows.

Reference centring uses the frozen mu_z for z_t. The raw lag-1 state is
standardized within each stratum for this descriptive contrast only; the
60-s history state uses the frozen D1-A reference scale (mu_S, sigma_S).
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
OUT = ROOT / "results" / "review_conditioning_contrast_v1"
ASSET = ROOT / "results" / "conditional_residual_asset_e1_v1" / "E1-v2" / "residual_asset.parquet"
ASSIGNMENTS = ROOT / "data" / "unified_evidence_two_partitions_v3" / "match_assignments.parquet"
ASSIGNMENT_MANIFEST = ROOT / "data" / "unified_evidence_two_partitions_v3" / "manifest.json"
CONFIRMATION_SUMMARY = ROOT / "results" / "unified_evidence_two_partition_confirmation_v1" / "checkpoint_05_summary.json"
PARTITIONS = ("E-A", "E-B")
SMOKE_MATCHES = 30
Z95 = 1.959963984540054
REFERENCE_SECONDS = 46.45
TICK_RATE = 64.0


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
    """Compute lag-1 raw outcome per player-map over time."""
    rows: list[dict[str, Any]] = []
    for (match_id, map_id, player_id), group in ledger.groupby(["match_id", "map_id", "player_id"], sort=False):
        prior_y: float | None = None
        prior_tick: float | None = None
        for tick, batch in group.groupby("event_tick", sort=False):
            tick_value = float(tick)
            for row in batch.itertuples(index=False):
                rows.append({
                    "physical_event_id": str(row.physical_event_id), "match_id": str(match_id), "map_id": str(map_id),
                    "player_id": str(player_id), "event_tick": tick_value, "Y_t": float(row.Y_t),
                    "lag1_raw": prior_y, "prior_tick": prior_tick,
                })
            prior_y = float(batch["Y_t"].iloc[-1])
            prior_tick = tick_value
    state = pd.DataFrame(rows)
    state["has_lag1"] = state["lag1_raw"].notna()
    return state


def build_history_state(asset: pd.DataFrame, matches: list[str]) -> pd.DataFrame:
    """Build the 60-s exponentially weighted history state per player-map."""
    subset = asset.loc[asset["match_id"].isin(matches)].copy()
    base = subset[["physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id", "opponent_player_id", "e_t"]].copy()
    focal = base[["physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id", "e_t"]].rename(columns={"focal_player_id": "player_id"})
    opponent = base[["physical_event_id", "match_id", "map_id", "event_tick", "opponent_player_id", "e_t"]].rename(columns={"opponent_player_id": "player_id"})
    opponent["e_t"] = -opponent["e_t"]
    ledger = pd.concat([focal, opponent], ignore_index=True)
    ledger["player_id"] = ledger["player_id"].astype(str)
    ledger = ledger.sort_values(["match_id", "map_id", "player_id", "event_tick", "physical_event_id"], kind="stable").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for (match_id, map_id, player_id), group in ledger.groupby(["match_id", "map_id", "player_id"], sort=False):
        running = 0.0
        prior_tick: float | None = None
        history_ticks: list[float] = []
        history_e: list[float] = []
        for tick, batch in group.groupby("event_tick", sort=False):
            tick_value = float(tick)
            if prior_tick is None:
                pre_state = 0.0
            else:
                q = 2.0 ** (-((tick_value - prior_tick) / TICK_RATE) / 60.0)
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


def fit_crv1(frame: pd.DataFrame, response_name: str, state_name: str, mu_z: float, standardize_within: bool) -> dict[str, Any]:
    """OLS with two-way CRV1 (match, player); returns beta and kappa estimates."""
    n = len(frame)
    require(n > 4, "too_few_rows")
    z = np.log2(1.0 + frame["d_tick"].to_numpy(dtype=float)) - mu_z
    state = frame[state_name].to_numpy(dtype=float)
    if standardize_within:
        state = (state - state.mean()) / (state.std(ddof=1) + 1e-12)
    design = np.column_stack([np.ones(n), z, state, state * z])
    k = design.shape[1]
    require(np.linalg.matrix_rank(design) == k, "rank_deficient")
    inverse = np.linalg.inv(design.T @ design)
    response = frame[response_name].to_numpy(dtype=float)
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
    ref = confirmation["reference"]
    mu_s = float(ref["mu_S"])
    sigma_s = float(ref["sigma_S_ddof1"])
    reference_ticks = REFERENCE_SECONDS * TICK_RATE
    if phase == "smoke":
        match_sets = {partition: matches[partition][:SMOKE_MATCHES] for partition in PARTITIONS}
    else:
        match_sets = matches

    block: dict[str, Any] = {
        "stage": f"review_conditioning_contrast_v1_{phase}",
        "created_at_utc": now(),
        "phase": phase,
        "classification": "review_driven_descriptive_conditioning_increment_contrast_no_change_to_frozen_primary_results",
        "inputs": {
            "asset": {"path": str(ASSET), "sha256": digest(ASSET)},
            "assignments": {"path": str(ASSIGNMENTS), "sha256": digest(ASSIGNMENTS)},
            "confirmation_summary": {"path": str(CONFIRMATION_SUMMARY), "sha256": digest(CONFIRMATION_SUMMARY)},
        },
        "mu_z": mu_z, "mu_S": mu_s, "sigma_S_ddof1": sigma_s,
        "reference_seconds": REFERENCE_SECONDS, "reference_ticks": reference_ticks,
        "partitions": {},
    }

    for partition in PARTITIONS:
        raw_ledger = build_ledger(asset, match_sets[partition])
        raw_states = build_raw_states(raw_ledger)
        hist = build_history_state(asset, match_sets[partition])
        # Merge: raw lag-1 state (Y scale) and residual 60-s history state (e scale)
        raw_use = raw_states.loc[raw_states["has_lag1"], ["physical_event_id", "player_id", "Y_t", "lag1_raw"]].copy()
        hist_use = hist[["physical_event_id", "player_id", "match_id", "e_t", "S_h60", "d_tick"]].copy()
        merged = hist_use.merge(raw_use, on=["physical_event_id", "player_id"], how="inner", validate="one_to_one")
        require(len(merged) == len(hist_use), "raw_history_join_incomplete")
        # Residual 60-s model on the frozen reference scale
        merged["S_ref"] = (merged["S_h60"].to_numpy(dtype=float) - mu_s) / sigma_s
        short = merged.loc[merged["d_tick"] < reference_ticks].copy()
        long = merged.loc[merged["d_tick"] >= reference_ticks].copy()
        strat = {}
        for label, stratum in (("short_interval_below_46p45s", short), ("long_interval_at_or_above_46p45s", long)):
            raw_model = fit_crv1(stratum, "Y_t", "lag1_raw", mu_z, standardize_within=True)
            res_model = fit_crv1(stratum, "e_t", "S_ref", mu_z, standardize_within=False)
            strat[label] = {"rows": int(len(stratum)), "raw_lag1_outcome_model": raw_model, "residual_60s_history_model": res_model}
        block["partitions"][partition] = {
            "physical_events": int(len(raw_ledger) // 2),
            "participant_rows": int(len(merged)),
            "strata": strat,
        }

    out_path = OUT / f"{phase}.json"
    if out_path.exists():
        saved = read_json(out_path)
        require(saved.get("stage") == block["stage"], "stage_mismatch")
        return
    write_json_new(out_path, block)
    manifest = {"stage": f"review_conditioning_contrast_v1_{phase}_manifest", "created_at_utc": now(), "files": {path.name: digest(path) for path in sorted(OUT.rglob("*")) if path.is_file()}}
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
