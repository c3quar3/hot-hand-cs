"""Descriptive per-player hot-hand degree supplement for the primary association.

The primary estimand beta is a pooled association at the fixed 46.45-s
reference interval. This supplement re-estimates the same fixed-form model per
player (using the frozen history state, continuous-time term, and reference
standardization) and reports the resulting per-player beta distribution, to
describe how the pooled positive association is distributed across players.

Construction: the participant-event state is rebuilt with the frozen
build_state construction (strict-prior 60-s half-life exponentially weighted
same-player residual accumulation, concurrent-tick groups sharing pre-tick
history). Each player's association is fit on that player's eligible rows with
the fixed model e_t = alpha + a z + beta S + kappa S z, one-way CRV1 clustering
by match (per player, two-way clustering is degenerate because the player is a
single cluster). Players with fewer than MIN_ROWS participant-event rows or
fewer than MIN_MATCHES distinct matches are not fit.

It reads only frozen inputs (the E1-v2 residual asset, the partition
assignments, and the confirmation summary for the frozen reference values) and
writes to a new versioned output root. It is a descriptive supplement, not a
new test or decision rule, does not claim per-player significance, and does not
modify any frozen result.
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

from analyze_exponential_history_state_sequence_v1 import build_state, read_json, require

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "describe_per_player_hot_hand_degree_v1"
ASSET = ROOT / "results" / "conditional_residual_asset_e1_v1" / "E1-v2" / "residual_asset.parquet"
ASSIGNMENTS = ROOT / "data" / "unified_evidence_two_partitions_v3" / "match_assignments.parquet"
ASSIGNMENT_MANIFEST = ROOT / "data" / "unified_evidence_two_partitions_v3" / "manifest.json"
CONFIRMATION_SUMMARY = ROOT / "results" / "unified_evidence_two_partition_confirmation_v1" / "checkpoint_05_summary.json"
PARTITIONS = ("E-A", "E-B")
SMOKE_MATCHES = 30
Z95 = 1.959963984540054
MIN_ROWS = 100
MIN_MATCHES = 2


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_parquet_new(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def fit_player(frame: pd.DataFrame, reference: dict[str, float]) -> dict[str, Any]:
    n = len(frame)
    z = np.log2(1.0 + frame["d_tick"].to_numpy(dtype=float)) - float(reference["mu_z"])
    s = (frame["S_h60"].to_numpy(dtype=float) - float(reference["mu_S"])) / float(reference["sigma_S_ddof1"])
    design = np.column_stack([np.ones(n), z, s, s * z])
    k = design.shape[1]
    match_ids = frame["match_id"].astype(str).to_numpy()
    groups = int(len(np.unique(match_ids)))
    if n <= k or np.linalg.matrix_rank(design) < k or groups < MIN_MATCHES:
        return {"n": int(n), "match_clusters": groups, "rank_fail": True}
    inverse = np.linalg.inv(design.T @ design)
    response = frame["e_t"].to_numpy(dtype=float)
    coefficient = inverse @ design.T @ response
    residual = response - design @ coefficient
    _, codes = np.unique(match_ids, return_inverse=True)
    summed = np.zeros((int(codes.max()) + 1, k))
    np.add.at(summed, codes, design * residual[:, None])
    correction = (groups / (groups - 1)) * ((n - 1) / (n - k))
    covariance = inverse @ (correction * (summed.T @ summed)) @ inverse
    covariance = (covariance + covariance.T) / 2
    se = math.sqrt(max(float(covariance[2, 2]), 0.0))
    beta = float(coefficient[2])
    return {
        "n": int(n), "match_clusters": groups, "rank_fail": False,
        "beta": beta, "se": se,
        "ci95_low": beta - Z95 * se, "ci95_high": beta + Z95 * se,
    }


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
    reference = {key: float(confirmation["reference"][key]) for key in ("mu_z", "mu_S", "sigma_S_ddof1")}
    matches = {part: sorted(assignments.loc[assignments["partition"].eq(part), "match_id"].tolist()) for part in PARTITIONS}
    if phase == "smoke":
        matches = {part: matches[part][:SMOKE_MATCHES] for part in PARTITIONS}

    block: dict[str, Any] = {
        "stage": f"describe_per_player_hot_hand_degree_v1_{phase}",
        "created_at_utc": now(),
        "phase": phase,
        "classification": "descriptive_per_player_hot_hand_degree_supplement_no_change_to_frozen_results",
        "inputs": {
            "asset": {"path": str(ASSET), "sha256": digest(ASSET)},
            "assignments": {"path": str(ASSIGNMENTS), "sha256": digest(ASSIGNMENTS)},
            "confirmation_summary": {"path": str(CONFIRMATION_SUMMARY), "sha256": digest(CONFIRMATION_SUMMARY)},
        },
        "reference": {key: float(reference[key]) for key in ("mu_z", "mu_S", "sigma_S_ddof1")},
        "method": {
            "model": "e_t = alpha + a z + beta S + kappa S z (fixed form)",
            "clustering": "one-way CRV1 by match (per player, two-way is degenerate)",
            "min_rows": MIN_ROWS,
            "min_matches": MIN_MATCHES,
            "history": "frozen strict-prior 60-s half-life exponentially weighted same-player residual accumulation",
            "reporting_boundary": "descriptive distribution; no per-player significance claims; not part of the prespecified test family",
        },
        "partitions": {},
    }
    player_frames: dict[str, pd.DataFrame] = {}
    for part in PARTITIONS:
        part_matches = matches[part]
        sub = merged.loc[merged["match_id"].isin(part_matches)].copy()
        state, audit = build_state(sub, part)
        eligible = state.loc[state["analysis_eligible"]].copy()
        eligible["d_tick"] = eligible["event_tick"] - eligible["prior_tick"]
        eligible["match_id"] = eligible["match_id"].astype(str)
        eligible["player_id"] = eligible["player_id"].astype(str)

        counts = eligible.groupby("player_id").size()
        quantiles = {f"q{int(q * 100):02d}": float(counts.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}

        records: list[dict[str, Any]] = []
        for player, rows in eligible.groupby("player_id"):
            if len(rows) >= MIN_ROWS:
                fit = fit_player(rows, reference)
                records.append({"player_id": player, **fit})
        ok = [r for r in records if not r.get("rank_fail")]
        betas = np.asarray([r["beta"] for r in ok], dtype=float)
        ses = np.asarray([r["se"] for r in ok], dtype=float)
        cexcl = sum(1 for r in ok if r["ci95_low"] > 0 or r["ci95_high"] < 0)
        pooled = float(confirmation["models"][part]["beta"]["estimate"])
        pooled_se = float(confirmation["models"][part]["beta"]["standard_error"])

        part_block: dict[str, Any] = {
            "physical_engagements": int(len(sub)),
            "participant_rows": int(len(eligible)),
            "distinct_players": int(counts.shape[0]),
            "player_row_count_quantiles": quantiles,
            "player_row_count_min": int(counts.min()),
            "player_row_count_max": int(counts.max()),
            "players_fitted": len(ok),
            "beta_distribution": {
                "mean": float(betas.mean()) if len(betas) else None,
                "sd_ddof1": float(betas.std(ddof=1)) if len(betas) > 1 else None,
                "median": float(np.median(betas)) if len(betas) else None,
                "p10": float(np.quantile(betas, 0.10)) if len(betas) else None,
                "p90": float(np.quantile(betas, 0.90)) if len(betas) else None,
                "fraction_positive": float((betas > 0).mean()) if len(betas) else None,
                "ci_excludes_zero": int(cexcl),
            },
            "per_player_se": {
                "median": float(np.median(ses)) if len(ses) else None,
                "min": float(ses.min()) if len(ses) else None,
                "max": float(ses.max()) if len(ses) else None,
            },
            "heterogeneity_vs_noise": {
                "sd_beta_j": float(betas.std(ddof=1)) if len(betas) > 1 else None,
                "median_se_j": float(np.median(ses)) if len(ses) else None,
                "ratio": float(betas.std(ddof=1) / np.median(ses)) if len(betas) > 1 and len(ses) and np.median(ses) > 0 else None,
            },
            "pooled_beta_frozen": {"estimate": pooled, "standard_error": pooled_se},
            "state_audit": audit,
        }
        block["partitions"][part] = part_block
        player_frames[part] = pd.DataFrame(records)

    out_summary = OUT / f"{phase}.json"
    if out_summary.exists():
        saved = read_json(out_summary)
        require(saved.get("stage") == block["stage"], "stage_mismatch")
        return
    write_json_new(out_summary, block)
    for part in PARTITIONS:
        write_parquet_new(OUT / f"{phase}_per_player_{part}.parquet", player_frames[part])
    manifest = {"stage": f"describe_per_player_hot_hand_degree_v1_{phase}_manifest", "created_at_utc": now(),
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
