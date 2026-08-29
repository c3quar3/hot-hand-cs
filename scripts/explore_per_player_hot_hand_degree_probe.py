"""Exploratory feasibility probe for a per-player hot-hand degree measure.

Reads only frozen inputs (the E1-v2 residual asset, the partition assignments,
and the frozen confirmation reference values), rebuilds the participant-event
state with the frozen construction, and reports per-player descriptive
statistics and per-player fixed-model fits to judge whether a per-player
hot-hand degree supplement would be informative.

Exploratory probe only: prints descriptive statistics and writes no versioned
result package. If the author decides to implement the supplement, a separate
versioned script and result package must follow the review-driven-supplement
pattern.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_exponential_history_state_sequence_v1 import build_state, read_json, digest, require

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "results" / "conditional_residual_asset_e1_v1" / "E1-v2" / "residual_asset.parquet"
ASSIGNMENTS = ROOT / "data" / "unified_evidence_two_partitions_v3" / "match_assignments.parquet"
ASSIGNMENT_MANIFEST = ROOT / "data" / "unified_evidence_two_partitions_v3" / "manifest.json"
CONFIRMATION_SUMMARY = ROOT / "results" / "unified_evidence_two_partition_confirmation_v1" / "checkpoint_05_summary.json"
PARTITIONS = ("E-A", "E-B")
Z95 = 1.959963984540054
MIN_ROWS = 100  # minimum participant-event rows for a per-player fit in the probe
MIN_MATCHES = 2  # minimum distinct matches for one-way match-clustered inference


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


def main() -> None:
    manifest = read_json(ASSIGNMENT_MANIFEST)
    require(digest(ASSIGNMENTS) == manifest["files"]["match_assignments.parquet"], "assignment_hash_mismatch")
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

    for part in PARTITIONS:
        part_matches = sorted(assignments.loc[assignments["partition"].eq(part), "match_id"].tolist())
        sub = merged.loc[merged["match_id"].isin(part_matches)].copy()
        state, audit = build_state(sub, part)
        eligible = state.loc[state["analysis_eligible"]].copy()
        eligible["d_tick"] = eligible["event_tick"] - eligible["prior_tick"]
        eligible["match_id"] = eligible["match_id"].astype(str)
        eligible["player_id"] = eligible["player_id"].astype(str)

        n_players = eligible["player_id"].nunique()
        counts = eligible.groupby("player_id").size()
        q = counts.quantile([0.05, 0.25, 0.5, 0.75, 0.95])

        fits = []
        for player, rows in eligible.groupby("player_id"):
            if len(rows) >= MIN_ROWS:
                fits.append((player, fit_player(rows, reference)))
        ok = [f for f in fits if not f[1].get("rank_fail")]
        betas = np.array([f[1]["beta"] for f in ok])
        ses = np.array([f[1]["se"] for f in ok])
        cexcl = sum(1 for f in ok if f[1]["ci95_low"] > 0 or f[1]["ci95_high"] < 0)
        pooled = float(confirmation["models"][part]["beta"]["estimate"])
        pooled_se = float(confirmation["models"][part]["beta"]["standard_error"])

        print(f"==== partition {part} ====")
        print(f"  source physical engagements   : {len(sub)}")
        print(f"  eligible participant rows      : {len(eligible)}")
        print(f"  distinct players               : {n_players}")
        print(f"  per-player row counts          : min={counts.min()}, p05={q[0.05]:.0f}, p25={q[0.25]:.0f}, "
              f"median={q[0.5]:.0f}, p75={q[0.75]:.0f}, p95={q[0.95]:.0f}, max={counts.max()}")
        print(f"  players with >= {MIN_ROWS} rows    : {len(fits)} (fits ok: {len(ok)})")
        if len(ok) >= 5:
            print(f"  per-player beta   : mean={betas.mean():.4f}, sd={betas.std(ddof=1):.4f}, "
                  f"median={np.median(betas):.4f}, p10={np.quantile(betas, 0.10):.4f}, p90={np.quantile(betas, 0.90):.4f}")
            print(f"  fraction beta>0   : {(betas > 0).mean():.3f}")
            print(f"  CI excludes 0     : {cexcl}/{len(ok)}")
            print(f"  per-player se     : median={np.median(ses):.4f}, min={ses.min():.4f}, max={ses.max():.4f}")
            print(f"  pooled beta (frozen) : {pooled:.4f} (SE {pooled_se:.4f})")
            print(f"  heterogeneity vs noise: sd(beta_j)={betas.std(ddof=1):.4f} vs median(se_j)={np.median(ses):.4f}")
            for label, idx in (("top rows", np.argmax(counts.to_numpy())),):
                pass
            top = counts.sort_values(ascending=False).head(5)
            print(f"  top-5 players by rows: {[f'{p}({r})' for p, r in top.items()]}")
        else:
            print(f"  insufficient players for per-player fits")


if __name__ == "__main__":
    main()
