"""Verify the role-swap antisymmetry of the frozen probability model.

Reviewer 1 (round 26, R1-M1-26.1, Blocking Yes) requires verifying that the
opponent-view residual -e_t is the opponent's conditional Pearson residual under
the frozen model, which holds only if the model satisfies role-swap
antisymmetry, p_opp = 1 - p_hat (focal view). This script:

1. Reads the E1-v2 candidate cache (frozen pre-outcome features).
2. Constructs the opponent-focal feature row by swapping all focal/opponent
   paired features, negating the difference features, flipping focal_side, and
   swapping the identity and weapon features.
3. Scores both the original focal row and the swapped opponent row with the
   frozen model.
4. Reports the distribution of p_opp + p_hat - 1 (the antisymmetry deviation)
   overall and within E-A and E-B partitions.

It reads only frozen inputs and writes to a new versioned output root. It is a
descriptive verification of a model property, not a new confirmatory test or a
change to any frozen result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "review_antisymmetry_verification_v1"
CANDIDATE = ROOT / "results" / "conditional_residual_asset_candidate_cache_v1" / "E1-v2" / "candidate.parquet"
CANDIDATE_MANIFEST = ROOT / "results" / "conditional_residual_asset_candidate_cache_v1" / "E1-v2" / "candidate_manifest.json"
ASSIGNMENTS = ROOT / "data" / "unified_evidence_two_partitions_v3" / "match_assignments.parquet"
ASSIGNMENT_MANIFEST = ROOT / "data" / "unified_evidence_two_partitions_v3" / "manifest.json"
MODEL = ROOT / "results" / "d1a_c_selection_training_v1" / "full_model.joblib"
MODEL_MANIFEST = ROOT / "results" / "d1a_c_selection_training_v1" / "manifest.json"

PAIRED_DIFF = (
    "focal_minus_opponent_health", "focal_minus_opponent_team_alive",
    "focal_minus_opponent_post_buy_money", "focal_minus_opponent_post_buy_equipment_value",
    "focal_minus_opponent_team_post_buy_equipment_value",
)
PAIRED_SWAP = (
    "focal_speed", "opponent_speed", "focal_has_armor", "opponent_has_armor",
    "focal_has_helmet", "opponent_has_helmet", "focal_has_defuser", "opponent_has_defuser",
    "focal_facing_error_radians", "opponent_facing_error_radians",
    "focal_nearest_living_teammate_distance", "opponent_nearest_living_teammate_distance",
    "focal_nearest_living_enemy_distance", "opponent_nearest_living_enemy_distance",
    "focal_is_scoped", "opponent_is_scoped", "focal_is_alone", "opponent_is_alone",
    "focal_is_flashed", "opponent_is_flashed", "focal_flash_remaining_seconds",
    "opponent_flash_remaining_seconds", "focal_is_defusing", "opponent_is_defusing",
)
PAIRED_SWAP_CAT = (
    "focal_speed_source", "opponent_speed_source", "focal_weapon", "opponent_weapon",
    "focal_steamid", "opponent_steamid",
)
SIDE_SWAP = {"T": "CT", "CT": "T"}


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


def swap_opponent(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct the opponent-focal feature row by role swap."""
    out = frame.copy()
    for col in PAIRED_DIFF:
        out[col] = -frame[col]
    for col in PAIRED_SWAP:
        peer = col.replace("focal_", "opponent_") if col.startswith("focal_") else col.replace("opponent_", "focal_")
        out[col] = frame[peer]
    for col in PAIRED_SWAP_CAT:
        peer = col.replace("focal_", "opponent_") if col.startswith("focal_") else col.replace("opponent_", "focal_")
        out[col] = frame[peer]
    out["focal_side"] = frame["focal_side"].map(SIDE_SWAP)
    return out


def design_matrix(frame: pd.DataFrame, design: dict[str, Any]) -> sparse.csr_matrix:
    numeric = np.column_stack([
        (pd.to_numeric(frame[column], errors="raise").to_numpy(float) - float(design["means"][column])) / float(design["stds"][column])
        for column in design["numeric"]
    ])
    blocks: list[sparse.csr_matrix] = [sparse.csr_matrix(numeric)]
    for column in design["categorical"]:
        categories = [str(value) for value in design["categories"][column]]
        reference = str(design["references"][column])
        lookup = {value: index for index, value in enumerate(value for value in categories if value != reference)}
        values = frame[column].astype(str).to_numpy()
        rows = np.fromiter((index for index, value in enumerate(values) if value in lookup), dtype=np.int64)
        columns = np.fromiter((lookup[value] for value in values if value in lookup), dtype=np.int64)
        blocks.append(sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(frame), len(lookup))))
    return sparse.hstack(blocks, format="csr")


def run(phase: str) -> None:
    # verify hashes
    cand_manifest = read_json(CANDIDATE_MANIFEST)
    require(digest(CANDIDATE) == cand_manifest.get("candidate_sha256"), "candidate_hash_mismatch")
    assign_manifest = read_json(ASSIGNMENT_MANIFEST)
    require(digest(ASSIGNMENTS) == assign_manifest["files"]["match_assignments.parquet"], "assignment_hash_mismatch")
    model_manifest = read_json(MODEL_MANIFEST)
    require(digest(MODEL) == model_manifest["files"]["full_model.joblib"], "model_hash_mismatch")

    artifact = joblib.load(MODEL)
    design = artifact["design"]
    model = artifact["model"]

    cand = pd.read_parquet(CANDIDATE)
    assign = pd.read_parquet(ASSIGNMENTS)
    assign["match_id"] = assign["match_id"].astype(str)
    cand["match_id"] = cand["match_id"].astype(str)
    if "partition" in cand.columns:
        cand = cand.drop(columns=["partition"])
    merged = cand.merge(assign[["match_id", "partition"]], on="match_id", how="left")
    require(merged["partition"].notna().all(), "partition_coverage_incomplete")
    require(merged["duel_id"].is_unique, "duel_nonunique")

    if phase == "smoke":
        merged = merged.iloc[:500].copy()

    p_focal = model.predict_proba(design_matrix(merged, design))[:, 1]
    swapped = swap_opponent(merged)
    p_opp = model.predict_proba(design_matrix(swapped, design))[:, 1]
    deviation = p_opp + p_focal - 1.0

    summary: dict[str, Any] = {"rows": int(len(merged))}
    all_dev = deviation
    summary["all"] = {
        "mean": float(all_dev.mean()), "sd": float(all_dev.std(ddof=1)),
        "median": float(np.median(all_dev)), "p2_5": float(np.quantile(all_dev, 0.025)),
        "p97_5": float(np.quantile(all_dev, 0.975)), "max_abs": float(np.max(np.abs(all_dev))),
        "frac_abs_gt_0p01": float((np.abs(all_dev) > 0.01).mean()),
        "frac_abs_gt_0p05": float((np.abs(all_dev) > 0.05).mean()),
    }
    for part in ("E-A", "E-B"):
        mask = merged["partition"] == part
        sub = deviation[mask.to_numpy()]
        if len(sub) == 0:
            summary[part] = {"rows": 0, "note": "no rows in this phase subset"}
            continue
        summary[part] = {
            "rows": int(mask.sum()), "mean": float(sub.mean()), "sd": float(sub.std(ddof=1)),
            "median": float(np.median(sub)), "p2_5": float(np.quantile(sub, 0.025)),
            "p97_5": float(np.quantile(sub, 0.975)), "max_abs": float(np.max(np.abs(sub))),
            "frac_abs_gt_0p01": float((np.abs(sub) > 0.01).mean()),
            "frac_abs_gt_0p05": float((np.abs(sub) > 0.05).mean()),
        }

    block: dict[str, Any] = {
        "stage": f"review_antisymmetry_verification_v1_{phase}",
        "created_at_utc": now(),
        "phase": phase,
        "classification": "review_driven_descriptive_role_swap_antisymmetry_verification_no_change_to_frozen_results",
        "inputs": {
            "candidate": {"path": str(CANDIDATE), "sha256": digest(CANDIDATE)},
            "assignments": {"path": str(ASSIGNMENTS), "sha256": digest(ASSIGNMENTS)},
            "model": {"path": str(MODEL), "sha256": digest(MODEL)},
        },
        "antisymmetry": summary,
        "interpretation": "deviation = p_opp + p_hat - 1; values near zero indicate the frozen model satisfies role-swap antisymmetry, so -e_t is the opponent-view conditional Pearson residual.",
    }
    out_path = OUT / f"{phase}.json"
    if out_path.exists():
        saved = read_json(out_path)
        require(saved.get("stage") == block["stage"], "stage_mismatch")
        return
    write_json_new(out_path, block)
    manifest = {"stage": f"review_antisymmetry_verification_v1_{phase}_manifest", "created_at_utc": now(),
                "files": {path.name: digest(path) for path in sorted(OUT.rglob("*")) if path.is_file()}}
    write_json_new(OUT / f"{phase}_manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


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
