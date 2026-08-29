"""Versioned uncertainty supplement for the already scored D1-B calibration fit."""

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
OUT = ROOT / "results" / "d1b_calibration_uncertainty_v1"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "64_D1B既有评分校准不确定性补充执行协议_v1_2026-08-27.md"
SOURCE = ROOT / "results" / "external_generalization_v3" / "D1-B"
SCORED = SOURCE / "scored.parquet"
SOURCE_REPORT = SOURCE / "report.json"
SOURCE_MANIFEST = SOURCE / "manifest.json"
SOURCE_LOCK = ROOT / "results" / "external_execution_lock_v1" / "D1-B" / "execution_lock.json"
Z95 = 1.959963984540054
SMOKE_MATCHES = 30


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required_input_missing:{path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


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


def audit_source() -> dict[str, Any]:
    report, manifest, lock = read_json(SOURCE_REPORT), read_json(SOURCE_MANIFEST), read_json(SOURCE_LOCK)
    require(report.get("scored_sha256") == digest(SCORED), "scored_hash_mismatch_with_report")
    require(manifest.get("scored_sha256") == digest(SCORED), "scored_hash_mismatch_with_manifest")
    require(manifest.get("report_sha256") == digest(SOURCE_REPORT), "report_hash_mismatch_with_manifest")
    require(report.get("execution_lock_sha256") == digest(SOURCE_LOCK), "execution_lock_hash_mismatch")
    require(lock.get("dataset") == "D1-B" and report.get("dataset") == "D1-B" and manifest.get("dataset") == "D1-B", "source_dataset_mismatch")
    return {"source_status": "passed", "rows": report.get("rows"), "matches": report.get("matches"), "maps": report.get("maps"), "scored_sha256": digest(SCORED), "report_sha256": digest(SOURCE_REPORT), "lock_sha256": digest(SOURCE_LOCK)}


def establish_lock() -> tuple[dict[str, Any], str]:
    audit = audit_source()
    inputs = {"protocol": record(PROTOCOL), "script": record(Path(__file__).resolve()), "scored": record(SCORED), "source_report": record(SOURCE_REPORT), "source_manifest": record(SOURCE_MANIFEST), "source_execution_lock": record(SOURCE_LOCK)}
    expected = {"stage": "d1b_calibration_uncertainty_v1_input_lock", "classification": "user_authorized_uncertainty_completion_of_existing_one_time_D1-B_scoring", "inputs": inputs, "source_hash_audit": audit, "model": "logit P(focal_wins=1) = gamma_0 + gamma_1 logit(p_hat); two_way_CRV1(match_id,focal_player_key)", "inference": "normal_approximation_95_percent_CI; joint_Wald_2df_against_ideal_(0,1)", "scope": "read_existing_scored_parquet_only; no_candidate_label_model_or_raw_match_access_no_rescoring_or_recalibration", "reporting_boundary": "descriptive_uncertainty_for_existing_D1-B_scoring_not_a_new_external_score_or_performance_gate_or_hot_hand_evidence"}
    path = OUT / "checkpoint_01_input_lock.json"
    if path.exists():
        saved = read_json(path)
        require({key: saved.get(key) for key in expected} == expected, "resume_input_lock_mismatch")
        return saved, digest(path)
    write_json_new(path, {**expected, "created_at_utc": now()})
    return expected, digest(path)


def load_frame() -> pd.DataFrame:
    columns = ["duel_id", "match_id", "map_id", "focal_player_key", "focal_wins", "p_hat"]
    frame = pd.read_parquet(SCORED, columns=columns)
    report = read_json(SOURCE_REPORT)
    require(len(frame) == 32786 == report.get("rows"), "row_count_mismatch")
    require(frame["match_id"].nunique() == 100 == report.get("matches") and frame["map_id"].nunique() == 226 == report.get("maps"), "cluster_count_mismatch")
    require(frame["duel_id"].is_unique and frame["focal_player_key"].notna().all(), "identity_mismatch")
    values = frame[["focal_wins", "p_hat"]].to_numpy(dtype=float)
    require(np.isfinite(values).all() and frame["focal_wins"].isin([0, 1]).all() and frame["p_hat"].between(0, 1, inclusive="neither").all(), "value_domain_mismatch")
    return frame


def calibration(frame: pd.DataFrame, details: bool) -> dict[str, Any]:
    p, y = frame["p_hat"].to_numpy(dtype=float), frame["focal_wins"].to_numpy(dtype=float)
    logit = np.log(p / (1 - p))
    design, coefficient = np.column_stack([np.ones(len(frame)), logit]), np.array([0.0, 1.0])
    for iteration in range(100):
        mu = 1 / (1 + np.exp(-np.clip(design @ coefficient, -30, 30)))
        weight = np.maximum(mu * (1 - mu), 1e-10)
        step = np.linalg.solve(design.T @ (weight[:, None] * design), design.T @ (y - mu))
        coefficient += step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    else:
        raise RuntimeError("calibration_nonconvergence")
    mu = 1 / (1 + np.exp(-np.clip(design @ coefficient, -30, 30)))
    bread_inverse = np.linalg.inv(design.T @ (np.maximum(mu * (1 - mu), 1e-10)[:, None] * design))

    def meat(groups: pd.Series) -> tuple[np.ndarray, int]:
        _, codes = np.unique(groups.astype(str).to_numpy(), return_inverse=True)
        summed = np.zeros((int(codes.max()) + 1, design.shape[1]))
        np.add.at(summed, codes, design * (y - mu)[:, None])
        return summed.T @ summed, len(summed)

    match, matches = meat(frame["match_id"])
    player, players = meat(frame["focal_player_key"])
    intersection, intersections = meat(frame["match_id"].astype(str) + "\x1f" + frame["focal_player_key"].astype(str))
    correction = lambda groups: (groups / (groups - 1)) * ((len(frame) - 1) / (len(frame) - design.shape[1]))
    covariance = bread_inverse @ (correction(matches) * match + correction(players) * player - correction(intersections) * intersection) @ bread_inverse
    covariance = (covariance + covariance.T) / 2
    leverage = np.einsum("ij,jk,ik->i", design, bread_inverse, design)
    passed = bool(np.isfinite(covariance).all() and np.linalg.det(covariance) > 0 and matches >= 30 and players >= 30 and intersections >= 30 and leverage.max() <= 0.20)
    result: dict[str, Any] = {"n": int(len(frame)), "iterations": iteration + 1, "match_clusters": matches, "player_clusters": players, "intersection_clusters": intersections, "max_leverage": float(leverage.max()), "estimation_pass": passed}
    if not details or not passed:
        return result
    se = np.sqrt(np.diag(covariance))
    ideal = np.array([0.0, 1.0])
    difference = coefficient - ideal
    wald = float(difference @ np.linalg.inv(covariance) @ difference)
    result.update({"coefficients": {"calibration_intercept": float(coefficient[0]), "calibration_slope": float(coefficient[1])}, "standard_errors": {"calibration_intercept": float(se[0]), "calibration_slope": float(se[1])}, "ci95": {"intercept": [float(coefficient[0] - Z95 * se[0]), float(coefficient[0] + Z95 * se[0])], "slope": [float(coefficient[1] - Z95 * se[1]), float(coefficient[1] + Z95 * se[1])]}, "joint_wald_ideal_calibration": {"statistic": wald, "df": 2, "p_value": float(math.exp(-wald / 2))}, "covariance_matrix": covariance.tolist()})
    return result


def smoke(frame: pd.DataFrame, lock_hash: str) -> str:
    path = OUT / "checkpoint_02_smoke.json"
    matches = sorted(frame["match_id"].astype(str).unique())[:SMOKE_MATCHES]
    result = calibration(frame.loc[frame["match_id"].astype(str).isin(matches)].copy(), details=False)
    expected = {"stage": "d1b_calibration_uncertainty_v1_smoke", "executed_phase": "smoke", "input_lock_sha256": lock_hash, "subset_rule": "30_lexicographically_smallest_match_id_clusters", "effect_output_withheld": True, "D1": {"status": "pass" if result["estimation_pass"] else "fail", "metrics": result}}
    if path.exists():
        saved = read_json(path)
        require(all(saved.get(key) == value for key, value in expected.items()), "resume_smoke_checkpoint_mismatch")
        return digest(path)
    write_json_new(path, {**expected, "created_at_utc": now()})
    require(result["estimation_pass"], "smoke_gate_failed")
    return digest(path)


def full(frame: pd.DataFrame, lock_hash: str, smoke_hash: str) -> dict[str, Any]:
    path = OUT / "checkpoint_03_full_report.json"
    expected = {"stage": "d1b_calibration_uncertainty_v1_full", "input_lock_sha256": lock_hash, "smoke_checkpoint_sha256": smoke_hash, "reporting_status": "descriptive_uncertainty_completion_only"}
    if path.exists():
        saved = read_json(path)
        require(all(saved.get(key) == value for key, value in expected.items()), "resume_full_checkpoint_mismatch")
        return saved
    result = calibration(frame, details=True)
    require(result["estimation_pass"], "full_estimation_failed")
    original = read_json(SOURCE_REPORT)
    for key in ("calibration_intercept", "calibration_slope"):
        require(math.isclose(result["coefficients"][key], float(original[key]), abs_tol=1e-12, rel_tol=0.0), f"point_estimate_reproduction_mismatch:{key}")
    metrics = [*result["coefficients"].values(), *result["standard_errors"].values(), *result["ci95"]["intercept"], *result["ci95"]["slope"], result["joint_wald_ideal_calibration"]["statistic"], result["joint_wald_ideal_calibration"]["p_value"]]
    require(np.isfinite(np.asarray(metrics)).all(), "nonfinite_full_metrics")
    value = {**expected, "created_at_utc": now(), "D0": {"status": "pass", "reason": "sealed_scored_table_identity_and_source_hash_chain_verified"}, "D2": {"status": "pass", "reason": "original_point_estimates_reproduced_with_finite_two_way_CRV1_uncertainty"}, "D3": {"status": "assessed", "reason": "no_new_score_performance_gate_or_model_change"}, "calibration": result}
    write_json_new(path, value)
    return value


def finish(full_checkpoint: dict[str, Any], lock_hash: str, smoke_hash: str) -> None:
    hashes_path = OUT / "file_sha256.json"
    hashes = {str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"} and not any(part.startswith("failed_") for part in path.relative_to(OUT).parts)}
    if hashes_path.exists():
        require(read_json(hashes_path) == hashes, "output_hash_mismatch")
    else:
        write_json_new(hashes_path, hashes)
    manifest = {"stage": "d1b_calibration_uncertainty_v1", "status": "passed", "created_at_utc": now(), "classification": "user_authorized_uncertainty_completion_of_existing_one_time_D1-B_scoring", "input_lock_sha256": lock_hash, "smoke_checkpoint_sha256": smoke_hash, "full_checkpoint_sha256": digest(OUT / "checkpoint_03_full_report.json"), "file_sha256_sha256": digest(hashes_path), "interpretation": "Two-way clustered uncertainty for the logistic calibration fit on the already sealed D1-B score table. This is not a rerun, new external score, model-selection input, performance gate, or hot-hand analysis."}
    path = OUT / "manifest.json"
    if path.exists():
        saved = read_json(path)
        require(saved.get("input_lock_sha256") == lock_hash and saved.get("file_sha256_sha256") == manifest["file_sha256_sha256"], "manifest_mismatch")
    else:
        write_json_new(path, manifest)


def run(phase: str) -> None:
    _, lock_hash = establish_lock()
    frame = load_frame()
    if phase == "smoke":
        smoke(frame, lock_hash)
        print(json.dumps({"status": "smoke_passed", "output_root": str(OUT)}, ensure_ascii=True))
        return
    smoke_path = OUT / "checkpoint_02_smoke.json"
    require(smoke_path.is_file() and read_json(smoke_path).get("input_lock_sha256") == lock_hash and read_json(smoke_path).get("D1", {}).get("status") == "pass", "full_requires_valid_smoke")
    smoke_hash = digest(smoke_path)
    completed = full(frame, lock_hash, smoke_hash)
    finish(completed, lock_hash, smoke_hash)
    print(json.dumps({"status": "full_passed", "output_root": str(OUT)}, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    try:
        run(args.phase)
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True)
        failed = OUT / f"failed_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        failed.mkdir(exist_ok=False)
        write_json_new(failed / "failure_manifest.json", {"stage": "d1b_calibration_uncertainty_v1", "status": "failed", "created_at_utc": now(), "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
