"""Run fixed E1-v2 calibration and map-terminal-tail diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = "E1-v2"
ASSET = ROOT / "results" / "conditional_residual_asset_e1_v1"
SOURCE = ASSET / DATASET / "residual_asset.parquet"
OUT = ROOT / "results" / "e1_calibration_map_terminal_tail_v1"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "30_E1v2校准与地图终止尾部审计执行协议_v1_2026-08-26.md"
METADATA = (ASSET / "manifest.json", ASSET / "file_sha256.json", ASSET / "input_lock.json", ASSET / "label_access_receipt.json", ASSET / "field_dictionary.json", ASSET / DATASET / "structural_temporal_audit.json")
REQUIRED = {"dataset", "physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id", "opponent_player_id", "Y_t", "p_hat", "r_t", "residual_valid", "primary_analysis_eligible"}


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_audit() -> dict[str, Any]:
    expected = read_json(ASSET / "file_sha256.json")
    mismatches = [name for name, expected_hash in expected.items() if not (ASSET / name).is_file() or digest(ASSET / name) != expected_hash]
    manifest, receipt = read_json(ASSET / "manifest.json"), read_json(ASSET / "label_access_receipt.json")
    return {"checked_files": len(expected), "mismatches": mismatches, "manifest_file_sha256_matches": manifest.get("file_sha256_sha256") == digest(ASSET / "file_sha256.json"), "asset_status": manifest.get("status"), "receipt_dataset": receipt.get("dataset"), "probability_role": receipt.get("probability_role")}


def ensure_lock() -> tuple[dict[str, Any], str]:
    audit = source_audit()
    require(not audit["mismatches"] and audit["manifest_file_sha256_matches"], "source_hash_chain_mismatch")
    require(audit["asset_status"] == "passed" and audit["receipt_dataset"] == DATASET and audit["probability_role"] == "fixed_model_external_score_nonconfirmatory", "asset_boundary_mismatch")
    inputs = {str(path): digest(path) for path in (PROTOCOL, Path(__file__).resolve(), SOURCE, *METADATA)}
    path = OUT / "checkpoint_01_input_lock.json"
    if path.exists():
        saved = read_json(path)
        require(saved.get("inputs") == inputs and saved.get("source_hash_audit") == audit, "resume_input_lock_mismatch")
        return saved, digest(path)
    value = {"stage": "e1_calibration_map_terminal_tail_v1_input_lock", "created_at_utc": now(), "classification": "e1_diagnostic_qualification_review", "inputs": inputs, "source_hash_audit": audit, "confirmation_eligibility": "not_adjudicated_by_this_executor", "user_attestation": "The diagnostics were omitted, not result-triggered; no sample selection used E1 results.", "outcome_access": "existing_authorized_residual_asset_only"}
    write_json_new(path, value)
    return value, digest(path)


def eligible(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED - set(frame.columns))
    require(not missing, f"missing_required_columns:{','.join(missing)}")
    require(frame["dataset"].astype(str).eq(DATASET).all(), "dataset_column_mismatch")
    require(not frame["physical_event_id"].duplicated().any(), "physical_event_id_not_unique")
    require(frame["focal_player_id"].notna().all() and frame["opponent_player_id"].notna().all(), "missing_player_id")
    require(np.isfinite(frame[["event_tick", "Y_t", "p_hat", "r_t"]].to_numpy(dtype=float)).all(), "nonfinite_required_values")
    require(frame["Y_t"].isin([0, 1]).all() and frame["p_hat"].between(0, 1, inclusive="neither").all(), "invalid_outcome_or_probability")
    value = frame.loc[frame["primary_analysis_eligible"].astype(bool) & frame["residual_valid"].astype(bool) & frame["p_hat"].between(0.01, 0.99, inclusive="both")].copy()
    require(len(value) > 2, "insufficient_eligible_rows")
    return value


def two_way_covariance(design: np.ndarray, score: np.ndarray, frame: pd.DataFrame, bread_inverse: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    def meat(groups: pd.Series) -> tuple[np.ndarray, int]:
        _, codes = np.unique(groups.astype(str).to_numpy(), return_inverse=True)
        summed = np.zeros((int(codes.max()) + 1, design.shape[1]))
        np.add.at(summed, codes, score)
        return summed.T @ summed, len(summed)
    match, match_count = meat(frame["match_id"])
    player, player_count = meat(frame["focal_player_id"])
    intersection, intersection_count = meat(frame["match_id"].astype(str) + "\x1f" + frame["focal_player_id"].astype(str))
    require(min(match_count, player_count, intersection_count) > 1, "insufficient_cluster_groups")
    correction = lambda groups: (groups / (groups - 1)) * ((len(frame) - 1) / (len(frame) - design.shape[1]))
    covariance = bread_inverse @ (correction(match_count) * match + correction(player_count) * player - correction(intersection_count) * intersection) @ bread_inverse
    return (covariance + covariance.T) / 2, match_count, player_count, intersection_count


def calibration(frame: pd.DataFrame, details: bool) -> dict[str, Any]:
    p, y = frame["p_hat"].to_numpy(dtype=float), frame["Y_t"].to_numpy(dtype=float)
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
        raise RuntimeError("logistic_calibration_nonconvergence")
    mu = 1 / (1 + np.exp(-np.clip(design @ coefficient, -30, 30)))
    bread_inverse = np.linalg.inv(design.T @ (np.maximum(mu * (1 - mu), 1e-10)[:, None] * design))
    covariance, matches, players, intersections = two_way_covariance(design, design * (y - mu)[:, None], frame, bread_inverse)
    leverage = np.einsum("ij,jk,ik->i", design, bread_inverse, design)
    valid = bool(np.isfinite(covariance).all() and np.linalg.det(covariance) > 0 and matches >= 30 and players >= 30 and leverage.max() <= 0.20)
    result: dict[str, Any] = {"n": int(len(frame)), "iterations": iteration + 1, "match_clusters": matches, "player_clusters": players, "intersection_clusters": intersections, "max_leverage": float(leverage.max()), "g3_pass": valid}
    if not valid or not details:
        return result
    normal = NormalDist()
    standard_errors = np.sqrt(np.diag(covariance))
    null = np.array([0.0, 1.0])
    diff = coefficient - null
    wald = float(diff @ np.linalg.inv(covariance) @ diff)
    bins = frame.assign(_bin=pd.qcut(frame["p_hat"], q=10, duplicates="drop")).groupby("_bin", observed=True).agg(n=("p_hat", "size"), mean_p_hat=("p_hat", "mean"), mean_r_t=("r_t", "mean")).reset_index()
    result.update({"coefficients": {"calibration_intercept": float(coefficient[0]), "calibration_slope": float(coefficient[1])}, "standard_errors": {"calibration_intercept": float(standard_errors[0]), "calibration_slope": float(standard_errors[1])}, "ci95": {"intercept": [float(coefficient[0] - 1.95996 * standard_errors[0]), float(coefficient[0] + 1.95996 * standard_errors[0])], "slope": [float(coefficient[1] - 1.95996 * standard_errors[1]), float(coefficient[1] + 1.95996 * standard_errors[1])]}, "joint_wald_null_intercept_0_slope_1": {"statistic": wald, "df": 2, "p_value": float(math.exp(-wald / 2))}, "mean_r_t": float(frame["r_t"].mean()), "covariance": covariance.tolist(), "decile_bins": [{"bin": str(row["_bin"]), "n": int(row["n"]), "mean_p_hat": float(row["mean_p_hat"]), "mean_r_t": float(row["mean_r_t"])} for _, row in bins.iterrows()]})
    return result


def map_tail(frame: pd.DataFrame) -> dict[str, Any]:
    focal = frame[["physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id"]].rename(columns={"focal_player_id": "player_id"})
    opponent = frame[["physical_event_id", "match_id", "map_id", "event_tick", "opponent_player_id"]].rename(columns={"opponent_player_id": "player_id"})
    ledger = pd.concat([focal, opponent], ignore_index=True).drop_duplicates()
    groups = ["match_id", "map_id", "player_id"]
    terminal = ledger["event_tick"].eq(ledger.groupby(groups)["event_tick"].transform("max"))
    terminal_sizes = ledger.loc[terminal].groupby(groups).size()
    return {"physical_events": int(len(frame)), "ledger_rows": int(len(ledger)), "map_player_runs": int(ledger.groupby(groups).ngroups), "terminal_rows": int(terminal.sum()), "terminal_fraction": float(terminal.mean()), "multi_terminal_runs": int((terminal_sizes > 1).sum()), "tail_threshold": 0.05, "tail_gate_pass": bool(float(terminal.mean()) <= 0.05)}


def preflight(lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = OUT / "checkpoint_02_preflight_E1-v2.json"
    source_hash = digest(SOURCE)
    frame = pd.read_parquet(SOURCE)
    data = eligible(frame)
    metrics = {"asset_rows": int(len(frame)), "eligible_rows": int(len(data)), "match_clusters": int(data["match_id"].astype(str).nunique()), "player_clusters": int(data["focal_player_id"].astype(str).nunique()), "source_hash": source_hash}
    if path.exists():
        saved = read_json(path)
        require(saved.get("input_lock_sha256") == lock_hash and saved.get("metrics") == metrics, "preflight_resume_mismatch")
        return data, saved
    value = {"stage": "e1_calibration_map_terminal_tail_v1_preflight", "dataset": DATASET, "created_at_utc": now(), "input_lock_sha256": lock_hash, "metrics": metrics, "G0": {"status": "pass", "reason": "authorized_asset_hash_chain_verified"}, "G1": {"status": "pass", "reason": "required_fields_and_eligibility_verified"}, "G2": {"status": "pass" if metrics["match_clusters"] >= 30 and metrics["player_clusters"] >= 30 else "fail", "reason": "two_way_cluster_support"}}
    write_json_new(path, value)
    return data, value


def smoke(lock_hash: str, frame: pd.DataFrame, preflight_value: dict[str, Any]) -> dict[str, Any]:
    path = OUT / "checkpoint_03_smoke_E1-v2.json"
    source_hash = digest(SOURCE)
    if path.exists():
        saved = read_json(path)
        require(saved.get("input_lock_sha256") == lock_hash and saved.get("source_residual_asset_sha256") == source_hash, "smoke_resume_mismatch")
        return saved
    selected = sorted(frame["match_id"].astype(str).unique())[:30]
    subset = frame.loc[frame["match_id"].astype(str).isin(selected)].copy()
    model = calibration(subset, details=False)
    value = {"stage": "e1_calibration_map_terminal_tail_v1_smoke", "dataset": DATASET, "created_at_utc": now(), "input_lock_sha256": lock_hash, "source_residual_asset_sha256": source_hash, "subset_rule": "30_lexicographically_smallest_match_id_clusters", "subset_rows": int(len(subset)), "G0": preflight_value["G0"], "G1": preflight_value["G1"], "G2": {"status": "pass" if model["match_clusters"] >= 30 and model["player_clusters"] >= 30 else "fail", "reason": "smoke_two_way_cluster_support"}, "G3": {"status": "pass" if model["g3_pass"] else "fail", "reason": "smoke_calibration_model_estimation", "metrics": model}}
    write_json_new(path, value)
    return value


def hashes() -> dict[str, str]:
    return {str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"} and not any(part.startswith("failed_") for part in path.relative_to(OUT).parts)}


def run(phase: str) -> None:
    _, lock_hash = ensure_lock()
    frame, preflight_value = preflight(lock_hash)
    require(all(preflight_value[name]["status"] == "pass" for name in ("G0", "G1", "G2")), "preflight_gate_failed")
    smoke_value = smoke(lock_hash, frame, preflight_value)
    require(all(smoke_value[name]["status"] == "pass" for name in ("G0", "G1", "G2", "G3")), "smoke_gate_failed")
    if phase == "smoke":
        print(json.dumps({"status": "smoke_passed", "output_root": str(OUT)}, ensure_ascii=True))
        return
    result_path = OUT / "checkpoint_04_diagnostics_E1-v2.json"
    source_hash = digest(SOURCE)
    if result_path.exists():
        saved = read_json(result_path)
        require(saved.get("input_lock_sha256") == lock_hash and saved.get("source_residual_asset_sha256") == source_hash, "diagnostic_resume_mismatch")
        calibration_result, tail_result = saved["calibration"], saved["map_terminal_tail"]
    else:
        calibration_result, tail_result = calibration(frame, details=True), map_tail(frame)
        write_json_new(result_path, {"stage": "e1_calibration_map_terminal_tail_v1_diagnostics", "dataset": DATASET, "created_at_utc": now(), "input_lock_sha256": lock_hash, "source_residual_asset_sha256": source_hash, "calibration": calibration_result, "map_terminal_tail": tail_result})
    require(calibration_result["g3_pass"], "calibration_model_gate_failed")
    gates = {"G0": {**preflight_value["G0"], "metrics": {"source_hash_audit": source_audit()}}, "G1": preflight_value["G1"], "G2": preflight_value["G2"], "G3": {"status": "pass", "reason": "two_way_crv1_calibration_model_estimated", "metrics": calibration_result}, "G4": {"status": "assessed", "reason": "fixed_model_external_e1_calibration_diagnostic", "metrics": calibration_result}, "G5": {"status": "map_terminal_natural_endpoint_supported" if tail_result["tail_gate_pass"] else "tail_threshold_exceeded", "reason": "map_terminal_structure_only", "metrics": tail_result, "other_opportunity_processes": "not_assessed"}, "overall": {"diagnostic_release": "pass", "confirmation_eligibility": "not_adjudicated_by_this_executor", "interpretation_release": "external_fixed_model_calibration_and_map_terminal_structure_only", "forbidden_interpretations": ["general_calibration_proof", "complete_opportunity_model", "state_effect", "causal_effect"]}}
    gates_path = OUT / "checkpoint_05_gate_audit.json"
    if gates_path.exists():
        require(read_json(gates_path).get("input_lock_sha256") == lock_hash, "gate_resume_mismatch")
    else:
        write_json_new(gates_path, {"stage": "e1_calibration_map_terminal_tail_v1_gate_audit", "created_at_utc": now(), "input_lock_sha256": lock_hash, "gates": gates})
    hashes_path = OUT / "file_sha256.json"
    current = hashes()
    if hashes_path.exists():
        require(read_json(hashes_path) == current, "output_hash_mismatch")
    else:
        write_json_new(hashes_path, current)
    manifest_path = OUT / "manifest.json"
    value = {"stage": "e1_calibration_map_terminal_tail_v1", "status": "passed", "created_at_utc": now(), "classification": "e1_diagnostic_qualification_review", "input_lock_sha256": lock_hash, "gate_statuses": {name: gate["status"] for name, gate in gates.items() if name.startswith("G")}, "confirmation_eligibility": "not_adjudicated_by_this_executor", "file_sha256_sha256": digest(hashes_path)}
    if manifest_path.exists():
        saved = read_json(manifest_path)
        require(saved.get("input_lock_sha256") == lock_hash and saved.get("file_sha256_sha256") == digest(hashes_path), "manifest_resume_mismatch")
    else:
        write_json_new(manifest_path, value)
    print(json.dumps({"status": "passed", "manifest": str(manifest_path)}, ensure_ascii=True))


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
        write_json_new(failed / "failure_manifest.json", {"stage": "e1_calibration_map_terminal_tail_v1", "status": "failed", "created_at_utc": now(), "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
