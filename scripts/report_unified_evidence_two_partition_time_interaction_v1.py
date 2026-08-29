"""Versioned descriptive completion of the fixed E-A/E-B time-interaction model."""

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
OUT = ROOT / "results" / "unified_evidence_two_partition_time_interaction_v1"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "63_统一证据池两分区时间交互完整报告执行协议_v1_2026-08-27.md"
BASE_OUT = ROOT / "results" / "unified_evidence_two_partition_confirmation_v1"
BASE_PROTOCOL = ROOT / "docs" / "修订与补丁" / "58_统一证据池两分区最终确认关联执行锁_v1_2026-08-27.md"
BASE_EXECUTOR = ROOT / "scripts" / "analyze_unified_evidence_two_partition_confirmation_v1.py"
BASE_MODEL = ROOT / "scripts" / "analyze_exponential_history_state_sequence_v1.py"
PARTITIONS = ("E-A", "E-B")
REFERENCE = {"mu_z": 11.538069409706317, "mu_S": 0.34010149087253094, "sigma_S_ddof1": 0.6490311893067302}
Z95 = 1.959963984540054
TICK_RATE = 64.0


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


def base_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {
        "reporting_protocol": PROTOCOL,
        "base_protocol": BASE_PROTOCOL,
        "reporting_script": Path(__file__).resolve(),
        "base_executor": BASE_EXECUTOR,
        "base_model": BASE_MODEL,
        "base_input_lock": BASE_OUT / "checkpoint_01_input_lock.json",
        "base_smoke": BASE_OUT / "checkpoint_02_smoke.json",
        "base_models": BASE_OUT / "checkpoint_03_models.json",
        "base_gate_audit": BASE_OUT / "checkpoint_04_gate_audit.json",
        "base_summary": BASE_OUT / "checkpoint_05_summary.json",
        "base_file_sha256": BASE_OUT / "file_sha256.json",
        "base_manifest": BASE_OUT / "manifest.json",
    }
    for partition in PARTITIONS:
        paths[f"{partition}_state_matrix"] = BASE_OUT / "full" / partition / "state_matrix.parquet"
        paths[f"{partition}_state_audit"] = BASE_OUT / "full" / partition / "state_audit.json"
    return paths


def audit_base() -> dict[str, Any]:
    hashes = read_json(BASE_OUT / "file_sha256.json")
    mismatches = []
    for name, expected in hashes.items():
        path = BASE_OUT / name
        if not path.is_file() or digest(path) != expected:
            mismatches.append(name)
    manifest = read_json(BASE_OUT / "manifest.json")
    require(not mismatches, "base_output_hash_mismatch")
    require(manifest.get("status") == "passed", "base_manifest_not_passed")
    require(manifest.get("file_sha256_sha256") == digest(BASE_OUT / "file_sha256.json"), "base_manifest_hash_chain_mismatch")
    return {"checked_files": len(hashes), "mismatches": mismatches, "manifest_status": manifest.get("status")}


def establish_lock() -> tuple[dict[str, Any], str]:
    audit = audit_base()
    inputs = {name: record(path) for name, path in base_paths().items()}
    expected = {
        "stage": "unified_evidence_two_partition_time_interaction_v1_input_lock",
        "classification": "user_authorized_descriptive_completion_of_existing_fixed_model",
        "inputs": inputs,
        "base_output_hash_audit": audit,
        "partitions": list(PARTITIONS),
        "reference": REFERENCE,
        "model": "e_t ~ 1 + z + S_star + S_star:z; OLS two_way_CRV1(match_id, player_id)",
        "time_support_quantiles": [0.10, 0.50, 0.90],
        "quantile_method": "numpy_linear",
        "inference": "two_way_CRV1_with_existing_finite_sample_corrections; normal_approximation_95_percent_CI",
        "scope": "read_existing_full_state_matrices_only; no_raw_data_label_access_rescoring_state_or_model_change",
        "reporting_boundary": "beta_primary_result_and_existing_Holm_family_unchanged; kappa_and_time_point_slopes_descriptive_only_no_p_values_or_new_gate",
    }
    path = OUT / "checkpoint_01_input_lock.json"
    if path.exists():
        saved = read_json(path)
        require({key: saved.get(key) for key in expected} == expected, "resume_input_lock_mismatch")
        return saved, digest(path)
    write_json_new(path, {**expected, "created_at_utc": now()})
    return expected, digest(path)


def load_state(partition: str, lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrix = BASE_OUT / "full" / partition / "state_matrix.parquet"
    audit = read_json(BASE_OUT / "full" / partition / "state_audit.json")
    original = read_json(BASE_OUT / "checkpoint_03_models.json")
    require(audit.get("state_matrix_sha256") == digest(matrix), f"state_audit_hash_mismatch:{partition}")
    require(original.get("state_hashes", {}).get(partition) == digest(matrix), f"base_model_hash_mismatch:{partition}")
    require(audit.get("input_lock_sha256") == read_json(BASE_OUT / "checkpoint_01_input_lock.json").get("sha256", audit.get("input_lock_sha256")) or audit.get("input_lock_sha256") == original.get("input_lock_sha256"), f"base_state_lock_mismatch:{partition}")
    frame = pd.read_parquet(matrix)
    required = {"match_id", "player_id", "e_t", "S_h60", "d_tick"}
    require(required.issubset(frame.columns), f"state_columns_missing:{partition}")
    values = frame[["e_t", "S_h60", "d_tick"]].to_numpy(dtype=float)
    require(len(frame) == original["models"][partition]["n"] and np.isfinite(values).all() and (frame["d_tick"] >= 1).all(), f"state_values_mismatch:{partition}")
    return frame, audit


def fit(frame: pd.DataFrame) -> dict[str, Any]:
    n = len(frame)
    z = np.log2(1.0 + frame["d_tick"].to_numpy(dtype=float)) - REFERENCE["mu_z"]
    s = (frame["S_h60"].to_numpy(dtype=float) - REFERENCE["mu_S"]) / REFERENCE["sigma_S_ddof1"]
    design = np.column_stack([np.ones(n), z, s, s * z])
    response = frame["e_t"].to_numpy(dtype=float)
    k = design.shape[1]
    require(n > k and np.linalg.matrix_rank(design) == k, "rank_deficient_design")
    inverse = np.linalg.inv(design.T @ design)
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
    intersection_meat, intersections = meat(frame["match_id"].astype(str) + "\x1f" + frame["player_id"].astype(str))
    correction = lambda groups: (groups / (groups - 1)) * ((n - 1) / (n - k))
    covariance = inverse @ (correction(matches) * match_meat + correction(players) * player_meat - correction(intersections) * intersection_meat) @ inverse
    covariance = (covariance + covariance.T) / 2
    return {"coefficient": coefficient, "covariance": covariance, "n": n, "match_clusters": matches, "player_clusters": players, "intersection_clusters": intersections, "max_leverage": float(leverage.max())}


def smoke(states: dict[str, pd.DataFrame], lock_hash: str) -> str:
    path = OUT / "checkpoint_02_smoke.json"
    metrics: dict[str, Any] = {}
    for partition, frame in states.items():
        result = fit(frame)
        covariance = result["covariance"]
        passed = bool(np.isfinite(covariance).all() and (np.diag(covariance) > 0).all() and result["match_clusters"] >= 30 and result["player_clusters"] >= 30 and result["intersection_clusters"] >= 30 and result["max_leverage"] <= 0.20)
        metrics[partition] = {key: result[key] for key in ("n", "match_clusters", "player_clusters", "intersection_clusters", "max_leverage")}
        metrics[partition]["estimable"] = passed
    expected = {"stage": "unified_evidence_two_partition_time_interaction_v1_smoke", "executed_phase": "smoke", "input_lock_sha256": lock_hash, "source": "existing_full_E-A_E-B_state_matrices", "effect_output_withheld": True, "T1": {"status": "pass" if all(value["estimable"] for value in metrics.values()) else "fail", "metrics": metrics}}
    if path.exists():
        saved = read_json(path)
        require(all(saved.get(key) == value for key, value in expected.items()), "resume_smoke_checkpoint_mismatch")
        return digest(path)
    write_json_new(path, {**expected, "created_at_utc": now()})
    require(expected["T1"]["status"] == "pass", "smoke_gate_failed")
    return digest(path)


def complete_partition(partition: str, frame: pd.DataFrame, result: dict[str, Any]) -> dict[str, Any]:
    original = read_json(BASE_OUT / "checkpoint_03_models.json")["models"][partition]
    names = ["alpha", "a_time", "beta_state", "kappa_state_time"]
    coefficient, covariance = result["coefficient"], result["covariance"]
    for index, name in enumerate(names):
        require(math.isclose(float(coefficient[index]), float(original["coefficients"][name]), abs_tol=1e-12, rel_tol=0.0), f"coefficient_reproduction_mismatch:{partition}:{name}")
    beta_se = math.sqrt(float(covariance[2, 2]))
    require(math.isclose(beta_se, float(original["beta"]["standard_error"]), abs_tol=1e-12, rel_tol=0.0), f"beta_se_reproduction_mismatch:{partition}")
    table = {}
    for index, name in enumerate(names):
        se = math.sqrt(float(covariance[index, index]))
        table[name] = {"estimate": float(coefficient[index]), "standard_error": se, "ci95_low": float(coefficient[index] - Z95 * se), "ci95_high": float(coefficient[index] + Z95 * se)}
    d_tick = frame["d_tick"].to_numpy(dtype=float)
    locations = {}
    for label, quantile in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
        tick = float(np.quantile(d_tick, quantile, method="linear"))
        z = float(math.log2(1.0 + tick) - REFERENCE["mu_z"])
        slope = float(coefficient[2] + coefficient[3] * z)
        variance = float(covariance[2, 2] + 2 * z * covariance[2, 3] + z * z * covariance[3, 3])
        require(math.isfinite(variance) and variance > 0, f"invalid_slope_variance:{partition}:{label}")
        se = math.sqrt(variance)
        locations[label] = {"quantile": quantile, "d_tick": tick, "seconds": tick / TICK_RATE, "z": z, "state_slope": slope, "standard_error": se, "ci95_low": float(slope - Z95 * se), "ci95_high": float(slope + Z95 * se)}
    return {"rows": result["n"], "match_clusters": result["match_clusters"], "player_clusters": result["player_clusters"], "intersection_clusters": result["intersection_clusters"], "max_leverage": result["max_leverage"], "coefficients": table, "state_slope_at_time_support": locations, "covariance_matrix": covariance.tolist(), "reproduces_base_beta": {"estimate": float(coefficient[2]), "standard_error": beta_se, "ci95_low": float(coefficient[2] - Z95 * beta_se), "ci95_high": float(coefficient[2] + Z95 * beta_se), "holm_p_value_across_partitions_from_base": float(original["beta"]["holm_p_value_across_partitions"])}}


def full(states: dict[str, pd.DataFrame], lock_hash: str, smoke_hash: str) -> dict[str, Any]:
    path = OUT / "checkpoint_03_full_report.json"
    expected = {"stage": "unified_evidence_two_partition_time_interaction_v1_full", "input_lock_sha256": lock_hash, "smoke_checkpoint_sha256": smoke_hash, "reporting_status": "descriptive_completion_only"}
    if path.exists():
        saved = read_json(path)
        require(all(saved.get(key) == value for key, value in expected.items()), "resume_full_checkpoint_mismatch")
        return saved
    partitions = {name: complete_partition(name, frame, fit(frame)) for name, frame in states.items()}
    finite = all(np.isfinite(np.asarray([item for partition in partitions.values() for row in partition["coefficients"].values() for item in row.values()], dtype=float)).all() for _ in [0])
    require(finite, "nonfinite_full_metrics")
    value = {**expected, "created_at_utc": now(), "T0": {"status": "pass", "reason": "base_hash_chain_and_state_identity_verified"}, "T2": {"status": "pass", "reason": "fixed_coefficients_and_beta_SE_reproduced_with_finite_full_covariance"}, "T3": {"status": "assessed", "reason": "interaction_and_time_point_slopes_are_descriptive_no_p_values_or_new_decision_rule"}, "partitions": partitions}
    write_json_new(path, value)
    return value


def finish(full_checkpoint: dict[str, Any], lock_hash: str, smoke_hash: str) -> None:
    hashes_path = OUT / "file_sha256.json"
    hashes = {str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"} and not any(part.startswith("failed_") for part in path.relative_to(OUT).parts)}
    if hashes_path.exists():
        require(read_json(hashes_path) == hashes, "output_hash_mismatch")
    else:
        write_json_new(hashes_path, hashes)
    manifest = {"stage": "unified_evidence_two_partition_time_interaction_v1", "status": "passed", "created_at_utc": now(), "classification": "user_authorized_descriptive_completion_of_existing_fixed_model", "input_lock_sha256": lock_hash, "smoke_checkpoint_sha256": smoke_hash, "full_checkpoint_sha256": digest(OUT / "checkpoint_03_full_report.json"), "file_sha256_sha256": digest(hashes_path), "interpretation": "Complete coefficient and representative-time slope reporting for the already fixed E-A/E-B model. It changes neither the beta primary test, its Holm family, nor the confirmation record; interaction quantities are descriptive only."}
    path = OUT / "manifest.json"
    if path.exists():
        saved = read_json(path)
        require(saved.get("input_lock_sha256") == lock_hash and saved.get("file_sha256_sha256") == manifest["file_sha256_sha256"], "manifest_mismatch")
    else:
        write_json_new(path, manifest)


def run(phase: str) -> None:
    _, lock_hash = establish_lock()
    states = {partition: load_state(partition, lock_hash)[0] for partition in PARTITIONS}
    if phase == "smoke":
        smoke(states, lock_hash)
        print(json.dumps({"status": "smoke_passed", "output_root": str(OUT)}, ensure_ascii=True))
        return
    smoke_path = OUT / "checkpoint_02_smoke.json"
    require(smoke_path.is_file() and read_json(smoke_path).get("input_lock_sha256") == lock_hash and read_json(smoke_path).get("T1", {}).get("status") == "pass", "full_requires_valid_smoke")
    smoke_hash = digest(smoke_path)
    completed = full(states, lock_hash, smoke_hash)
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
        write_json_new(failed / "failure_manifest.json", {"stage": "unified_evidence_two_partition_time_interaction_v1", "status": "failed", "created_at_utc": now(), "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
