"""Audit fixed p_hat calibration and loss on the E-A/E-B final-analysis partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "unified_evidence_two_partition_p_hat_calibration_v1"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "62_统一证据池两分区p_hat同样本诊断执行协议_v1_2026-08-27.md"
FINAL_RECORD = ROOT / "docs" / "修订与补丁" / "61_统一证据池两分区最终确认性证据资格登记_v1_2026-08-27.md"
ASSIGNMENT_ROOT = ROOT / "data" / "unified_evidence_two_partitions_v3"
ASSIGNMENTS = ASSIGNMENT_ROOT / "match_assignments.parquet"
ASSIGNMENT_MANIFEST = ASSIGNMENT_ROOT / "manifest.json"
E1_ASSET_ROOT = ROOT / "results" / "conditional_residual_asset_e1_v1"
E1_ASSET = E1_ASSET_ROOT / "E1-v2" / "residual_asset.parquet"
E1_MANIFEST = E1_ASSET_ROOT / "manifest.json"
E1_HASHES = E1_ASSET_ROOT / "file_sha256.json"
PARTITIONS = ("E-A", "E-B")
SMOKE_MATCHES = 30
BOOTSTRAP_DRAWS = 2000
BASE_SEED = 20260827
REQUIRED = {"dataset", "physical_event_id", "match_id", "focal_player_id", "Y_t", "p_hat", "r_t", "residual_valid", "primary_analysis_eligible"}


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


def write_parquet_new(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def source_hash_audit() -> dict[str, Any]:
    expected = read_json(E1_HASHES)
    mismatches = [name for name, expected_hash in expected.items() if not (E1_ASSET_ROOT / name).is_file() or digest(E1_ASSET_ROOT / name) != expected_hash]
    manifest = read_json(E1_MANIFEST)
    return {"checked_files": len(expected), "mismatches": mismatches, "manifest_file_sha256_matches": manifest.get("file_sha256_sha256") == digest(E1_HASHES), "asset_status": manifest.get("status"), "asset_probability_role": manifest.get("partition", {}).get("probability_role")}


def establish_lock() -> tuple[dict[str, Any], str]:
    audit = source_hash_audit()
    if audit["mismatches"] or not audit["manifest_file_sha256_matches"] or audit["asset_status"] != "passed":
        raise RuntimeError("e1_asset_hash_chain_mismatch")
    inputs = {name: record(path) for name, path in {
        "execution_protocol": PROTOCOL,
        "final_evidence_record": FINAL_RECORD,
        "script": Path(__file__).resolve(),
        "assignment_manifest": ASSIGNMENT_MANIFEST,
        "assignments": ASSIGNMENTS,
        "e1_residual_asset": E1_ASSET,
        "e1_manifest": E1_MANIFEST,
        "e1_file_sha256": E1_HASHES,
    }.items()}
    expected = {
        "stage": "unified_evidence_two_partition_p_hat_calibration_v1_input_lock",
        "classification": "user_authorized_result_informed_same_sample_fixed_p_hat_diagnostic",
        "inputs": inputs,
        "source_hash_audit": audit,
        "partitions": list(PARTITIONS),
        "analysis_unit": "unique_physical_engagement_with_original_focal_outcome",
        "eligibility": "dataset=E1-v2; primary_analysis_eligible; residual_valid; 0.01<=p_hat<=0.99",
        "loss": {"aggregation": "match_equal", "metrics": ["log_loss", "brier"], "bootstrap_draws": BOOTSTRAP_DRAWS, "seed_rule": "20260827_plus_partition_ordinal"},
        "calibration": "logit P(Y=1)=gamma_0+gamma_1 logit(p_hat); two_way_CRV1(match_id,focal_player_id); ten_equal_frequency_bins",
        "outcome_boundary": "existing_E1-v2_outcomes_and_residuals_only; no_D1-C_access; no_rescoring_or_refitting",
        "confirmation_boundary": "supplemental_same_sample_diagnostic_only; does_not_amend_final_evidence_record",
    }
    path = OUT / "checkpoint_01_input_lock.json"
    if path.exists():
        saved = read_json(path)
        if {key: saved.get(key) for key in expected} != expected:
            raise RuntimeError("resume_input_lock_mismatch")
        return saved, digest(path)
    write_json_new(path, {**expected, "created_at_utc": now()})
    return expected, digest(path)


def load_assignments() -> dict[str, list[str]]:
    manifest = read_json(ASSIGNMENT_MANIFEST)
    if manifest.get("files", {}).get("match_assignments.parquet") != digest(ASSIGNMENTS):
        raise RuntimeError("assignment_hash_mismatch")
    frame = pd.read_parquet(ASSIGNMENTS)
    required = {"partition", "match_id", "coverage_stratum"}
    if not required.issubset(frame.columns):
        raise RuntimeError("assignment_columns_missing")
    frame["match_id"] = frame["match_id"].astype(str)
    if frame["match_id"].duplicated().any():
        raise RuntimeError("assignment_match_not_unique")
    output: dict[str, list[str]] = {}
    for partition in PARTITIONS:
        subset = frame.loc[frame["partition"].eq(partition)].copy()
        if len(subset) != 100 or subset["coverage_stratum"].value_counts().to_dict() != {"E1-only": 70, "R1-covered": 30}:
            raise RuntimeError(f"assignment_quota_mismatch:{partition}")
        output[partition] = sorted(subset["match_id"].tolist())
    if set(output["E-A"]) & set(output["E-B"]):
        raise RuntimeError("partition_overlap")
    return output


def load_asset() -> pd.DataFrame:
    frame = pd.read_parquet(E1_ASSET, columns=sorted(REQUIRED))
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise RuntimeError("asset_columns_missing:" + ",".join(sorted(missing)))
    frame["match_id"] = frame["match_id"].astype(str)
    if frame["dataset"].astype(str).ne("E1-v2").any() or frame["physical_event_id"].duplicated().any():
        raise RuntimeError("asset_identity_mismatch")
    return frame


def eligible(asset: pd.DataFrame, match_ids: list[str], partition: str) -> pd.DataFrame:
    frame = asset.loc[asset["match_id"].isin(match_ids)].copy()
    if frame["match_id"].nunique() != len(match_ids):
        raise RuntimeError(f"asset_match_coverage_mismatch:{partition}")
    frame = frame.loc[frame["primary_analysis_eligible"].astype(bool) & frame["residual_valid"].astype(bool) & frame["p_hat"].between(0.01, 0.99, inclusive="both")].copy()
    if frame.empty or frame["physical_event_id"].duplicated().any() or frame["focal_player_id"].isna().any():
        raise RuntimeError(f"eligible_identity_mismatch:{partition}")
    numeric = frame[["Y_t", "p_hat", "r_t"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not frame["Y_t"].isin([0, 1]).all() or not frame["p_hat"].between(0.01, 0.99, inclusive="both").all():
        raise RuntimeError(f"eligible_value_mismatch:{partition}")
    return frame


def two_way_covariance(design: np.ndarray, score: np.ndarray, frame: pd.DataFrame, bread_inverse: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    def meat(groups: pd.Series) -> tuple[np.ndarray, int]:
        _, codes = np.unique(groups.astype(str).to_numpy(), return_inverse=True)
        summed = np.zeros((int(codes.max()) + 1, design.shape[1]))
        np.add.at(summed, codes, score)
        return summed.T @ summed, len(summed)
    match, match_count = meat(frame["match_id"])
    player, player_count = meat(frame["focal_player_id"])
    intersection, intersection_count = meat(frame["match_id"].astype(str) + "\x1f" + frame["focal_player_id"].astype(str))
    if min(match_count, player_count, intersection_count) <= 1:
        raise RuntimeError("insufficient_cluster_groups")
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
        raise RuntimeError("calibration_nonconvergence")
    mu = 1 / (1 + np.exp(-np.clip(design @ coefficient, -30, 30)))
    bread_inverse = np.linalg.inv(design.T @ (np.maximum(mu * (1 - mu), 1e-10)[:, None] * design))
    covariance, matches, players, intersections = two_way_covariance(design, design * (y - mu)[:, None], frame, bread_inverse)
    leverage = np.einsum("ij,jk,ik->i", design, bread_inverse, design)
    model_pass = bool(np.isfinite(covariance).all() and np.linalg.det(covariance) > 0 and matches >= 30 and players >= 30 and leverage.max() <= 0.20)
    result: dict[str, Any] = {"n": int(len(frame)), "iterations": iteration + 1, "match_clusters": matches, "player_clusters": players, "intersection_clusters": intersections, "max_leverage": float(leverage.max()), "estimation_pass": model_pass}
    if not details or not model_pass:
        return result
    standard_errors = np.sqrt(np.diag(covariance))
    diff = coefficient - np.array([0.0, 1.0])
    wald = float(diff @ np.linalg.inv(covariance) @ diff)
    bins = frame.assign(_bin=pd.qcut(frame["p_hat"], q=10, duplicates="drop")).groupby("_bin", observed=True).agg(n=("p_hat", "size"), mean_p_hat=("p_hat", "mean"), empirical_win_rate=("Y_t", "mean")).reset_index()
    ece = float(np.sum((bins["n"] / len(frame)) * np.abs(bins["mean_p_hat"] - bins["empirical_win_rate"])))
    result.update({
        "coefficients": {"calibration_intercept": float(coefficient[0]), "calibration_slope": float(coefficient[1])},
        "standard_errors": {"calibration_intercept": float(standard_errors[0]), "calibration_slope": float(standard_errors[1])},
        "ci95": {"intercept": [float(coefficient[0] - 1.95996 * standard_errors[0]), float(coefficient[0] + 1.95996 * standard_errors[0])], "slope": [float(coefficient[1] - 1.95996 * standard_errors[1]), float(coefficient[1] + 1.95996 * standard_errors[1])]},
        "joint_wald_ideal_calibration": {"statistic": wald, "df": 2, "p_value": float(math.exp(-wald / 2))},
        "ece_ten_equal_frequency_bins": ece,
        "bins": [{"bin": str(row["_bin"]), "n": int(row["n"]), "mean_p_hat": float(row["mean_p_hat"]), "empirical_win_rate": float(row["empirical_win_rate"])} for _, row in bins.iterrows()],
    })
    return result


def match_losses(frame: pd.DataFrame) -> pd.DataFrame:
    p, y = frame["p_hat"].to_numpy(dtype=float), frame["Y_t"].to_numpy(dtype=float)
    values = frame.assign(_log_loss=-(y * np.log(p) + (1 - y) * np.log1p(-p)), _brier=(p - y) ** 2)
    output = values.groupby("match_id", as_index=False).agg(rows=("Y_t", "size"), log_loss=("_log_loss", "mean"), brier=("_brier", "mean"))
    if not np.isfinite(output[["log_loss", "brier"]].to_numpy()).all():
        raise RuntimeError("nonfinite_match_losses")
    return output.sort_values("match_id").reset_index(drop=True)


def bootstrap(losses: pd.DataFrame, partition_ordinal: int) -> tuple[dict[str, Any], pd.DataFrame]:
    if len(losses) != 100:
        raise RuntimeError("bootstrap_requires_100_matches")
    values = losses[["log_loss", "brier"]].to_numpy(dtype=float)
    rng = np.random.default_rng(BASE_SEED + partition_ordinal)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))].mean(axis=1)
    output = pd.DataFrame({"draw": np.arange(1, BOOTSTRAP_DRAWS + 1), "log_loss": draws[:, 0], "brier": draws[:, 1]})
    summary = {name: {"estimate": float(values[:, index].mean()), "ci95_percentile": [float(np.quantile(draws[:, index], 0.025)), float(np.quantile(draws[:, index], 0.975))]} for index, name in enumerate(("log_loss", "brier"))}
    return summary, output


def smoke(asset: pd.DataFrame, assignments: dict[str, list[str]], lock_hash: str) -> tuple[dict[str, Any], str]:
    path = OUT / "checkpoint_02_smoke.json"
    metrics: dict[str, Any] = {}
    for partition in PARTITIONS:
        subset = eligible(asset, assignments[partition][:SMOKE_MATCHES], partition)
        calibration_result = calibration(subset, details=False)
        metrics[partition] = {"rows": int(len(subset)), "matches": int(subset["match_id"].nunique()), "calibration_estimation": calibration_result}
    expected = {"stage": "unified_evidence_two_partition_p_hat_calibration_v1_smoke", "executed_phase": "smoke", "input_lock_sha256": lock_hash, "subset_rule": "30_lexicographically_smallest_match_id_clusters_within_each_partition", "effect_output_withheld": True, "P2": {"status": "pass" if all(item["matches"] == SMOKE_MATCHES and item["calibration_estimation"]["estimation_pass"] for item in metrics.values()) else "fail", "metrics": metrics}}
    if path.exists():
        saved = read_json(path)
        if all(saved.get(key) == value for key, value in expected.items()):
            return saved, digest(path)
        raise RuntimeError("resume_smoke_checkpoint_mismatch")
    write_json_new(path, {**expected, "created_at_utc": now()})
    if expected["P2"]["status"] != "pass":
        raise RuntimeError("smoke_gate_failed")
    return expected, digest(path)


def require_smoke(lock_hash: str) -> str:
    path = OUT / "checkpoint_02_smoke.json"
    if not path.exists():
        raise RuntimeError("full_requires_smoke")
    value = read_json(path)
    if value.get("input_lock_sha256") != lock_hash or value.get("executed_phase") != "smoke" or value.get("P2", {}).get("status") != "pass":
        raise RuntimeError("smoke_checkpoint_invalid")
    return digest(path)


def full(asset: pd.DataFrame, assignments: dict[str, list[str]], lock_hash: str, smoke_hash: str) -> dict[str, Any]:
    path = OUT / "checkpoint_03_metrics.json"
    expected_prefix = {"stage": "unified_evidence_two_partition_p_hat_calibration_v1_metrics", "input_lock_sha256": lock_hash, "smoke_checkpoint_sha256": smoke_hash}
    if path.exists():
        saved = read_json(path)
        if all(saved.get(key) == value for key, value in expected_prefix.items()):
            return saved
        raise RuntimeError("resume_metrics_checkpoint_mismatch")
    metrics: dict[str, Any] = {}
    bootstrap_frames: list[pd.DataFrame] = []
    for ordinal, partition in enumerate(PARTITIONS):
        subset = eligible(asset, assignments[partition], partition)
        losses = match_losses(subset)
        loss_summary, draws = bootstrap(losses, ordinal)
        calibration_result = calibration(subset, details=True)
        if not calibration_result["estimation_pass"]:
            raise RuntimeError(f"full_calibration_gate_failed:{partition}")
        draws.insert(0, "partition", partition)
        bootstrap_frames.append(draws)
        metrics[partition] = {"rows": int(len(subset)), "physical_events": int(subset["physical_event_id"].nunique()), "matches": int(subset["match_id"].nunique()), "focal_player_clusters": int(subset["focal_player_id"].astype(str).nunique()), "p_hat_range": [float(subset["p_hat"].min()), float(subset["p_hat"].max())], "match_equal_losses": loss_summary, "calibration": calibration_result}
    bootstrap_path = OUT / "match_bootstrap_metrics.parquet"
    write_parquet_new(bootstrap_path, pd.concat(bootstrap_frames, ignore_index=True))
    value = {**expected_prefix, "created_at_utc": now(), "bootstrap_metrics_sha256": digest(bootstrap_path), "metrics": metrics}
    write_json_new(path, value)
    return value


def output_hashes() -> dict[str, str]:
    return {str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"} and not any(part.startswith("failed_") for part in path.relative_to(OUT).parts)}


def finish(metrics_checkpoint: dict[str, Any], lock_hash: str, smoke_hash: str) -> None:
    metrics = metrics_checkpoint["metrics"]
    finite = True
    for partition in PARTITIONS:
        calibration_result = metrics[partition]["calibration"]
        values = [*metrics[partition]["p_hat_range"], *[value for summary in metrics[partition]["match_equal_losses"].values() for value in [summary["estimate"], *summary["ci95_percentile"]]], *calibration_result["coefficients"].values(), *calibration_result["standard_errors"].values(), *calibration_result["ci95"]["intercept"], *calibration_result["ci95"]["slope"], calibration_result["joint_wald_ideal_calibration"]["statistic"], calibration_result["joint_wald_ideal_calibration"]["p_value"], calibration_result["ece_ten_equal_frequency_bins"]]
        finite = finite and bool(np.isfinite(np.asarray(values, dtype=float)).all()) and calibration_result["estimation_pass"] and len(calibration_result["bins"]) == 10
    gates = {
        "P0_input_hash_chain": {"status": "pass", "reason": "locked_E1_asset_and_partition_hash_chain_verified"},
        "P1_rows": {"status": "pass" if all(metrics[p]["matches"] == 100 and metrics[p]["rows"] == metrics[p]["physical_events"] for p in PARTITIONS) else "fail", "metrics": {p: {key: metrics[p][key] for key in ("rows", "physical_events", "matches", "focal_player_clusters", "p_hat_range")} for p in PARTITIONS}},
        "P2_smoke": {"status": "pass", "smoke_checkpoint_sha256": smoke_hash},
        "P3_finite_metrics": {"status": "pass" if finite else "fail"},
        "P4_diagnostic_disclosure": {"status": "assessed", "reason": "no_post_hoc_performance_threshold_or_main_test_gate"},
        "P5_evidence_boundary": {"status": "assessed", "reason": "same_sample_E1-v2_diagnostic_with_prior_outcome_access_disclosed; final_evidence_record_unchanged"},
    }
    gate_path = OUT / "checkpoint_04_gate_audit.json"
    if gate_path.exists():
        if read_json(gate_path).get("input_lock_sha256") != lock_hash:
            raise RuntimeError("resume_gate_checkpoint_mismatch")
    else:
        write_json_new(gate_path, {"stage": "unified_evidence_two_partition_p_hat_calibration_v1_gate_audit", "created_at_utc": now(), "input_lock_sha256": lock_hash, "gates": gates})
    if gates["P1_rows"]["status"] != "pass" or gates["P3_finite_metrics"]["status"] != "pass":
        raise RuntimeError("full_gate_failed")
    hashes_path = OUT / "file_sha256.json"
    hashes = output_hashes()
    if hashes_path.exists():
        if read_json(hashes_path) != hashes:
            raise RuntimeError("output_hash_mismatch")
    else:
        write_json_new(hashes_path, hashes)
    manifest_path = OUT / "manifest.json"
    manifest = {"stage": "unified_evidence_two_partition_p_hat_calibration_v1", "status": "passed", "created_at_utc": now(), "classification": "user_authorized_result_informed_same_sample_fixed_p_hat_diagnostic", "input_lock_sha256": lock_hash, "smoke_checkpoint_sha256": smoke_hash, "gate_statuses": {name: value["status"] for name, value in gates.items()}, "file_sha256_sha256": digest(hashes_path), "interpretation": "Same-sample fixed-p_hat loss and calibration diagnostics on E-A/E-B physical engagements only; not independent validation, a main-test gate, causal evidence, latent-state evidence, opportunity-process evidence, or p_hat improvement evidence."}
    if manifest_path.exists():
        saved = read_json(manifest_path)
        if saved.get("input_lock_sha256") != lock_hash or saved.get("file_sha256_sha256") != manifest["file_sha256_sha256"]:
            raise RuntimeError("resume_manifest_mismatch")
    else:
        write_json_new(manifest_path, manifest)


def run(phase: str) -> None:
    _, lock_hash = establish_lock()
    assignments, asset = load_assignments(), load_asset()
    if phase == "smoke":
        smoke(asset, assignments, lock_hash)
        print(json.dumps({"status": "smoke_passed", "output_root": str(OUT)}, ensure_ascii=True))
        return
    smoke_hash = require_smoke(lock_hash)
    metrics = full(asset, assignments, lock_hash, smoke_hash)
    finish(metrics, lock_hash, smoke_hash)
    print(json.dumps({"status": "full_passed", "output_root": str(OUT)}, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    phase = parser.parse_args().phase
    try:
        run(phase)
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True)
        failed = OUT / f"failed_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        failed.mkdir(exist_ok=False)
        write_json_new(failed / "failure_manifest.json", {"stage": "unified_evidence_two_partition_p_hat_calibration_v1", "status": "failed", "created_at_utc": now(), "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
