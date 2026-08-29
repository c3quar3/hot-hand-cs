"""Run the user-authorized E-A/E-B final confirmation analysis."""
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

import pandas as pd

import analyze_exponential_history_state_sequence_v1 as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "unified_evidence_two_partition_confirmation_v1"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "58_统一证据池两分区最终确认关联执行锁_v1_2026-08-27.md"
REPAIR_PROTOCOL = ROOT / "docs" / "修订与补丁" / "57_统一证据池两分区分配修复执行锁_v1_2026-08-27.md"
ASSIGNMENT_ROOT = ROOT / "data" / "unified_evidence_two_partitions_v3"
ASSIGNMENTS = ASSIGNMENT_ROOT / "match_assignments.parquet"
ASSIGNMENT_MANIFEST = ASSIGNMENT_ROOT / "manifest.json"
E1_ASSET = ROOT / "results" / "conditional_residual_asset_e1_v1" / "E1-v2" / "residual_asset.parquet"
E1_LOCK = ROOT / "results" / "conditional_residual_asset_e1_v1" / "input_lock.json"
BASE_SCRIPT = ROOT / "scripts" / "analyze_exponential_history_state_sequence_v1.py"
PARTITIONS = ("E-A", "E-B")
HALF_LIFE_SECONDS = 60.0
TICK_RATE = 64.0
REFERENCE = {
    "mu_z": 11.538069409706317,
    "mu_S": 0.34010149087253094,
    "sigma_S_ddof1": 0.6490311893067302,
}
SMOKE_MATCHES = 30
CI_HALF_WIDTH_MAX = 0.020
MDE_80_MAX = 0.030


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
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


def establish_input_lock() -> tuple[dict[str, Any], str]:
    inputs = {
        "execution_protocol": file_record(PROTOCOL),
        "repair_protocol": file_record(REPAIR_PROTOCOL),
        "script": file_record(Path(__file__).resolve()),
        "state_builder_and_model": file_record(BASE_SCRIPT),
        "assignment_manifest": file_record(ASSIGNMENT_MANIFEST),
        "assignments": file_record(ASSIGNMENTS),
        "e1_residual_asset": file_record(E1_ASSET),
        "e1_input_lock": file_record(E1_LOCK),
    }
    expected = {
        "stage": "unified_evidence_two_partition_confirmation_v1_input_lock",
        "classification": "user_authorized_final_confirmatory_data_source_with_prior_outcome_access_disclosed",
        "partitions": list(PARTITIONS),
        "inputs": inputs,
        "state": {
            "half_life_seconds": HALF_LIFE_SECONDS,
            "tick_rate": TICK_RATE,
            "same_player_same_map_all_strict_history": True,
            "same_tick_shared_pre_state": True,
            "across_round_decay": True,
            "across_map_reset": True,
        },
        "reference": REFERENCE,
        "model": "e_t ~ 1 + z + S_star + S_star:z; OLS two_way_CRV1(match_id, player_id)",
        "inference": "two_sided_alpha_0.05; 95_percent_CI; Holm_across_E_A_and_E_B",
        "precision": {"ci_half_width_max": CI_HALF_WIDTH_MAX, "mde_80_max": MDE_80_MAX},
        "outcome_boundary": "all residual assets were previously materialized; no raw input, rescoring, label materialization, state selection, or half-life selection is authorized",
    }
    path = OUT / "checkpoint_01_input_lock.json"
    if path.exists():
        saved = read_json(path)
        if {key: saved.get(key) for key in expected} != expected:
            raise RuntimeError("resume_input_lock_mismatch")
        return saved, digest(path)
    write_json_new(path, {**expected, "created_at_utc": now()})
    return expected, digest(path)


def load_assignments() -> dict[str, pd.DataFrame]:
    manifest = read_json(ASSIGNMENT_MANIFEST)
    if digest(ASSIGNMENTS) != manifest["files"]["match_assignments.parquet"]:
        raise RuntimeError("assignment_parquet_hash_mismatch")
    frame = pd.read_parquet(ASSIGNMENTS)
    expected_columns = {"partition", "canonical_source", "coverage_stratum", "match_id", "map_count", "stable_rank"}
    if set(frame.columns) != expected_columns:
        raise RuntimeError("assignment_columns_mismatch")
    frame["match_id"] = frame["match_id"].astype(str)
    if frame["match_id"].duplicated().any() or frame["match_id"].nunique() != 237:
        raise RuntimeError("assignment_match_uniqueness_failed")
    if set(frame["partition"]) != {"E-A", "E-B", "E-RESERVE"}:
        raise RuntimeError("assignment_partition_set_mismatch")
    output = {partition: frame.loc[frame["partition"].eq(partition)].copy() for partition in PARTITIONS}
    for partition, subset in output.items():
        strata = subset["coverage_stratum"].value_counts().to_dict()
        if len(subset) != 100 or strata != {"E1-only": 70, "R1-covered": 30}:
            raise RuntimeError(f"assignment_quota_mismatch:{partition}")
    if set(output["E-A"]["match_id"]) & set(output["E-B"]["match_id"]):
        raise RuntimeError("partition_overlap")
    return output


def load_e1_asset() -> pd.DataFrame:
    columns = sorted(base.REQUIRED)
    frame = pd.read_parquet(E1_ASSET, columns=columns)
    frame["match_id"] = frame["match_id"].astype(str)
    base.validate_asset(frame, "E1-v2")
    return frame


def build_state(asset: pd.DataFrame, partition: str, tier: str, lock_hash: str, match_ids: list[str]) -> tuple[pd.DataFrame, dict[str, Any], str]:
    directory = OUT / tier / partition
    matrix_path = directory / "state_matrix.parquet"
    audit_path = directory / "state_audit.json"
    match_hash = hashlib.sha256("\n".join(match_ids).encode("utf-8")).hexdigest()
    if matrix_path.exists() or audit_path.exists():
        if not matrix_path.exists() or not audit_path.exists():
            raise RuntimeError(f"incomplete_state_checkpoint:{tier}:{partition}")
        audit = read_json(audit_path)
        if audit.get("input_lock_sha256") != lock_hash or audit.get("match_id_sha256") != match_hash:
            raise RuntimeError(f"state_checkpoint_mismatch:{tier}:{partition}")
        state = pd.read_parquet(matrix_path)
        if digest(matrix_path) != audit.get("state_matrix_sha256"):
            raise RuntimeError(f"state_matrix_hash_mismatch:{tier}:{partition}")
        return state, audit, digest(matrix_path)
    subset = asset.loc[asset["match_id"].isin(match_ids)].copy()
    if subset["match_id"].nunique() != len(match_ids):
        raise RuntimeError(f"asset_match_coverage_mismatch:{tier}:{partition}")
    state, metrics = base.build_state(subset, f"{partition}:{tier}")
    write_parquet_new(matrix_path, state)
    matrix_hash = digest(matrix_path)
    audit = {
        "stage": "unified_evidence_two_partition_confirmation_v1_state",
        "created_at_utc": now(),
        "input_lock_sha256": lock_hash,
        "partition": partition,
        "tier": tier,
        "matches": len(match_ids),
        "match_id_sha256": match_hash,
        "state_matrix_sha256": matrix_hash,
        "metrics": metrics,
        "G0_event_asset": {"status": "pass", "physical_event_rows": int(len(subset))},
        "G1_state": {"status": "pass" if metrics["maximum_direct_sum_error"] <= 1e-10 else "fail"},
        "G2_cluster_support": {"status": "pass" if metrics["match_clusters"] >= 30 and metrics["player_clusters"] >= 30 else "fail"},
    }
    write_json_new(audit_path, audit)
    if audit["G1_state"]["status"] != "pass" or audit["G2_cluster_support"]["status"] != "pass":
        raise RuntimeError(f"state_preflight_failed:{tier}:{partition}")
    return state, audit, matrix_hash


def smoke(assets: pd.DataFrame, assignments: dict[str, pd.DataFrame], lock_hash: str) -> tuple[dict[str, Any], str]:
    path = OUT / "checkpoint_02_smoke.json"
    metrics: dict[str, Any] = {}
    state_hashes: dict[str, str] = {}
    state_audits: dict[str, Any] = {}
    for partition in PARTITIONS:
        matches = sorted(assignments[partition]["match_id"].tolist())[:SMOKE_MATCHES]
        state, audit, state_hash = build_state(assets, partition, "smoke", lock_hash, matches)
        model = base.fit(state, REFERENCE, expose=False)
        metrics[partition] = model
        state_hashes[partition] = state_hash
        state_audits[partition] = {key: audit[key] for key in ("G0_event_asset", "G1_state", "G2_cluster_support")}
    expected = {
        "stage": "unified_evidence_two_partition_confirmation_v1_smoke",
        "executed_phase": "smoke",
        "input_lock_sha256": lock_hash,
        "subset_rule": "30_lexicographically_smallest_match_id_clusters_within_each_partition",
        "effect_output_withheld": True,
        "state_hashes": state_hashes,
        "state_preflight": state_audits,
        "G3": {"status": "pass" if all(item["g3_pass"] for item in metrics.values()) else "fail", "metrics": metrics},
    }
    if path.exists():
        saved = read_json(path)
        if all(saved.get(key) == value for key, value in expected.items()):
            return saved, digest(path)
        raise RuntimeError("resume_smoke_checkpoint_mismatch")
    write_json_new(path, {**expected, "created_at_utc": now()})
    if expected["G3"]["status"] != "pass":
        raise RuntimeError("smoke_g3_failed")
    return expected, digest(path)


def require_prior_smoke(lock_hash: str) -> str:
    path = OUT / "checkpoint_02_smoke.json"
    if not path.exists():
        raise RuntimeError("full_requires_prior_smoke")
    smoke_checkpoint = read_json(path)
    if smoke_checkpoint.get("executed_phase") != "smoke" or smoke_checkpoint.get("input_lock_sha256") != lock_hash:
        raise RuntimeError("full_smoke_checkpoint_mismatch")
    if smoke_checkpoint.get("G3", {}).get("status") != "pass":
        raise RuntimeError("full_requires_smoke_g3_pass")
    return digest(path)


def holm(models: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(PARTITIONS, key=lambda name: models[name]["beta"]["p_value"])
    running = 0.0
    for index, name in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - index) * models[name]["beta"]["p_value"])
        running = max(running, adjusted)
        models[name]["beta"]["holm_p_value_across_partitions"] = running


def full(assets: pd.DataFrame, assignments: dict[str, pd.DataFrame], lock_hash: str, smoke_hash: str) -> dict[str, Any]:
    path = OUT / "checkpoint_03_models.json"
    state_hashes: dict[str, str] = {}
    states: dict[str, pd.DataFrame] = {}
    audits: dict[str, Any] = {}
    for partition in PARTITIONS:
        matches = sorted(assignments[partition]["match_id"].tolist())
        states[partition], audits[partition], state_hashes[partition] = build_state(assets, partition, "full", lock_hash, matches)
    expected_prefix = {
        "stage": "unified_evidence_two_partition_confirmation_v1_models",
        "input_lock_sha256": lock_hash,
        "smoke_checkpoint_sha256": smoke_hash,
        "state_hashes": state_hashes,
    }
    if path.exists():
        saved = read_json(path)
        if all(saved.get(key) == value for key, value in expected_prefix.items()):
            return saved
        raise RuntimeError("resume_model_checkpoint_mismatch")
    models = {partition: base.fit(states[partition], REFERENCE, expose=True) for partition in PARTITIONS}
    if not all(model["g3_pass"] for model in models.values()):
        raise RuntimeError("full_g3_failed")
    holm(models)
    value = {
        **expected_prefix,
        "created_at_utc": now(),
        "models": models,
        "state_preflight": {partition: {key: audits[partition][key] for key in ("G0_event_asset", "G1_state", "G2_cluster_support")} for partition in PARTITIONS},
    }
    write_json_new(path, value)
    return value


def finish(models_checkpoint: dict[str, Any], lock_hash: str, smoke_hash: str) -> None:
    models = models_checkpoint["models"]
    precision = {
        partition: {
            "status": "pass" if model["ci_half_width_95"] <= CI_HALF_WIDTH_MAX and model["mde_80_single"] <= MDE_80_MAX else "fail",
            "ci_half_width_95": model["ci_half_width_95"],
            "mde_80_single": model["mde_80_single"],
        }
        for partition, model in models.items()
    }
    positive = {
        partition: bool(
            models[partition]["beta"]["estimate"] > 0
            and models[partition]["beta"]["ci95_low"] > 0
            and models[partition]["beta"]["holm_p_value_across_partitions"] < 0.05
        )
        for partition in PARTITIONS
    }
    gates = {
        "C3": {"status": "pass", "metrics": {partition: {key: models[partition][key] for key in ("n", "match_clusters", "player_clusters", "intersection_clusters", "max_leverage")} for partition in PARTITIONS}},
        "C4_precision": {"status": "pass" if all(value["status"] == "pass" for value in precision.values()) else "fail", "items": precision},
        "C5_positive_confirmatory_association": {"status": "pass" if all(positive.values()) and all(value["status"] == "pass" for value in precision.values()) else "fail", "per_partition": positive},
        "C6_confirmation_record": {
            "status": "user_adjudicated_final_data_source",
            "reason": "user explicitly designated E-A/E-B as the final confirmatory data source; both partitions derive from previously materialized E1-v2 residual assets and this historical access remains disclosed",
        },
    }
    gate_path = OUT / "checkpoint_04_gate_audit.json"
    if not gate_path.exists():
        write_json_new(gate_path, {"stage": "unified_evidence_two_partition_confirmation_v1_gate_audit", "created_at_utc": now(), "input_lock_sha256": lock_hash, "smoke_checkpoint_sha256": smoke_hash, "gates": gates})
    summary_path = OUT / "checkpoint_05_summary.json"
    if not summary_path.exists():
        write_json_new(summary_path, {"stage": "unified_evidence_two_partition_confirmation_v1_summary", "created_at_utc": now(), "input_lock_sha256": lock_hash, "reference": REFERENCE, "models": models, "inference": "unweighted_ols_two_way_CRV1_match_and_player_normal_approximation; Holm_across_E_A_and_E_B"})
    hashes_path = OUT / "file_sha256.json"
    hashes = {str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"}}
    if hashes_path.exists():
        if read_json(hashes_path) != hashes:
            raise RuntimeError("output_hash_mismatch")
    else:
        write_json_new(hashes_path, hashes)
    manifest_path = OUT / "manifest.json"
    manifest = {
        "stage": "unified_evidence_two_partition_confirmation_v1",
        "status": "passed" if gates["C5_positive_confirmatory_association"]["status"] == "pass" else "completed_without_positive_confirmation_pass",
        "created_at_utc": now(),
        "classification": "user_authorized_final_confirmatory_data_source_with_prior_outcome_access_disclosed",
        "input_lock_sha256": lock_hash,
        "smoke_checkpoint_sha256": smoke_hash,
        "gate_statuses": {name: value["status"] for name, value in gates.items()},
        "file_sha256_sha256": digest(hashes_path),
        "interpretation": "Observed engagement-sequence association only. User adjudicated E-A/E-B as final confirmatory data source. The historical E1-v2 outcome/residual exposure remains part of the research record; no causal, latent-state, opportunity-process, or p_hat predictive-gain claim is authorized.",
    }
    if manifest_path.exists():
        saved = read_json(manifest_path)
        if saved.get("input_lock_sha256") != lock_hash or saved.get("file_sha256_sha256") != manifest["file_sha256_sha256"]:
            raise RuntimeError("resume_manifest_mismatch")
    else:
        write_json_new(manifest_path, manifest)


def run(phase: str) -> None:
    _, lock_hash = establish_input_lock()
    assignments = load_assignments()
    asset = load_e1_asset()
    if phase == "smoke":
        smoke(asset, assignments, lock_hash)
        print(json.dumps({"status": "smoke_passed", "output_root": str(OUT)}, ensure_ascii=True))
        return
    smoke_hash = require_prior_smoke(lock_hash)
    models = full(asset, assignments, lock_hash, smoke_hash)
    finish(models, lock_hash, smoke_hash)
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
        write_json_new(failed / "failure_manifest.json", {"stage": "unified_evidence_two_partition_confirmation_v1", "status": "failed", "created_at_utc": now(), "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
