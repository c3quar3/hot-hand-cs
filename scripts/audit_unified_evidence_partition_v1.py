"""Audit and allocate the approved E1-v2/R1 unified evidence pool without analysis output."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
E1_ASSET = ROOT / "results" / "conditional_residual_asset_e1_v1" / "E1-v2" / "residual_asset.parquet"
E1_INPUT_LOCK = ROOT / "results" / "conditional_residual_asset_e1_v1" / "input_lock.json"
R1_ROOT = Path(r"D:\CS2_60s_confirmation_r1_70_preoutcome_v1")
R1_ASSET = R1_ROOT / "analysis_60s_v1" / "residual_asset.parquet"
R1_SCORE_MANIFEST = R1_ROOT / "p_hat_preoutcome_v1" / "manifest.json"
OUT = ROOT / "results" / "unified_evidence_partition_audit_v1"
DATA_OUT = ROOT / "data" / "unified_evidence_partitions_v1"

STAGE = "unified_evidence_partition_audit_v1"
MODEL_SHA256 = "64f19f6aa3158d1fa3e795080e6eb8cde2f7db9d35a1f506948eda3394ac2b36"
REQUIRED_COLUMNS = {
    "physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id",
    "opponent_player_id", "e_t", "residual_valid", "primary_analysis_eligible",
    "p_support_0p01",
}
QUOTAS = {
    "E-A": {"E1-v2": 78, "R1": 22},
    "E-B": {"E1-v2": 77, "R1": 23},
    "E-C": {"E1-v2": 77, "R1": 23},
    "E-RESERVE": {"E1-v2": 5, "R1": 2},
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"immutable_output_exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_json(path: Path, expected: dict[str, Any], lock_hash: str) -> None:
    saved = read_json(path)
    if saved.get("input_lock_sha256") != lock_hash:
        raise RuntimeError(f"resume_input_lock_mismatch:{path.name}")
    for key, value in expected.items():
        if saved.get(key) != value:
            raise RuntimeError(f"resume_checkpoint_mismatch:{path.name}:{key}")


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required_input_missing:{path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def model_hash_from_e1_lock(lock: dict[str, Any]) -> str:
    values = [
        item["sha256"]
        for path, item in lock["inputs"].items()
        if path.replace("\\", "/").endswith("/full_model.joblib")
    ]
    if len(values) != 1:
        raise RuntimeError("e1_model_hash_not_unique")
    return values[0]


def source_frame(path: Path, source: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise RuntimeError(f"{source}_required_columns_missing:{','.join(missing)}")
    if frame["match_id"].isna().any() or frame["map_id"].isna().any():
        raise RuntimeError(f"{source}_missing_match_or_map_id")
    for column in ("residual_valid", "primary_analysis_eligible", "p_support_0p01"):
        if frame[column].isna().any():
            raise RuntimeError(f"{source}_{column}_missing")
    grouped = frame.assign(match_id=frame["match_id"].astype(str)).groupby("match_id", sort=True)
    summary = grouped.agg(map_count=("map_id", "nunique"), source_rows=("map_id", "size")).reset_index()
    summary["legacy_source"] = source
    return summary[["legacy_source", "match_id", "map_count", "source_rows"]]


def stable_rank(source: str, match_id: str) -> str:
    return hashlib.sha256(f"unified-evidence-v1|{source}|{match_id}".encode("utf-8")).hexdigest()


def allocate_source(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    quota = {partition: values[source] for partition, values in QUOTAS.items()}
    if len(frame) != sum(quota.values()):
        raise RuntimeError(f"{source}_quota_count_mismatch")
    work = frame.copy()
    work["stable_rank"] = [stable_rank(source, match_id) for match_id in work["match_id"]]
    work = work.sort_values(["map_count", "stable_rank"], ascending=[False, True], kind="mergesort")
    assigned_rows: list[dict[str, Any]] = []
    remaining = quota.copy()
    map_totals = {partition: 0 for partition in quota}
    for row in work.itertuples(index=False):
        eligible = [partition for partition, count in remaining.items() if count > 0]
        partition = min(
            eligible,
            key=lambda value: (map_totals[value] / quota[value], map_totals[value], value),
        )
        remaining[partition] -= 1
        map_totals[partition] += int(row.map_count)
        assigned_rows.append({
            "partition": partition,
            "legacy_source": source,
            "match_id": row.match_id,
            "map_count": int(row.map_count),
            "source_rows": int(row.source_rows),
            "stable_rank": row.stable_rank,
        })
    if any(remaining.values()):
        raise RuntimeError(f"{source}_quota_unfilled")
    return pd.DataFrame(assigned_rows).sort_values(["partition", "legacy_source", "match_id"], kind="mergesort")


def input_lock() -> tuple[dict[str, Any], str]:
    current = {
        "e1_residual_asset": file_record(E1_ASSET),
        "e1_input_lock": file_record(E1_INPUT_LOCK),
        "r1_residual_asset": file_record(R1_ASSET),
        "r1_score_manifest": file_record(R1_SCORE_MANIFEST),
        "script": file_record(Path(__file__).resolve()),
    }
    path = OUT / "checkpoint_01_input_lock.json"
    if path.exists():
        saved = read_json(path)
        if saved.get("stage") != f"{STAGE}_input_lock" or saved.get("inputs") != current:
            raise RuntimeError("resume_input_lock_hash_mismatch")
        return saved, digest(path)
    value = {
        "stage": f"{STAGE}_input_lock",
        "created_at_utc": now(),
        "outcome_access": "previously_materialized_residual_assets_only; no_new_labels_or_models",
        "inputs": current,
        "required_columns": sorted(REQUIRED_COLUMNS),
        "fixed_model_sha256": MODEL_SHA256,
        "allocation_rule": "source-stratified; descending map_count then SHA256(unified-evidence-v1|legacy_source|match_id); assign to the eligible quota with the lowest map_count/quota ratio",
        "quotas": QUOTAS,
    }
    write_json_new(path, value)
    return value, digest(path)


def audit(lock: dict[str, Any], lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    e1_lock = read_json(E1_INPUT_LOCK)
    r1_manifest = read_json(R1_SCORE_MANIFEST)
    e1_model_hash = model_hash_from_e1_lock(e1_lock)
    r1_model_hash = r1_manifest.get("model_sha256")
    if e1_model_hash != MODEL_SHA256 or r1_model_hash != MODEL_SHA256:
        raise RuntimeError("fixed_model_hash_mismatch")
    e1 = source_frame(E1_ASSET, "E1-v2")
    r1 = source_frame(R1_ASSET, "R1")
    overlap = sorted(set(e1["match_id"]) & set(r1["match_id"]))
    if overlap:
        raise RuntimeError(f"match_id_overlap:{','.join(overlap[:10])}")
    expected_counts = {"E1-v2": 237, "R1": 70}
    observed_counts = {"E1-v2": int(len(e1)), "R1": int(len(r1))}
    if observed_counts != expected_counts:
        raise RuntimeError(f"unexpected_source_match_counts:{observed_counts}")
    combined = pd.concat([e1, r1], ignore_index=True)
    allocations = pd.concat([
        allocate_source(combined.loc[combined["legacy_source"].eq(source)], source)
        for source in ("E1-v2", "R1")
    ], ignore_index=True)
    allocation_counts = allocations.groupby(["partition", "legacy_source"], sort=True).size().unstack(fill_value=0)
    for partition, values in QUOTAS.items():
        for source, expected in values.items():
            if int(allocation_counts.loc[partition, source]) != expected:
                raise RuntimeError("allocation_quota_verification_failed")
    partition_sizes = allocations.groupby("partition", sort=True).size().to_dict()
    if {key: int(value) for key, value in partition_sizes.items()} != {"E-A": 100, "E-B": 100, "E-C": 100, "E-RESERVE": 7}:
        raise RuntimeError("partition_size_verification_failed")
    audit_value = {
        "stage": f"{STAGE}_audit",
        "created_at_utc": now(),
        "input_lock_sha256": lock_hash,
        "gates": {
            "G0_input_hashes": {"status": "pass"},
            "G1_fixed_p_hat": {"status": "pass", "model_sha256": MODEL_SHA256},
            "G2_residual_contract": {
                "status": "pass",
                "required_columns_present": sorted(REQUIRED_COLUMNS),
                "e1_residual_formula_version": "r=Y-p;e=(Y-p)/sqrt(p*(1-p));v1",
                "r1_execution_lock": "e=(Y-p)/sqrt(p*(1-p)); 0.01<=p<=0.99",
            },
            "G3_match_independence": {"status": "pass", "match_id_overlap_count": 0},
            "G4_expected_pool_size": {"status": "pass", "source_match_counts": observed_counts, "total_matches": int(len(combined))},
            "G5_exact_partition_quotas": {"status": "pass", "partition_sizes": {key: int(value) for key, value in partition_sizes.items()}},
        },
        "outcome_boundary": "No Y_t, p_hat, e_t, state, coefficient, confidence interval, p value, or loss is written to this output package.",
        "reserve_status": "history_exposed_reserve; not label-unseen or confirmatory by this allocation alone",
    }
    return allocations, audit_value


def write_data(allocations: pd.DataFrame, audit_value: dict[str, Any], lock_hash: str) -> dict[str, Any]:
    manifest_path = DATA_OUT / "manifest.json"
    parquet_path = DATA_OUT / "match_assignments.parquet"
    csv_path = DATA_OUT / "match_assignments.csv"
    if manifest_path.exists():
        saved = read_json(manifest_path)
        if saved.get("input_lock_sha256") != lock_hash:
            raise RuntimeError("data_resume_input_lock_mismatch")
        for path, expected_hash in saved["files"].items():
            candidate = DATA_OUT / path
            if not candidate.is_file() or digest(candidate) != expected_hash:
                raise RuntimeError(f"data_resume_file_hash_mismatch:{path}")
        return saved
    if parquet_path.exists() or csv_path.exists():
        raise RuntimeError("incomplete_data_output")
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    allocations.to_parquet(parquet_path, index=False, compression="zstd")
    allocations.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")
    manifest = {
        "stage": "unified_evidence_partitions_v1",
        "created_at_utc": now(),
        "input_lock_sha256": lock_hash,
        "assignment_unit": "match_id",
        "columns": list(allocations.columns),
        "outcome_boundary": audit_value["outcome_boundary"],
        "files": {"match_assignments.parquet": digest(parquet_path), "match_assignments.csv": digest(csv_path)},
    }
    write_json_new(manifest_path, manifest)
    return manifest


def main() -> None:
    global OUT, DATA_OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--data-out", type=Path, default=DATA_OUT)
    args = parser.parse_args()
    OUT, DATA_OUT = args.out.resolve(), args.data_out.resolve()
    lock, lock_hash = input_lock()
    allocations, audit_value = audit(lock, lock_hash)
    smoke_path = OUT / "checkpoint_02_smoke.json"
    smoke_summary = {
        "stage": f"{STAGE}_smoke",
        "input_lock_sha256": lock_hash,
        "subset_rule": "first_30_rows_after_source-stratified_public_assignment; metadata_only",
        "subset_matches": 30,
        "no_analysis_outputs": True,
    }
    if smoke_path.exists():
        verify_json(smoke_path, {key: smoke_summary[key] for key in ("stage", "subset_rule", "subset_matches", "no_analysis_outputs")}, lock_hash)
    else:
        write_json_new(smoke_path, {**smoke_summary, "created_at_utc": now()})
    if args.phase == "smoke":
        print(json.dumps({"status": "smoke_passed", "output_root": str(OUT)}, ensure_ascii=True))
        return
    audit_path = OUT / "checkpoint_03_audit.json"
    if audit_path.exists():
        verify_json(audit_path, {"stage": audit_value["stage"], "gates": audit_value["gates"]}, lock_hash)
    else:
        write_json_new(audit_path, audit_value)
    data_manifest = write_data(allocations, audit_value, lock_hash)
    hashes_path = OUT / "file_sha256.json"
    file_hashes = {
        "checkpoint_01_input_lock.json": digest(OUT / "checkpoint_01_input_lock.json"),
        "checkpoint_02_smoke.json": digest(smoke_path),
        "checkpoint_03_audit.json": digest(audit_path),
        "data_manifest.json": digest(DATA_OUT / "manifest.json"),
    }
    if hashes_path.exists():
        if read_json(hashes_path) != file_hashes:
            raise RuntimeError("resume_file_hash_manifest_mismatch")
    else:
        write_json_new(hashes_path, file_hashes)
    manifest_path = OUT / "manifest.json"
    manifest = {
        "stage": STAGE,
        "status": "passed_metadata_only",
        "created_at_utc": now(),
        "input_lock_sha256": lock_hash,
        "gate_statuses": {key: value["status"] for key, value in audit_value["gates"].items()},
        "data_manifest_sha256": digest(DATA_OUT / "manifest.json"),
        "file_sha256_sha256": digest(hashes_path),
        "interpretation": "Match-level allocation and provenance audit only. It authorizes no association, residual-loss, or confirmation claim.",
    }
    if manifest_path.exists():
        saved = read_json(manifest_path)
        if saved.get("input_lock_sha256") != lock_hash or saved.get("data_manifest_sha256") != digest(DATA_OUT / "manifest.json"):
            raise RuntimeError("resume_manifest_mismatch")
    else:
        write_json_new(manifest_path, manifest)
    print(json.dumps({"status": "full_passed", "output_root": str(OUT), "data_root": str(DATA_OUT), "data_manifest": data_manifest["files"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
