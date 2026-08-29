"""Repair the v2 match allocation under the frozen protocol-55 tie rule."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
E1_ASSET = ROOT / "results" / "conditional_residual_asset_e1_v1" / "E1-v2" / "residual_asset.parquet"
E1_LOCK = ROOT / "results" / "conditional_residual_asset_e1_v1" / "input_lock.json"
R1_ASSET = Path(r"D:\CS2_60s_confirmation_r1_70_preoutcome_v1\analysis_60s_v1\residual_asset.parquet")
OUT = ROOT / "results" / "unified_evidence_two_partition_audit_v3"
DATA_OUT = ROOT / "data" / "unified_evidence_two_partitions_v3"
STAGE = "unified_evidence_two_partition_audit_v3"
MODEL_SHA256 = "64f19f6aa3158d1fa3e795080e6eb8cde2f7db9d35a1f506948eda3394ac2b36"
QUOTAS = {
    "E-A": {"R1-covered": 30, "E1-only": 70},
    "E-B": {"R1-covered": 30, "E1-only": 70},
    "E-RESERVE": {"R1-covered": 10, "E1-only": 27},
}
SMOKE_PER_STRATUM = 15


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required_input_missing:{path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"immutable_output_exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def source_model_hash(lock: dict[str, Any]) -> str:
    matches = [
        item["sha256"] for path, item in lock["inputs"].items()
        if path.replace("\\", "/").endswith("/full_model.joblib")
    ]
    if len(matches) != 1:
        raise RuntimeError("e1_model_hash_not_unique")
    return matches[0]


def match_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["match_id", "map_id"])
    if frame["match_id"].isna().any() or frame["map_id"].isna().any():
        raise RuntimeError(f"missing_match_or_map_id:{path}")
    frame["match_id"] = frame["match_id"].astype(str)
    return frame.groupby("match_id", sort=True).agg(map_count=("map_id", "nunique")).reset_index()


def rank_key(stratum: str, match_id: str) -> str:
    text = f"unified-two-partitions-v2|{stratum}|{match_id}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ranked(frame: pd.DataFrame, stratum: str) -> pd.DataFrame:
    work = frame.copy()
    work["stable_rank"] = [rank_key(stratum, match_id) for match_id in work["match_id"]]
    return work.sort_values(["map_count", "stable_rank"], ascending=[False, True], kind="mergesort")


def allocate_ranked(work: pd.DataFrame, stratum: str, require_complete: bool) -> pd.DataFrame:
    quotas = {partition: values[stratum] for partition, values in QUOTAS.items()}
    if require_complete and len(work) != sum(quotas.values()):
        raise RuntimeError(f"stratum_quota_count_mismatch:{stratum}")
    remaining, map_totals, rows = quotas.copy(), {key: 0 for key in quotas}, []
    for row in work.itertuples(index=False):
        eligible = [key for key, value in remaining.items() if value]
        if not eligible:
            raise RuntimeError(f"stratum_capacity_exhausted:{stratum}")
        # Protocol 55 fixes the partition-name lexicographic tie break.
        partition = min(eligible, key=lambda key: (Fraction(map_totals[key], quotas[key]), key))
        remaining[partition] -= 1
        map_totals[partition] += int(row.map_count)
        rows.append({
            "partition": partition,
            "canonical_source": "E1-v2",
            "coverage_stratum": stratum,
            "match_id": row.match_id,
            "map_count": int(row.map_count),
            "stable_rank": row.stable_rank,
        })
    if require_complete and any(remaining.values()):
        raise RuntimeError(f"stratum_quota_unfilled:{stratum}")
    return pd.DataFrame(rows)


def establish_lock() -> tuple[dict[str, Any], str]:
    inputs = {
        "e1_residual_asset": record(E1_ASSET),
        "e1_input_lock": record(E1_LOCK),
        "r1_residual_asset": record(R1_ASSET),
        "script": record(Path(__file__).resolve()),
    }
    path = OUT / "checkpoint_01_input_lock.json"
    if path.exists():
        saved = read_json(path)
        if saved.get("stage") != f"{STAGE}_input_lock" or saved.get("inputs") != inputs:
            raise RuntimeError("resume_input_lock_hash_mismatch")
        return saved, digest(path)
    lock = {
        "stage": f"{STAGE}_input_lock",
        "created_at_utc": now(),
        "outcome_access": "previously_materialized_assets_only; match_metadata_used_for_allocation; no_new_models_or_statistics",
        "inputs": inputs,
        "canonical_analytical_representation": "E1-v2",
        "r1_role": "nested_match_coverage_stratum_only; not_additional_matches",
        "fixed_model_sha256": MODEL_SHA256,
        "allocation_rule": "within coverage_stratum: descending map_count then SHA256(unified-two-partitions-v2|stratum|match_id); assign to eligible quota with lowest map_count/quota ratio; ties by partition name lexicographically",
        "quotas": QUOTAS,
        "supersedes": "unified_evidence_two_partition_audit_v2 allocation only; v2 evidence remains immutable",
    }
    write_json_new(path, lock)
    return lock, digest(path)


def source_work() -> pd.DataFrame:
    if source_model_hash(read_json(E1_LOCK)) != MODEL_SHA256:
        raise RuntimeError("e1_fixed_model_hash_mismatch")
    e1, r1 = match_summary(E1_ASSET), match_summary(R1_ASSET)
    e1_ids, r1_ids = set(e1["match_id"]), set(r1["match_id"])
    if not r1_ids <= e1_ids:
        raise RuntimeError("r1_not_nested_in_e1")
    if len(e1_ids) != 237 or len(r1_ids) != 70:
        raise RuntimeError("unexpected_match_counts")
    work = e1.copy()
    work["coverage_stratum"] = work["match_id"].isin(r1_ids).map({True: "R1-covered", False: "E1-only"})
    return work


def smoke_payload(work: pd.DataFrame, lock_hash: str) -> dict[str, Any]:
    assignments = []
    for stratum in ("R1-covered", "E1-only"):
        subset = ranked(work.loc[work["coverage_stratum"].eq(stratum), ["match_id", "map_count"]], stratum).head(SMOKE_PER_STRATUM)
        assignments.extend(allocate_ranked(subset, stratum, require_complete=False).to_dict(orient="records"))
    assignments = sorted(assignments, key=lambda row: (row["coverage_stratum"], row["match_id"]))
    encoded = json.dumps(assignments, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return {
        "stage": f"{STAGE}_smoke",
        "executed_phase": "smoke",
        "input_lock_sha256": lock_hash,
        "subset_rule": "first 15 ranked matches in each coverage stratum, allocated with the production rule",
        "subset_matches": len(assignments),
        "assignment_sha256": hashlib.sha256(encoded).hexdigest(),
        "assignments": assignments,
        "no_analysis_outputs": True,
    }


def validate_or_write_smoke(work: pd.DataFrame, lock_hash: str, phase: str) -> tuple[dict[str, Any], str]:
    expected = smoke_payload(work, lock_hash)
    path = OUT / "checkpoint_02_smoke.json"
    if phase == "smoke":
        if path.exists():
            saved = read_json(path)
            if any(saved.get(key) != value for key, value in expected.items()):
                raise RuntimeError("resume_smoke_checkpoint_mismatch")
            return saved, digest(path)
        write_json_new(path, {**expected, "created_at_utc": now()})
        return expected, digest(path)
    if not path.exists():
        raise RuntimeError("full_requires_prior_smoke")
    saved = read_json(path)
    if any(saved.get(key) != value for key, value in expected.items()):
        raise RuntimeError("full_smoke_checkpoint_mismatch")
    return saved, digest(path)


def full_audit(work: pd.DataFrame, lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    allocations = pd.concat([
        allocate_ranked(ranked(work.loc[work["coverage_stratum"].eq(stratum), ["match_id", "map_count"]], stratum), stratum, require_complete=True)
        for stratum in ("R1-covered", "E1-only")
    ], ignore_index=True).sort_values(["partition", "coverage_stratum", "match_id"], kind="mergesort")
    summary = allocations.groupby(["partition", "coverage_stratum"], sort=True).size().unstack(fill_value=0)
    for partition, values in QUOTAS.items():
        for stratum, count in values.items():
            if int(summary.loc[partition, stratum]) != count:
                raise RuntimeError("allocation_quota_verification_failed")
    if allocations["match_id"].nunique() != 237 or allocations["match_id"].duplicated().any():
        raise RuntimeError("assignment_match_uniqueness_failed")
    partition_sizes = {key: int(value) for key, value in allocations.groupby("partition", sort=True).size().items()}
    if partition_sizes != {"E-A": 100, "E-B": 100, "E-RESERVE": 37}:
        raise RuntimeError("partition_size_verification_failed")
    audit = {
        "stage": f"{STAGE}_audit",
        "created_at_utc": now(),
        "input_lock_sha256": lock_hash,
        "gates": {
            "G0_input_hashes": {"status": "pass"},
            "G1_canonical_fixed_model": {"status": "pass", "model_sha256": MODEL_SHA256},
            "G2_nested_r1_coverage": {"status": "pass", "e1_unique_matches": 237, "r1_matches": 70, "r1_matches_nested_in_e1": 70},
            "G3_unique_pool_size": {"status": "pass", "unique_match_count": 237},
            "G4_exact_partition_quotas": {"status": "pass", "partition_sizes": partition_sizes},
            "G5_frozen_allocation_rule": {"status": "pass", "tie_break": "partition_name_lexicographic"},
        },
        "outcome_boundary": "No Y_t, p_hat, e_t, state, coefficient, confidence interval, p value, or loss is written to this package.",
        "reserve_status": "37-match history-exposed reserve; not label-unseen or independently confirmatory by allocation alone",
    }
    return allocations, audit


def write_assignments(allocations: pd.DataFrame, audit: dict[str, Any], lock_hash: str, smoke_hash: str) -> dict[str, Any]:
    parquet_path = DATA_OUT / "match_assignments.parquet"
    csv_path = DATA_OUT / "match_assignments.csv"
    manifest_path = DATA_OUT / "manifest.json"
    if manifest_path.exists():
        saved = read_json(manifest_path)
        if saved.get("input_lock_sha256") != lock_hash or saved.get("smoke_checkpoint_sha256") != smoke_hash:
            raise RuntimeError("data_resume_lock_mismatch")
        if any(digest(DATA_OUT / name) != value for name, value in saved["files"].items()):
            raise RuntimeError("data_resume_file_hash_mismatch")
        return saved
    if parquet_path.exists() or csv_path.exists():
        raise RuntimeError("incomplete_data_output")
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    allocations.to_parquet(parquet_path, index=False, compression="zstd")
    allocations.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")
    manifest = {
        "stage": "unified_evidence_two_partitions_v3",
        "created_at_utc": now(),
        "input_lock_sha256": lock_hash,
        "smoke_checkpoint_sha256": smoke_hash,
        "assignment_unit": "match_id",
        "columns": list(allocations.columns),
        "outcome_boundary": audit["outcome_boundary"],
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
    _, lock_hash = establish_lock()
    work = source_work()
    _, smoke_hash = validate_or_write_smoke(work, lock_hash, args.phase)
    if args.phase == "smoke":
        print(json.dumps({"status": "smoke_passed", "output_root": str(OUT)}, ensure_ascii=True))
        return
    allocations, audit = full_audit(work, lock_hash)
    audit_path = OUT / "checkpoint_03_audit.json"
    if audit_path.exists():
        saved = read_json(audit_path)
        if saved.get("input_lock_sha256") != lock_hash or saved.get("gates") != audit["gates"]:
            raise RuntimeError("resume_audit_checkpoint_mismatch")
    else:
        write_json_new(audit_path, audit)
    data_manifest = write_assignments(allocations, audit, lock_hash, smoke_hash)
    hashes = {name: digest(path) for name, path in {
        "checkpoint_01_input_lock.json": OUT / "checkpoint_01_input_lock.json",
        "checkpoint_02_smoke.json": OUT / "checkpoint_02_smoke.json",
        "checkpoint_03_audit.json": audit_path,
        "data_manifest.json": DATA_OUT / "manifest.json",
    }.items()}
    hashes_path = OUT / "file_sha256.json"
    if hashes_path.exists():
        if read_json(hashes_path) != hashes:
            raise RuntimeError("resume_file_hash_manifest_mismatch")
    else:
        write_json_new(hashes_path, hashes)
    manifest_path = OUT / "manifest.json"
    manifest = {
        "stage": STAGE,
        "status": "passed_metadata_only",
        "created_at_utc": now(),
        "input_lock_sha256": lock_hash,
        "smoke_checkpoint_sha256": smoke_hash,
        "gate_statuses": {key: value["status"] for key, value in audit["gates"].items()},
        "data_manifest_sha256": digest(DATA_OUT / "manifest.json"),
        "file_sha256_sha256": digest(hashes_path),
        "interpretation": "Protocol-55 match-level allocation repair and provenance audit only. It authorizes no association, residual-loss, or confirmation claim.",
    }
    if manifest_path.exists():
        saved = read_json(manifest_path)
        if saved.get("input_lock_sha256") != lock_hash or saved.get("data_manifest_sha256") != manifest["data_manifest_sha256"]:
            raise RuntimeError("resume_manifest_mismatch")
    else:
        write_json_new(manifest_path, manifest)
    print(json.dumps({"status": "full_passed", "output_root": str(OUT), "data_root": str(DATA_OUT), "assignment_files": data_manifest["files"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
