"""Read-only aggregate sample-flow audit for the final E-A/E-B evidence pool.

The audit deliberately stops before a public upstream flow table when any map's
stored candidate input/output difference cannot be reconciled to named reasons.
It never selects outcome, probability, residual, state, or inference columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "unified_evidence_two_partition_aggregate_sample_flow_audit_v1"
INPUTS = {
    "candidate_manifest": ROOT / "results" / "conditional_residual_asset_candidate_cache_v1" / "E1-v2" / "candidate_manifest.json",
    "candidate": ROOT / "results" / "conditional_residual_asset_candidate_cache_v1" / "E1-v2" / "candidate.parquet",
    "residual_asset": ROOT / "results" / "conditional_residual_asset_e1_v1" / "E1-v2" / "residual_asset.parquet",
    "structural_temporal_audit": ROOT / "results" / "conditional_residual_asset_e1_v1" / "E1-v2" / "structural_temporal_audit.json",
    "assignments": ROOT / "data" / "unified_evidence_two_partitions_v3" / "match_assignments.parquet",
    "assignment_manifest": ROOT / "data" / "unified_evidence_two_partitions_v3" / "manifest.json",
    "state_audit_ea": ROOT / "results" / "unified_evidence_two_partition_confirmation_v1" / "full" / "E-A" / "state_audit.json",
    "state_audit_eb": ROOT / "results" / "unified_evidence_two_partition_confirmation_v1" / "full" / "E-B" / "state_audit.json",
}
PARTITIONS = ("E-A", "E-B")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def partition_for_map(map_id: str, match_partition: dict[str, str]) -> str | None:
    return match_partition.get(str(map_id).split("_", 1)[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace this audit's own result directory")
    args = parser.parse_args()

    if OUT.exists() and not args.force:
        manifest_path = OUT / "manifest.json"
        if manifest_path.exists():
            print(f"resuming existing audit: {load_json(manifest_path)['status']}")
            return 0
        raise RuntimeError(f"refusing to overwrite incomplete output: {OUT}")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    missing = [str(path) for path in INPUTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required inputs: " + "; ".join(missing))

    input_lock = {
        "stage": "unified_evidence_two_partition_aggregate_sample_flow_audit_v1",
        "created_at_utc": now(),
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": {name: sha256(path) for name, path in INPUTS.items()},
        "outcome_boundary": (
            "Only identifiers and saved aggregate audit fields are read. The executor does not select "
            "Y_t, p_hat, residuals, states, losses, coefficients, confidence intervals, or p values."
        ),
    }
    write_json(OUT / "input_lock.json", input_lock)

    assignments = pd.read_parquet(INPUTS["assignments"], columns=["partition", "match_id"])
    assignments["match_id"] = assignments["match_id"].astype(str)
    match_partition = dict(zip(assignments["match_id"], assignments["partition"]))
    assignment_counts = assignments.groupby("partition")["match_id"].nunique().to_dict()

    candidate_manifest = load_json(INPUTS["candidate_manifest"])
    candidate = pd.read_parquet(INPUTS["candidate"], columns=["match_id"])
    candidate["partition"] = candidate["match_id"].astype(str).map(match_partition)
    residual = pd.read_parquet(INPUTS["residual_asset"], columns=["match_id"])
    residual["partition"] = residual["match_id"].astype(str).map(match_partition)
    structural_audit = load_json(INPUTS["structural_temporal_audit"])
    state_audits = {"E-A": load_json(INPUTS["state_audit_ea"]), "E-B": load_json(INPUTS["state_audit_eb"])}

    map_rows: list[dict] = []
    for item in candidate_manifest["map_audits"]:
        partition = partition_for_map(item["map_id"], match_partition)
        if partition not in PARTITIONS:
            continue
        named_exclusions = sum(item.get("exclusions", {}).values())
        unresolved = item["input_rows"] - item["output_rows"] - named_exclusions
        map_rows.append(
            {
                "partition": partition,
                "input_rows": int(item["input_rows"]),
                "output_rows": int(item["output_rows"]),
                "named_exclusions": int(named_exclusions),
                "unresolved_rows": int(unresolved),
            }
        )
    maps = pd.DataFrame(map_rows)
    reconciliation: dict[str, dict] = {}
    for partition in PARTITIONS:
        subset = maps.loc[maps["partition"] == partition]
        candidate_rows = int((candidate["partition"] == partition).sum())
        residual_rows = int((residual["partition"] == partition).sum())
        audit = state_audits[partition]
        reconciliation[partition] = {
            "assigned_matches": int(assignment_counts.get(partition, 0)),
            "maps_with_candidate_audit": int(len(subset)),
            "candidate_manifest_input_rows": int(subset["input_rows"].sum()),
            "candidate_manifest_output_rows": int(subset["output_rows"].sum()),
            "candidate_rows_by_assignment": candidate_rows,
            "residual_rows_by_assignment": residual_rows,
            "state_audit_source_rows": int(audit["metrics"]["source_rows"]),
            "named_candidate_exclusions": int(subset["named_exclusions"].sum()),
            "maps_with_unreconciled_difference": int((subset["unresolved_rows"] != 0).sum()),
            "unreconciled_input_output_rows": int(subset["unresolved_rows"].sum()),
            "candidate_minus_residual_rows": candidate_rows - residual_rows,
        }

    g0_pass = (
        all(reconciliation[p]["assigned_matches"] == 100 for p in PARTITIONS)
        and all(reconciliation[p]["candidate_manifest_output_rows"] == reconciliation[p]["candidate_rows_by_assignment"] for p in PARTITIONS)
        and all(reconciliation[p]["residual_rows_by_assignment"] == reconciliation[p]["state_audit_source_rows"] for p in PARTITIONS)
        and sum(reconciliation[p]["candidate_minus_residual_rows"] for p in PARTITIONS)
        == int(structural_audit["duplicate_physical_event_rows_excluded"])
    )
    scope_checkpoint = {
        "stage": "G0_scope_and_downstream_identity",
        "status": "pass" if g0_pass else "fail",
        "partitions": reconciliation,
        "recorded_duplicate_physical_event_rows_excluded": int(structural_audit["duplicate_physical_event_rows_excluded"]),
    }
    write_json(OUT / "checkpoint_01_scope.json", scope_checkpoint)

    g1_pass = g0_pass and all(
        reconciliation[p]["maps_with_unreconciled_difference"] == 0
        and reconciliation[p]["unreconciled_input_output_rows"] == 0
        for p in PARTITIONS
    )
    reconciliation_checkpoint = {
        "stage": "G1_candidate_manifest_reconciliation",
        "status": "pass" if g1_pass else "stop",
        "rule": "For every eligible map, input_rows - output_rows must equal the sum of named exclusions.",
        "partitions": reconciliation,
        "action": (
            "No aggregate upstream sample-flow table was written; unresolved upstream rows are not imputed."
            if not g1_pass
            else "Eligible for a separately generated aggregate upstream sample-flow table."
        ),
    }
    write_json(OUT / "checkpoint_02_reconciliation.json", reconciliation_checkpoint)

    status = "completed" if g1_pass else "stopped_unreconciled_candidate_manifest"
    manifest = {
        "stage": "unified_evidence_two_partition_aggregate_sample_flow_audit_v1",
        "created_at_utc": now(),
        "status": status,
        "protocol": "docs/修订与补丁/67_统一证据池两分区上游聚合样本流审计执行协议_v1_2026-08-28.md",
        "input_lock_sha256": sha256(OUT / "input_lock.json"),
        "scope_checkpoint_sha256": sha256(OUT / "checkpoint_01_scope.json"),
        "reconciliation_checkpoint_sha256": sha256(OUT / "checkpoint_02_reconciliation.json"),
        "output_boundary": "The stopped package contains only input hashes and aggregate reconciliation diagnostics; no new sample-flow table or inferential result is present.",
    }
    write_json(OUT / "manifest.json", manifest)
    print(status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
