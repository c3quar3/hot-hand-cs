"""Audit the map-terminal tail in the D1 cross-round participant-event ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "first_order_residual_persistence_d1_v4"
OUT = ROOT / "results" / "map_terminal_tail_d1_v1"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "26_D1图内跨回合地图终止尾部右删失审计执行协议_v1_D1_only_2026-08-26.md"
DRAFT = ROOT / "docs" / "修订与补丁" / "25_D1图内跨回合地图终止尾部右删失审计协议_草案_v1_D1_only_2026-08-26.md"
DATASETS = ("D1-A", "D1-B", "D1-C")
ALLOWED_COLUMNS = [
    "physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id",
    "opponent_player_id", "primary_eligible",
]
EXPECTED = {
    "D1-A": {"physical_events": 57501, "ledger_rows": 115002, "map_player_runs": 4608, "terminal_rows": 4611, "terminal_fraction": 0.04009495487035008, "multi_terminal_runs": 3},
    "D1-B": {"physical_events": 32786, "ledger_rows": 65572, "map_player_runs": 2260, "terminal_rows": 2261, "terminal_fraction": 0.03448118099188678, "multi_terminal_runs": 1},
    "D1-C": {"physical_events": 25062, "ledger_rows": 50124, "map_player_runs": 1800, "terminal_rows": 1801, "terminal_fraction": 0.0359308913893544, "multi_terminal_runs": 1},
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def source_paths() -> list[Path]:
    paths = [
        PROTOCOL, DRAFT, Path(__file__).resolve(), SOURCE / "manifest.json",
        SOURCE / "file_sha256.json", SOURCE / "checkpoint_03_gate_audit.json",
    ]
    for dataset in DATASETS:
        paths.extend([
            SOURCE / "state" / dataset / "history_state.parquet",
            SOURCE / "state" / dataset / f"checkpoint_02_history_{dataset}.json",
        ])
    return paths


def verify_source_hashes() -> dict[str, Any]:
    expected = read_json(SOURCE / "file_sha256.json")
    mismatches = [
        relative for relative, expected_hash in expected.items()
        if not (SOURCE / relative).is_file() or digest(SOURCE / relative) != expected_hash
    ]
    manifest = read_json(SOURCE / "manifest.json")
    return {
        "checked_files": len(expected),
        "mismatches": mismatches,
        "manifest_file_sha256_matches": manifest.get("file_sha256_sha256") == digest(SOURCE / "file_sha256.json"),
    }


def input_lock() -> tuple[dict[str, Any], str]:
    path = OUT / "checkpoint_01_input_lock.json"
    inputs = {str(item): digest(item) for item in source_paths()}
    source_audit = verify_source_hashes()
    require(not source_audit["mismatches"], "source_file_sha256_mismatch")
    require(source_audit["manifest_file_sha256_matches"], "source_manifest_file_sha256_mismatch")
    if path.exists():
        saved = read_json(path)
        require(saved.get("stage") == "map_terminal_tail_d1_v1_input_lock", "invalid_input_lock_stage")
        require(saved.get("inputs") == inputs, "input_lock_source_hash_changed")
        return saved, digest(path)
    value = {
        "stage": "map_terminal_tail_d1_v1_input_lock",
        "created_at_utc": now(),
        "classification": "exploratory_nonconfirmatory_d1_only_structure_audit",
        "inputs": inputs,
        "source_hash_audit": source_audit,
        "e1_path_access_count": 0,
        "forbidden_field_access_count": 0,
        "allowed_columns": ALLOWED_COLUMNS,
    }
    write_json_new(path, value)
    return value, digest(path)


def build_ledger(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | float]]:
    required = set(ALLOWED_COLUMNS)
    require(required.issubset(frame.columns), "missing_allowed_columns")
    require(int(frame["primary_eligible"].isna().sum()) == 0, "missing_primary_eligible")
    primary = frame.loc[frame["primary_eligible"].astype(bool)].copy()
    require(np.isfinite(primary["event_tick"].to_numpy(dtype=float)).all(), "nonfinite_event_tick")
    focal = primary[["physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id"]].rename(columns={"focal_player_id": "player_id"})
    opponent = primary[["physical_event_id", "match_id", "map_id", "event_tick", "opponent_player_id"]].rename(columns={"opponent_player_id": "player_id"})
    require(int(focal.isna().sum().sum()) == 0 and int(opponent.isna().sum().sum()) == 0, "missing_ledger_identity")
    ledger = pd.concat([focal, opponent], ignore_index=True).drop_duplicates().copy()
    group_columns = ["match_id", "map_id", "player_id"]
    terminal_tick = ledger.groupby(group_columns)["event_tick"].transform("max")
    ledger["is_map_terminal"] = ledger["event_tick"].eq(terminal_tick)
    terminal = ledger.loc[ledger["is_map_terminal"]]
    terminal_sizes = terminal.groupby(group_columns).size()
    metrics: dict[str, int | float] = {
        "physical_events": int(len(primary)),
        "ledger_rows": int(len(ledger)),
        "map_player_runs": int(ledger.groupby(group_columns).ngroups),
        "terminal_rows": int(len(terminal)),
        "terminal_fraction": float(len(terminal) / len(ledger)),
        "multi_terminal_runs": int((terminal_sizes > 1).sum()),
        "unclassified_rows": int(ledger["is_map_terminal"].isna().sum()),
        "terminal_definition_violations": int((terminal["event_tick"] != terminal_tick.loc[terminal.index]).sum()),
    }
    return ledger, metrics


def summary_matches(metrics: dict[str, int | float], dataset: str) -> bool:
    expected = EXPECTED[dataset]
    return (
        all(int(metrics[key]) == expected[key] for key in ("physical_events", "ledger_rows", "map_player_runs", "terminal_rows", "multi_terminal_runs"))
        and math.isclose(float(metrics["terminal_fraction"]), expected["terminal_fraction"], rel_tol=0.0, abs_tol=1e-12)
    )


def preflight(dataset: str, lock_hash: str) -> dict[str, Any]:
    source_path = SOURCE / "state" / dataset / "history_state.parquet"
    source_checkpoint = SOURCE / "state" / dataset / f"checkpoint_02_history_{dataset}.json"
    checkpoint = read_json(source_checkpoint)
    require(checkpoint.get("state_sha256") == digest(source_path), f"source_state_hash_mismatch:{dataset}")
    frame = pd.read_parquet(source_path, columns=ALLOWED_COLUMNS)
    _, metrics = build_ledger(frame)
    g0_audit = verify_source_hashes()
    g0_pass = not g0_audit["mismatches"] and g0_audit["manifest_file_sha256_matches"]
    g1_pass = bool(
        summary_matches(metrics, dataset) and metrics["unclassified_rows"] == 0
        and metrics["terminal_definition_violations"] == 0
    )
    g2_pass = bool(float(metrics["terminal_fraction"]) <= 0.05 and metrics["terminal_definition_violations"] == 0)
    value = {
        "stage": "map_terminal_tail_d1_v1_ledger_preflight",
        "dataset": dataset,
        "created_at_utc": now(),
        "input_lock_sha256": lock_hash,
        "source_state_sha256": digest(source_path),
        "metrics": metrics,
        "expected_metrics": EXPECTED[dataset],
        "G0": {"status": "pass" if g0_pass else "fail", "reason": "source_hash_chain_verified" if g0_pass else "source_hash_chain_mismatch"},
        "G1": {"status": "pass" if g1_pass else "fail", "reason": "participant_event_ledger_exact_reproduction"},
        "G2": {"status": "pass" if g2_pass else "fail", "reason": "map_terminal_tail_at_most_five_percent"},
        "G3": {"status": "pass", "reason": "structural_fields_only", "metrics": {"forbidden_field_access_count": 0, "allowed_columns": ALLOWED_COLUMNS}},
        "G4": {"status": "not_assessed", "reason": "tail_structure_only"},
        "G5": {"status": "pending", "reason": "no_complete_opportunity_model"},
    }
    path = OUT / "state" / dataset / f"checkpoint_02_ledger_preflight_{dataset}.json"
    if path.exists():
        saved = read_json(path)
        require(saved.get("input_lock_sha256") == lock_hash, f"preflight_input_lock_mismatch:{dataset}")
        require(saved.get("source_state_sha256") == digest(source_path), f"preflight_state_mismatch:{dataset}")
        return saved
    write_json_new(path, value)
    return value


def gates_for(value: dict[str, Any]) -> dict[str, Any]:
    audit_release = "pass" if all(value[name]["status"] == "pass" for name in ("G0", "G1", "G2", "G3")) else "fail"
    return {
        "G0": {**value["G0"], "metrics": {"e1_path_access_count": 0, "source_hash_audit": verify_source_hashes()}},
        "G1": {**value["G1"], "metrics": value["metrics"]},
        "G2": {**value["G2"], "metrics": {"terminal_fraction": value["metrics"]["terminal_fraction"], "threshold": 0.05}},
        "G3": value["G3"], "G4": value["G4"], "G5": value["G5"],
        "overall": {
            "audit_release": audit_release,
            "tail_release": "pass" if value["G2"]["status"] == "pass" else "fail",
            "interpretation_release": "map_terminal_tail_structure_only" if audit_release == "pass" else "blocked",
            "forbidden_interpretations": ["no_censoring", "random_censoring", "complete_opportunity_model", "state_effect", "causal_effect"],
        },
    }


def write_or_verify(path: Path, value: dict[str, Any], lock_hash: str) -> None:
    if path.exists():
        require(read_json(path).get("input_lock_sha256") == lock_hash, f"checkpoint_input_lock_mismatch:{path}")
        return
    write_json_new(path, value)


def output_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(OUT)): digest(path)
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"}
        and not any(part.startswith("failed_") for part in path.relative_to(OUT).parts)
    }


def run(selected: tuple[str, ...]) -> None:
    _, lock_hash = input_lock()
    values = {dataset: preflight(dataset, lock_hash) for dataset in selected}
    require(all(all(values[dataset][gate]["status"] == "pass" for gate in ("G0", "G1", "G3")) for dataset in selected), "hard_gate_failed")
    if set(selected) != set(DATASETS):
        print(json.dumps({"status": "partial_checkpoint_complete", "datasets": selected, "output_root": str(OUT)}, ensure_ascii=True))
        return

    gates = {dataset: gates_for(values[dataset]) for dataset in DATASETS}
    require(all(value["overall"]["audit_release"] == "pass" for value in gates.values()), "audit_release_failed")
    write_or_verify(OUT / "checkpoint_03_gate_audit.json", {
        "stage": "map_terminal_tail_d1_v1_gate_audit", "created_at_utc": now(),
        "input_lock_sha256": lock_hash, "gates": gates,
    }, lock_hash)
    hashes_path = OUT / "file_sha256.json"
    hashes = output_hashes()
    if hashes_path.exists():
        require(read_json(hashes_path) == hashes, "output_file_sha256_mismatch")
    else:
        write_json_new(hashes_path, hashes)
    manifest_path = OUT / "manifest.json"
    manifest = {
        "stage": "map_terminal_tail_d1_v1", "status": "passed", "created_at_utc": now(),
        "classification": "exploratory_nonconfirmatory_d1_only_structure_audit", "input_lock_sha256": lock_hash,
        "gate_statuses": {dataset: {name: value["status"] for name, value in gates[dataset].items() if name.startswith("G")} for dataset in DATASETS},
        "interpretation": "D1-only map-terminal tail structure; not a censoring, opportunity-process, state, or causal conclusion.",
        "file_sha256_sha256": digest(hashes_path),
    }
    if manifest_path.exists():
        saved = read_json(manifest_path)
        require(saved.get("input_lock_sha256") == lock_hash, "manifest_input_lock_mismatch")
        require(saved.get("file_sha256_sha256") == digest(hashes_path), "manifest_file_sha256_mismatch")
    else:
        write_json_new(manifest_path, manifest)
    print(json.dumps({"status": "passed", "manifest": str(manifest_path)}, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    try:
        run(tuple(args.datasets))
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True)
        failed = OUT / f"failed_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        failed.mkdir(exist_ok=False)
        write_json_new(failed / "failure_manifest.json", {
            "stage": "map_terminal_tail_d1_v1", "status": "failed", "created_at_utc": now(),
            "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
