"""Create a read-only completion verification package for the v4 analysis artifacts."""

from __future__ import annotations

import hashlib
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "results" / "d1a_dynamic_history_half_life_v4"
OUT = ROOT / "results" / "d1a_dynamic_history_half_life_v5_completion"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "75_D1A动态历史残差概率模型半衰期开发完整性核验锁_v1_2026-08-29.md"
V4_PROTOCOL = ROOT / "docs" / "修订与补丁" / "74_D1A动态历史残差概率模型半衰期开发执行锁_v4_2026-08-29.md"
V4_EXECUTOR = ROOT / "scripts" / "explore_d1a_dynamic_history_half_life_v4.py"
COMMON = ROOT / "scripts" / "explore_d1a_dynamic_history_half_life_v1.py"


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


def write_json(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def core_files() -> list[Path]:
    return sorted(path for path in V4.rglob("*") if path.is_file() and path.suffix != ".log" and path.name not in {"file_sha256.json", "manifest.json"})


def input_lock() -> tuple[dict[str, Any], str]:
    path = OUT / "checkpoint_01_input_lock.json"
    sources = [PROTOCOL, Path(__file__).resolve(), V4_PROTOCOL, V4_EXECUTOR, COMMON, V4 / "checkpoint_01_input_lock.json", V4 / "checkpoint_05_selection.json", V4 / "checkpoint_07_gate_audit.json", V4 / "file_sha256.json", V4 / "manifest.json", *core_files()]
    records = {str(item.relative_to(ROOT)): digest(item) for item in sources}
    if path.exists():
        saved = read_json(path)
        require(saved.get("inputs") == records, "input_lock_source_hash_changed")
        return saved, digest(path)
    value = {"stage": "d1a_dynamic_history_half_life_v5_completion_input_lock", "created_at_utc": now(), "classification": "read_only_completion_verification_of_exploratory_result_informed_d1_analysis", "inputs": records, "v4_log_policy": "exclude_execution_logs_from_core_hash_reverification; record_full_stdout_log_discrepancy", "e1_path_access_count": 0}
    write_json(path, value)
    return value, digest(path)


def run() -> None:
    _, lock_hash = input_lock()
    expected = read_json(V4 / "file_sha256.json")
    discrepancies = {name: {"expected": value, "observed": digest(V4 / name) if (V4 / name).is_file() else None} for name, value in expected.items() if not (V4 / name).is_file() or digest(V4 / name) != value}
    core_discrepancies = {name: value for name, value in discrepancies.items() if Path(name).suffix != ".log"}
    require(not core_discrepancies, f"v4_core_hash_mismatch:{sorted(core_discrepancies)}")
    require(set(discrepancies) == {"full_stdout.log"}, f"unexpected_v4_log_discrepancy:{sorted(discrepancies)}")
    full_stdout = (V4 / "full_stdout.log").read_text(encoding="utf-8").strip()
    require(full_stdout.startswith('{"status": "passed"') and "M3_h2" in full_stdout, "unexpected_full_stdout_content")
    manifest, gates, selection = read_json(V4 / "manifest.json"), read_json(V4 / "checkpoint_07_gate_audit.json"), read_json(V4 / "checkpoint_05_selection.json")
    require(all(gates["gates"][name]["status"] == "pass" for name in ("G0", "G1", "G2", "G3", "G4", "G5")), "v4_gate_not_passed")
    require(selection["selected"]["name"] == manifest["selected_model"]["name"] == "M3_h2", "selected_model_inconsistency")
    audit_path = OUT / "checkpoint_02_v4_ledger_audit.json"
    write_json(audit_path, {"stage": "d1a_dynamic_history_half_life_v5_v4_ledger_audit", "created_at_utc": now(), "input_lock_sha256": lock_hash, "v4_ledger_entries": len(expected), "core_entries_passed": len(expected) - 1, "registered_log_discrepancy": discrepancies, "v4_manifest_file_sha256_matches": manifest.get("file_sha256_sha256") == digest(V4 / "file_sha256.json")})
    summary_path = OUT / "checkpoint_03_completion_summary.json"
    write_json(summary_path, {"stage": "d1a_dynamic_history_half_life_v5_completion_summary", "created_at_utc": now(), "input_lock_sha256": lock_hash, "actual_d1a_folds": [{"name": "F1", "train": [101, 150], "evaluation": [151, 200]}, {"name": "F2", "train": [101, 200], "evaluation": [201, 250]}, {"name": "F3", "train": [101, 250], "evaluation": [251, 300]}], "selected_model": selection["selected"], "d1b_fixed_score": read_json(V4 / "state" / "D1-B" / "checkpoint_06_fixed_score.json"), "d1c_fixed_score": read_json(V4 / "state" / "D1-C" / "checkpoint_06_fixed_score.json"), "v4_stage_label_note": "v4 inherited v1 stage strings, but its input lock and three OOF checkpoints establish the authoritative v4 three-fold execution."})
    hashes_path = OUT / "file_sha256.json"
    hashes = {str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name != "file_sha256.json"}
    write_json(hashes_path, hashes)
    manifest_path = OUT / "manifest.json"
    write_json(manifest_path, {"stage": "d1a_dynamic_history_half_life_v5_completion", "status": "passed", "created_at_utc": now(), "classification": "read_only_completion_verification_of_exploratory_result_informed_d1_analysis", "input_lock_sha256": lock_hash, "v4_core_hash_status": "passed", "registered_v4_log_discrepancy": "full_stdout.log_only", "selected_model": selection["selected"], "file_sha256_sha256": digest(hashes_path)})
    print(json.dumps({"status": "passed", "manifest": str(manifest_path)}, ensure_ascii=True))


def main() -> None:
    try:
        run()
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True)
        failed = OUT / f"failed_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        failed.mkdir(exist_ok=False)
        write_json(failed / "failure_manifest.json", {"stage": "d1a_dynamic_history_half_life_v5_completion", "status": "failed", "created_at_utc": now(), "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
