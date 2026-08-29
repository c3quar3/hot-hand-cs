"""Run the frozen metadata-only support preflight for the independent head-to-head study."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data" / "future_independent_confirmation_candidates_v1" / "candidate_archives.csv"
V2 = ROOT / "data" / "future_independent_confirmation_candidates_v2" / "additional_archives.csv"
R1 = Path(r"C:\CS2_p_hat_v3_r1_preoutcome_intake_v1\p_hat_v3_r1_preoutcome_intake_freeze_v1.json")
PROTOCOL = ROOT / "docs" / "协议" / "21_独立60秒相对lag-1预检执行锁_v1_2026-08-27.md"
OUT = ROOT / "results" / "independent_60s_lag1_head_to_head_preflight_v1"
MINIMUM_ARCHIVES = 60


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if (OUT / "manifest.json").is_file():
        manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "stopped_preoutcome_insufficient_nonoverlapping_archives":
            raise RuntimeError("completed_output_status_mismatch")
        print("completed_preflight_verified")
        return
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("incomplete_output_requires_manual_audit")
    if not all(path.is_file() for path in (V1, V2, R1, PROTOCOL)):
        raise RuntimeError("locked_input_missing")

    v1, v2 = load_rows(V1), load_rows(V2)
    candidates = {row["source_archive"]: row for row in v1 + v2}
    r1 = json.loads(R1.read_text(encoding="utf-8"))
    r1_names = {row["archive_filename"] for row in r1["archive_assignments"]}
    ordered = [candidates[key] for key in sorted(candidates)]
    overlap = [row for row in ordered if Path(row["source_archive"]).name in r1_names]
    remaining = [row for row in ordered if Path(row["source_archive"]).name not in r1_names]

    OUT.mkdir(parents=True)
    lock = {
        "stage": "independent_60s_lag1_head_to_head_preflight_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "v1_sha256": digest(V1), "v2_sha256": digest(V2), "r1_freeze_sha256": digest(R1),
        "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__).resolve()),
        "minimum_archives": MINIMUM_ARCHIVES,
        "outcome_access": "not_read",
    }
    write_json(OUT / "input_lock.json", lock)
    for name, rows in (("excluded_r1_overlap_archives.csv", overlap), ("remaining_candidate_archives.csv", remaining)):
        target = OUT / name
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_archive", "bytes", "last_write_utc"])
            writer.writeheader(); writer.writerows(rows)
    p0 = len(ordered) == 78 and len(r1_names) == 70
    p1 = len(overlap) == 70 and len(overlap) + len(remaining) == len(ordered)
    p2 = len(remaining) >= MINIMUM_ARCHIVES
    gates = {
        "P0_input": {"status": "pass" if p0 else "fail", "candidate_archives": len(ordered), "r1_archives": len(r1_names)},
        "P1_r1_exclusion": {"status": "pass" if p1 else "fail", "r1_basename_overlap": len(overlap), "remaining_archives": len(remaining)},
        "P2_c2_conservative_support_upper_bound": {"status": "pass" if p2 else "fail", "upper_bound_matches": len(remaining), "minimum_required_matches": MINIMUM_ARCHIVES},
    }
    write_json(OUT / "preflight.json", {"stage": lock["stage"], "outcome_access": "not_read", "gates": gates})
    hashes = {path.name: digest(path) for path in (OUT / "input_lock.json", OUT / "excluded_r1_overlap_archives.csv", OUT / "remaining_candidate_archives.csv", OUT / "preflight.json")}
    write_json(OUT / "file_sha256.json", hashes)
    status = "passed_preoutcome" if p0 and p1 and p2 else "stopped_preoutcome_insufficient_nonoverlapping_archives"
    write_json(OUT / "manifest.json", {"stage": lock["stage"], "status": status, "outcome_access": "not_read", "gate_statuses": {key: value["status"] for key, value in gates.items()}, "input_lock_sha256": digest(OUT / "input_lock.json"), "file_sha256_sha256": digest(OUT / "file_sha256.json"), "interpretation": "Metadata-only preflight. No archive contents, outcomes, labels, residuals, states, model fits, or losses were accessed."})
    print(status)


if __name__ == "__main__":
    main()
