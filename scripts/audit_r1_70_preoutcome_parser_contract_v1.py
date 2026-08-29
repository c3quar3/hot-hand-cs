"""Run a resumable, outcome-withheld parser-contract smoke for frozen R1 maps."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from datetime import UTC, datetime
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from demoparser2 import DemoParser


R1_ROOT = Path(r"C:\CS2_p_hat_v3_r1_preoutcome_intake_v1")
FREEZE = R1_ROOT / "p_hat_v3_r1_preoutcome_intake_freeze_v1.json"
RAW_ROOT = Path(r"D:\CS2_raw_matches")
SEVEN_ZIP = Path(r"C:\Program Files\7-Zip\7z.exe")
DEFAULT_OUTPUT = Path(r"C:\CS2_60s_confirmation_r1_70_preoutcome_v1\parser_contract_smoke_v1")
EXPECTED_FREEZE_SHA256 = "dfb2d2389e78ad53b263086e2f117f29b285e92d1202b67a1bb200590df7c4cb"
TICK_FIELDS = (
    "X", "Y", "Z",
    "CCSPlayerPawn.m_iHealth", "CCSPlayerPawn.m_ArmorValue", "CCSPlayerPawn.m_iTeamNum",
    "CCSPlayerController.m_bPawnIsAlive",
    "CCSPlayerController.CCSPlayerController_InGameMoneyServices.m_iAccount",
    "CCSPlayerPawn.m_unFreezetimeEndEquipmentValue",
    "CCSPlayerPawn.CCSPlayer_ItemServices.m_bHasHelmet",
    "CCSPlayerPawn.CCSPlayer_ItemServices.m_bHasDefuser",
    "CCSPlayerPawn.m_bIsDefusing", "CCSPlayerPawn.m_flFlashDuration",
    "CCSPlayerPawn.m_angEyeAngles", "Weapon.m_iItemDefinitionIndex", "Weapon.m_zoomLevel",
)
EVENTS = ("player_death", "weapon_fire", "round_freeze_end", "round_start", "bomb_planted")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def to_pandas(value: Any):
    converter = getattr(value, "to_pandas", None)
    return converter() if converter is not None else value


def validate_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    if not FREEZE.is_file() or sha256(FREEZE) != EXPECTED_FREEZE_SHA256:
        raise RuntimeError("r1_freeze_hash_mismatch")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    maps = freeze.get("maps", [])
    if freeze.get("archive_assignment_count") != 70 or len(maps) != 194:
        raise RuntimeError("r1_freeze_scope_not_70_matches_194_maps")
    maps = sorted(maps, key=lambda value: int(value["map_ordinal"]))
    if [int(value["map_ordinal"]) for value in maps] != list(range(1, 195)):
        raise RuntimeError("r1_map_ordinals_not_contiguous")
    item = maps[0]
    if str(item["source_match_id"]) != "2378781" or str(item["demo_member"]) != "falcons-vs-eternal-fire-m1-inferno.dem":
        raise RuntimeError("r1_prespecified_smoke_map_mismatch")
    return freeze, item


def extract_member(archive: Path, member: str, output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="temporary-", suffix=".dem", dir=output_root, delete=False) as handle:
        target = Path(handle.name)
        completed = subprocess.run(
            [str(SEVEN_ZIP), "e", "-so", "--", str(archive), member],
            stdout=handle,
            stderr=subprocess.PIPE,
            check=False,
        )
    if completed.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"seven_zip_extract_failed:{detail}")
    return target


def event_schema(parser: DemoParser, event_name: str) -> dict[str, object]:
    try:
        parsed = parser.parse_events([event_name])
        if len(parsed) != 1 or not isinstance(parsed[0], tuple) or len(parsed[0]) != 2:
            raise RuntimeError("unexpected_parser_result")
        frame = to_pandas(parsed[0][1])
        return {"available": True, "rows": int(len(frame)), "columns": [str(column) for column in frame.columns]}
    except Exception as exc:
        return {"available": False, "error_type": type(exc).__name__, "error": str(exc)[:500]}


def run_smoke(output_root: Path) -> dict[str, Any]:
    if not SEVEN_ZIP.is_file():
        raise RuntimeError("seven_zip_missing")
    freeze, item = validate_freeze()
    archive = RAW_ROOT / str(item["archive_filename"])
    expected = item["archive"]
    if not archive.is_file() or record(archive) != expected:
        raise RuntimeError("frozen_archive_changed_or_missing")

    staged = extract_member(archive, str(item["demo_member"]), output_root)
    try:
        parser = DemoParser(str(staged))
        header = parser.parse_header()
        updated_fields = sorted(str(value) for value in parser.list_updated_fields())
        field_set = set(updated_fields)
        event_schemas = {event: event_schema(parser, event) for event in EVENTS}
        missing_fields = sorted(set(TICK_FIELDS) - field_set)
        missing_events = sorted(event for event, value in event_schemas.items() if not value.get("available"))
        return {
            "stage": "r1_70_preoutcome_parser_contract_smoke_v1",
            "status": "passed" if not missing_fields and not missing_events else "contract_incomplete",
            "created_at_utc": now(),
            "outcome_access": "parser_schema_and_aggregate_counts_only",
            "not_materialized": ["event_rows", "attacker_or_victim_identity", "focal_wins", "p_hat", "residual", "history_state", "model_estimate"],
            "r1_freeze": record(FREEZE),
            "selection": {
                "selection_rule": "fixed_map_ordinal_1_only_for_predefined_minimal_smoke",
                "archive_assignment_count": freeze["archive_assignment_count"],
                "map_member_count": len(freeze["maps"]),
                "map": item,
                "archive": record(archive),
            },
            "parser": {
                "package": "demoparser2",
                "version": importlib.metadata.version("demoparser2"),
                "header_keys": sorted(str(key) for key in header.keys()),
                "updated_fields": updated_fields,
                "required_fields": list(TICK_FIELDS),
                "missing_required_fields": missing_fields,
                "event_schemas": event_schemas,
                "missing_required_events": missing_events,
            },
            "gates": {
                "G1_r1_freeze_verified": True,
                "G2_archive_verified": True,
                "G3_no_outcome_rows_persisted": True,
                "G4_parser_contract_complete": not missing_fields and not missing_events,
            },
        }
    finally:
        staged.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_root = args.output_root
    final = output_root / "manifest.json"
    if final.is_file():
        print(final.read_text(encoding="utf-8"))
        return
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("incomplete_output_root_requires_manual_review")
    output_root.mkdir(parents=True, exist_ok=True)
    payload = run_smoke(output_root)
    write_new_json(final, payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
