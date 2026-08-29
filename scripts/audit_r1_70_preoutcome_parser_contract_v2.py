"""Correct the position-alias check in the frozen R1 parser-contract smoke."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from demoparser2 import DemoParser

import audit_r1_70_preoutcome_parser_contract_v1 as v1


DEFAULT_OUTPUT = Path(r"C:\CS2_60s_confirmation_r1_70_preoutcome_v1\parser_contract_smoke_v2")
RAW_TICK_FIELDS = tuple(field for field in v1.TICK_FIELDS if field not in {"X", "Y", "Z"})
POSITION_ALIASES = ("X", "Y", "Z")


def round_start_tick(parser: DemoParser) -> int:
    parsed = parser.parse_events(["round_start"])
    if len(parsed) != 1 or not isinstance(parsed[0], tuple) or len(parsed[0]) != 2:
        raise RuntimeError("round_start_parser_result_invalid")
    frame = v1.to_pandas(parsed[0][1])
    if frame.empty or "tick" not in frame.columns:
        raise RuntimeError("round_start_tick_unavailable")
    return int(frame["tick"].iloc[0])


def run_smoke(output_root: Path) -> dict[str, Any]:
    if not v1.SEVEN_ZIP.is_file():
        raise RuntimeError("seven_zip_missing")
    freeze, item = v1.validate_freeze()
    archive = v1.RAW_ROOT / str(item["archive_filename"])
    expected = item["archive"]
    if not archive.is_file() or v1.record(archive) != expected:
        raise RuntimeError("frozen_archive_changed_or_missing")

    staged = v1.extract_member(archive, str(item["demo_member"]), output_root)
    try:
        parser = DemoParser(str(staged))
        header = parser.parse_header()
        updated_fields = set(str(value) for value in parser.list_updated_fields())
        event_schemas = {event: v1.event_schema(parser, event) for event in v1.EVENTS}
        probe_tick = round_start_tick(parser)
        positions = v1.to_pandas(parser.parse_ticks(list(POSITION_ALIASES), ticks=[probe_tick]))
        position_columns = [str(column) for column in positions.columns]
        missing_fields = sorted(set(RAW_TICK_FIELDS) - updated_fields)
        missing_positions = sorted(set(POSITION_ALIASES) - set(position_columns))
        missing_events = sorted(event for event, value in event_schemas.items() if not value.get("available"))
        return {
            "stage": "r1_70_preoutcome_parser_contract_smoke_v2",
            "status": "passed" if not missing_fields and not missing_positions and not missing_events else "contract_incomplete",
            "created_at_utc": v1.now(),
            "outcome_access": "parser_schema_and_aggregate_counts_only",
            "not_materialized": ["event_rows", "coordinates", "tick_values", "attacker_or_victim_identity", "focal_wins", "p_hat", "residual", "history_state", "model_estimate"],
            "supersedes": {"artifact": str(v1.DEFAULT_OUTPUT / "manifest.json"), "reason": "X/Y/Z are parse_ticks aliases rather than list_updated_fields members"},
            "r1_freeze": v1.record(v1.FREEZE),
            "selection": {
                "selection_rule": "fixed_map_ordinal_1_only_for_predefined_minimal_smoke",
                "archive_assignment_count": freeze["archive_assignment_count"],
                "map_member_count": len(freeze["maps"]),
                "map": item,
                "archive": v1.record(archive),
            },
            "parser": {
                "package": "demoparser2",
                "version": importlib.metadata.version("demoparser2"),
                "header_keys": sorted(str(key) for key in header.keys()),
                "raw_required_fields": list(RAW_TICK_FIELDS),
                "missing_raw_required_fields": missing_fields,
                "position_aliases": list(POSITION_ALIASES),
                "position_probe_columns": position_columns,
                "position_probe_row_count": int(len(positions)),
                "missing_position_aliases": missing_positions,
                "event_schemas": event_schemas,
                "missing_required_events": missing_events,
            },
            "gates": {
                "G1_r1_freeze_verified": True,
                "G2_archive_verified": True,
                "G3_no_outcome_rows_persisted": True,
                "G4_parser_contract_complete": not missing_fields and not missing_positions and not missing_events,
            },
        }
    finally:
        staged.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    final = args.output_root / "manifest.json"
    if final.is_file():
        print(final.read_text(encoding="utf-8"))
        return
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError("incomplete_output_root_requires_manual_review")
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = run_smoke(args.output_root)
    v1.write_new_json(final, payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
