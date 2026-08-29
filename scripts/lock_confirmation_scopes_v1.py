"""Lock outcome-free candidate scopes and raw-feature availability for the restart track."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[3]
TRACK_ROOT = REPO_ROOT / "main" / "restart_20260825"
PROTOCOL_PATH = TRACK_ROOT / "docs" / "00_重启研究协议_v1_2026-08-25.md"
OUTPUT_ROOT = TRACK_ROOT / "data" / "confirmation_scopes_v1"

SOURCES = {
    "D1-B": {
        "root": Path(r"D:\CS2_p_hat_v3_d1_extracted_v1"),
        "candidate": Path(r"D:\CS2_p_hat_v3_d1_extracted_v1\candidate\p_hat_v7a_d1_v1b_candidate.parquet"),
        "source_partition": "D1-B",
    },
    "D1-C": {
        "root": Path(r"D:\CS2_p_hat_v3_d1_extracted_v1"),
        "candidate": Path(r"D:\CS2_p_hat_v3_d1_extracted_v1\candidate\p_hat_v7a_d1_v1b_candidate.parquet"),
        "source_partition": "D1-C",
    },
    "E1-v2": {
        "root": Path(r"D:\CS2_p_hat_v3_d1_extracted_e1v2"),
        "candidate": Path(r"D:\CS2_p_hat_v3_d1_extracted_e1v2\candidate\p_hat_v3_d1_e1v2_candidate_outcome_withheld.parquet"),
        "source_partition": "E1-V2",
    },
}

# This intentionally excludes outcome files.  The files are sufficient to rebuild every
# model covariate once the fresh training contract is frozen.
REQUIRED_RAW_FILES = (
    "map_manifest.json",
    "event_stream.parquet",
    "full_tick_player_state.parquet",
    "position_remedy.parquet",
    "context.parquet",
    "weapon_inventory.parquet",
)
SCOPE_COLUMNS = (
    "duel_id", "match_id", "map_id", "partition", "queue_order", "event_tick",
    "focal_player_key", "opponent_player_key", "map_name",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def map_name(part: Path) -> str:
    manifest = json.loads((part / "map_manifest.json").read_text(encoding="utf-8"))
    return str(manifest["header"]["map_name"]).removeprefix("de_")


def raw_index(root: Path) -> dict[str, list[Path]]:
    indexed: dict[str, list[Path]] = {}
    for match_dir in sorted((root / "maps").iterdir()):
        if not match_dir.is_dir():
            continue
        for part in sorted(match_dir.iterdir()):
            if part.is_dir() and (part / "map_manifest.json").is_file():
                indexed.setdefault(f"{match_dir.name}_{map_name(part)}", []).append(part)
    return indexed


def schema_record(path: Path) -> list[dict[str, str]]:
    return [{"name": field.name, "type": str(field.type)} for field in pq.ParquetFile(path).schema_arrow]


def lock_one(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    candidate_path = Path(spec["candidate"])
    expected_partition = str(spec["source_partition"])
    available = set(pq.ParquetFile(candidate_path).schema_arrow.names)
    missing = sorted(set(SCOPE_COLUMNS) - available)
    if missing:
        raise RuntimeError(f"scope_columns_missing:{name}:{','.join(missing)}")
    scope = pq.read_table(candidate_path, columns=list(SCOPE_COLUMNS)).to_pandas()
    scope = scope.loc[scope["partition"].astype(str).eq(expected_partition)].copy()
    if scope.empty or scope["duel_id"].duplicated().any() or scope["duel_id"].isna().any():
        raise RuntimeError(f"invalid_scope_duel_id:{name}")
    if scope["match_id"].isna().any() or scope["map_id"].isna().any():
        raise RuntimeError(f"invalid_scope_map_key:{name}")
    if scope.groupby("match_id")["queue_order"].nunique().gt(1).any():
        raise RuntimeError(f"match_queue_order_not_one_to_one:{name}")

    raw = raw_index(Path(spec["root"]))
    map_rows: list[dict[str, Any]] = []
    for map_id, rows in scope.groupby("map_id", sort=True):
        parts = raw.get(str(map_id), [])
        missing_files = [str(part / filename) for part in parts for filename in REQUIRED_RAW_FILES if not (part / filename).is_file()]
        map_rows.append({
            "map_id": str(map_id), "rows": int(len(rows)), "matches": int(rows["match_id"].nunique()),
            "raw_part_count": len(parts), "complete_raw_input": bool(parts) and not missing_files,
            "missing_files": missing_files,
        })
    map_audit = pd.DataFrame(map_rows)
    complete_map_ids = set(map_audit.loc[map_audit["complete_raw_input"], "map_id"])
    complete_scope = scope.loc[scope["map_id"].astype(str).isin(complete_map_ids)].copy()
    out_dir = OUTPUT_ROOT / name
    if out_dir.exists():
        raise RuntimeError(f"refusing_to_overwrite_output:{out_dir}")
    out_dir.mkdir(parents=True)
    scope_path = out_dir / "candidate_scope_outcome_free.parquet"
    complete_scope.to_parquet(scope_path, index=False, compression="zstd")
    map_audit.to_parquet(out_dir / "raw_availability_audit.parquet", index=False, compression="zstd")

    return {
        "dataset": name,
        "source_partition": expected_partition,
        "source_candidate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path), "schema": schema_record(candidate_path)},
        "locked_scope": {"path": str(scope_path), "sha256": sha256_file(scope_path), "rows": int(len(complete_scope)), "matches": int(complete_scope["match_id"].nunique()), "maps": int(complete_scope["map_id"].nunique())},
        "structurally_excluded": {"rows": int(len(scope) - len(complete_scope)), "matches": int(scope.loc[~scope.index.isin(complete_scope.index), "match_id"].nunique()), "maps": int((~map_audit["complete_raw_input"]).sum())},
        "raw_availability_audit": {"path": str(out_dir / "raw_availability_audit.parquet"), "sha256": sha256_file(out_dir / "raw_availability_audit.parquet")},
        "scope_columns": list(SCOPE_COLUMNS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("lock",))
    args = parser.parse_args()
    if not PROTOCOL_PATH.is_file():
        raise RuntimeError(f"protocol_missing:{PROTOCOL_PATH}")
    if OUTPUT_ROOT.exists():
        raise RuntimeError(f"refusing_to_overwrite_track_lock:{OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    datasets = [lock_one(name, spec) for name, spec in SOURCES.items()]
    manifest = {
        "stage": "restart_confirmation_scope_lock_v1", "created_at_utc": utc_now(),
        "outcome_access": "not_read", "forbidden_files": ["outcome_withheld.parquet"],
        "protocol_path": str(PROTOCOL_PATH), "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "script_path": str(Path(__file__).resolve()), "script_sha256": sha256_file(Path(__file__).resolve()),
        "required_raw_files": list(REQUIRED_RAW_FILES), "datasets": datasets,
    }
    write_new_json(OUTPUT_ROOT / "manifest.json", manifest)
    print(json.dumps({"manifest": str(OUTPUT_ROOT / "manifest.json"), "datasets": [{"name": row["dataset"], **row["locked_scope"]} for row in datasets]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
