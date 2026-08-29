"""Audit outcome-free raw schemas for rebuilding the restart-track feature contract."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[3]
TRACK_ROOT = REPO_ROOT / "main" / "restart_20260825"
LOCK_ROOT = TRACK_ROOT / "data" / "confirmation_scopes_v1"
LOCK_MANIFEST = LOCK_ROOT / "manifest.json"
OUTPUT_ROOT = TRACK_ROOT / "data" / "confirmation_feature_audit_v1"

DATA_ROOTS = {
    "D1-B": Path(r"D:\CS2_p_hat_v3_d1_extracted_v1"),
    "D1-C": Path(r"D:\CS2_p_hat_v3_d1_extracted_v1"),
    "E1-v2": Path(r"D:\CS2_p_hat_v3_d1_extracted_e1v2"),
}
REQUIRED_COLUMNS = {
    "event_stream.parquet": {"tick", "event_type", "user_steamid"},
    "full_tick_player_state.parquet": {
        "tick", "player_id_hash", "CCSPlayerPawn.m_iHealth", "CCSPlayerPawn.m_ArmorValue",
        "CCSPlayerPawn.m_iTeamNum", "CCSPlayerController.m_bPawnIsAlive",
        "CCSPlayerController.CCSPlayerController_InGameMoneyServices.m_iAccount",
        "CCSPlayerPawn.m_unFreezetimeEndEquipmentValue",
        "CCSPlayerPawn.CCSPlayer_ItemServices.m_bHasHelmet",
        "CCSPlayerPawn.CCSPlayer_ItemServices.m_bHasDefuser", "CCSPlayerPawn.m_bIsDefusing",
        "CCSPlayerPawn.m_flFlashDuration", "CCSPlayerPawn.m_angEyeAngles",
    },
    "position_remedy.parquet": {"tick", "player_id_hash", "is_alive", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"},
    "context.parquet": {"round", "tick", "event_type"},
    "weapon_inventory.parquet": {"tick", "player_id_hash", "Weapon.m_iItemDefinitionIndex", "Weapon.m_zoomLevel"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def map_name(part: Path) -> str:
    manifest = json.loads((part / "map_manifest.json").read_text(encoding="utf-8"))
    return str(manifest["header"]["map_name"]).removeprefix("de_")


def source_parts(root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for match_dir in sorted((root / "maps").iterdir()):
        if not match_dir.is_dir():
            continue
        for part in sorted(match_dir.iterdir()):
            if part.is_dir() and (part / "map_manifest.json").is_file():
                result.setdefault(f"{match_dir.name}_{map_name(part)}", []).append(part)
    return result


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise RuntimeError(f"refusing_to_overwrite:{OUTPUT_ROOT}")
    lock = json.loads(LOCK_MANIFEST.read_text(encoding="utf-8"))
    if lock.get("outcome_access") != "not_read":
        raise RuntimeError("scope_lock_outcome_access_not_clean")
    all_rows = []
    datasets = []
    for dataset in lock["datasets"]:
        name = str(dataset["dataset"])
        scope_path = Path(dataset["locked_scope"]["path"])
        if sha256_file(scope_path) != dataset["locked_scope"]["sha256"]:
            raise RuntimeError(f"locked_scope_hash_mismatch:{name}")
        scope = pq.read_table(scope_path, columns=["map_id", "duel_id"]).to_pandas()
        parts_by_map = source_parts(DATA_ROOTS[name])
        records = []
        for map_id, rows in scope.groupby("map_id", sort=True):
            for part in parts_by_map.get(str(map_id), []):
                missing = {}
                for filename, needed in REQUIRED_COLUMNS.items():
                    present = set(pq.ParquetFile(part / filename).schema_arrow.names)
                    absent = sorted(needed - present)
                    if absent:
                        missing[filename] = absent
                records.append({"dataset": name, "map_id": str(map_id), "part_path": str(part), "scope_rows": int(len(rows)), "schema_complete": not missing, "missing_columns_json": json.dumps(missing, ensure_ascii=True, sort_keys=True)})
        frame = pd.DataFrame(records)
        if frame.empty or not frame["schema_complete"].all():
            raise RuntimeError(f"feature_schema_incomplete:{name}")
        all_rows.append(frame)
        datasets.append({"dataset": name, "locked_scope_sha256": dataset["locked_scope"]["sha256"], "maps": int(scope["map_id"].nunique()), "rows": int(len(scope)), "parts": int(len(frame)), "schema_complete": True})
    OUTPUT_ROOT.mkdir(parents=True)
    detail = pd.concat(all_rows, ignore_index=True)
    detail_path = OUTPUT_ROOT / "schema_detail.parquet"
    detail.to_parquet(detail_path, index=False, compression="zstd")
    manifest = {
        "stage": "restart_confirmation_feature_source_audit_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "outcome_access": "not_read",
        "lock_manifest_path": str(LOCK_MANIFEST), "lock_manifest_sha256": sha256_file(LOCK_MANIFEST),
        "script_path": str(Path(__file__).resolve()), "script_sha256": sha256_file(Path(__file__).resolve()),
        "required_columns": {key: sorted(value) for key, value in REQUIRED_COLUMNS.items()},
        "datasets": datasets,
        "detail_path": str(detail_path), "detail_sha256": sha256_file(detail_path),
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
