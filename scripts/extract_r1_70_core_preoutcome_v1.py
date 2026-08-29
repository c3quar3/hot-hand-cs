"""Resumable outcome-withheld core-table extraction for the frozen R1 70-match scope."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from demoparser2 import DemoParser


ROOT = Path(__file__).resolve().parents[1]
R1_ROOT = Path(r"C:\CS2_p_hat_v3_r1_preoutcome_intake_v1")
FREEZE = R1_ROOT / "p_hat_v3_r1_preoutcome_intake_freeze_v1.json"
RAW_ROOT = Path(r"D:\CS2_raw_matches")
SEVEN_ZIP = Path(r"C:\Program Files\7-Zip\7z.exe")
OUTPUT = Path(r"D:\CS2_60s_confirmation_r1_70_preoutcome_v1\core_tables_v1")
PILOT_SOURCE = R1_ROOT / "protocol" / "source" / "run_stage4_external_hltv_pilot.py"
EXPECTED_FREEZE_SHA256 = "dfb2d2389e78ad53b263086e2f117f29b285e92d1202b67a1bb200590df7c4cb"
MINIMUM_FREE_BYTES = 80 * 1024**3
SALT_NAME = Path("access_control") / "r1_70_player_hmac_salt.bin"
PROGRESS_NAME = "progress.json"
FINAL_NAME = "manifest.json"

STATE_FIELDS = (
    "CCSPlayerPawn.m_iHealth", "CCSPlayerPawn.m_ArmorValue", "CCSPlayerPawn.m_iTeamNum",
    "CCSPlayerController.m_bPawnIsAlive",
    "CCSPlayerController.CCSPlayerController_InGameMoneyServices.m_iAccount",
    "CCSPlayerPawn.m_unFreezetimeEndEquipmentValue",
    "CCSPlayerPawn.CCSPlayer_ItemServices.m_bHasHelmet",
    "CCSPlayerPawn.CCSPlayer_ItemServices.m_bHasDefuser",
    "CCSPlayerPawn.m_bIsDefusing", "CCSPlayerPawn.m_flFlashDuration", "CCSPlayerPawn.m_angEyeAngles",
)
WEAPON_FIELDS = ("Weapon.m_iItemDefinitionIndex", "Weapon.m_zoomLevel")
CONTEXT_EVENTS = ("round_start", "round_freeze_end", "bomb_planted")


def now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: dict[str, Any], replace: bool = False) -> None:
    if path.exists() and not replace:
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def to_pandas(value: Any) -> pd.DataFrame:
    converter = getattr(value, "to_pandas", None)
    return converter() if converter is not None else value.copy()


def load_pilot():
    specification = importlib.util.spec_from_file_location("r1_pilot_frozen", PILOT_SOURCE)
    if specification is None or specification.loader is None:
        raise RuntimeError("frozen_r1_pilot_load_failed")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def validate_freeze() -> list[dict[str, Any]]:
    if not FREEZE.is_file() or sha256(FREEZE) != EXPECTED_FREEZE_SHA256:
        raise RuntimeError("r1_freeze_hash_mismatch")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    maps = sorted(freeze.get("maps", []), key=lambda item: int(item["map_ordinal"]))
    if freeze.get("archive_assignment_count") != 70 or len(maps) != 194:
        raise RuntimeError("r1_scope_not_70_matches_194_maps")
    if [int(item["map_ordinal"]) for item in maps] != list(range(1, 195)):
        raise RuntimeError("r1_map_ordinals_invalid")
    return maps


def ensure_salt(output_root: Path, resume: bool) -> Path:
    path = output_root / SALT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if resume:
            raise RuntimeError("missing_hmac_salt_while_resuming")
        with path.open("xb") as handle:
            handle.write(os.urandom(32))
    if path.stat().st_size < 32:
        raise RuntimeError("invalid_hmac_salt")
    return path


def hmac_key(salt: bytes, value: object) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return ""
    return hmac.new(salt, str(number).encode("ascii"), hashlib.sha256).hexdigest() if number > 0 else ""


def hmac_state(frame: pd.DataFrame, salt: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "steamid" not in frame.columns:
        raise RuntimeError("state_missing_transient_steamid")
    raw = pd.to_numeric(frame["steamid"], errors="coerce")
    result = frame.copy()
    result["player_id_hash"] = raw.map(lambda value: hmac_key(salt, value)).astype("string")
    crosswalk = pd.DataFrame({"player_id_hash": result["player_id_hash"], "steamid": raw.astype("Int64")})
    crosswalk = crosswalk.loc[crosswalk["player_id_hash"].ne("") & crosswalk["steamid"].notna()].drop_duplicates()
    result = result.drop(columns=[column for column in ("steamid", "name") if column in result.columns])
    if result["player_id_hash"].eq("").any():
        raise RuntimeError("unmapped_player_state")
    return result, crosswalk


def extract_member(archive: Path, member: str, staging: Path) -> Path:
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="active-", suffix=".dem", dir=staging, delete=False) as handle:
        target = Path(handle.name)
        completed = subprocess.run([str(SEVEN_ZIP), "e", "-so", "--", str(archive), member], stdout=handle, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        raise RuntimeError("archive_member_extraction_failed:" + completed.stderr.decode("utf-8", errors="replace")[-500:])
    return target


def event_frame(parser: DemoParser, name: str) -> pd.DataFrame:
    parsed = parser.parse_events([name])
    if len(parsed) != 1 or not isinstance(parsed[0], tuple) or len(parsed[0]) != 2:
        raise RuntimeError(f"unexpected_event_result:{name}")
    return to_pandas(parsed[0][1])


def public_event(frame: pd.DataFrame, event_type: str, salt: bytes) -> pd.DataFrame:
    result = pd.DataFrame({"tick": pd.to_numeric(frame.get("tick"), errors="coerce"), "event_type": event_type})
    if "user_steamid" in frame.columns:
        result["user_player_key"] = frame["user_steamid"].map(lambda value: hmac_key(salt, value)).astype("string")
    result = result.dropna(subset=["tick"]).copy()
    result["tick"] = result["tick"].astype("int64")
    return result


def scoped_duels(pilot: Any, staged: Path, item: dict[str, Any], salt: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    snapshots, duels, _, summary = pilot.parse_map(
        demo=staged,
        source_match_id=str(item["source_match_id"]),
        source_map_id=f"r1m{int(item['map_ordinal']):04d}",
        match_date_utc=str(item["match_date_utc"]),
        tick_rate=64.0,
        player_id_key=salt,
    )
    focal = snapshots[["event_tick", "player_slot", "player_id_hash"]].drop_duplicates().rename(columns={"player_slot": "focal_player_slot", "player_id_hash": "focal_player_key"})
    opponent = snapshots[["event_tick", "player_slot", "player_id_hash"]].drop_duplicates().rename(columns={"player_slot": "opponent_player_slot", "player_id_hash": "opponent_player_key"})
    scope = duels.merge(focal, on=["event_tick", "focal_player_slot"], how="left", validate="many_to_one")
    scope = scope.merge(opponent, on=["event_tick", "opponent_player_slot"], how="left", validate="many_to_one")
    scope = scope.dropna(subset=["focal_player_key", "opponent_player_key"]).copy()
    keep = ["duel_id", "match_id", "map_id", "round", "event_tick", "pre_event_tick", "focal_player_key", "opponent_player_key", "atomic_tick_batch_id", "atomic_tick_batch_size", "atomic_tick_participant_count", "atomic_tick_is_multidual", "atomic_tick_shared_participant", "atomic_tick_history_policy"]
    scope = scope[keep].drop_duplicates("duel_id").sort_values(["event_tick", "duel_id"], kind="stable")
    return scope, {"strict_duel_count": int(len(scope)), "pilot_summary": {"map_name": summary.get("map_name"), "missing_parser_fields": summary.get("missing_parser_fields", [])}}


def map_directory(output_root: Path, item: dict[str, Any]) -> Path:
    slug = "".join(char if char.isalnum() or char in "._-" else "_" for char in Path(str(item["demo_member"])).stem)
    return output_root / "maps" / f"{int(item['map_ordinal']):04d}_{item['source_match_id']}_{slug}"


def process_map(pilot: Any, item: dict[str, Any], output_root: Path, salt: bytes) -> dict[str, Any]:
    directory = map_directory(output_root, item)
    manifest_path = directory / "map_manifest.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("status") == "completed" and prior.get("selection") == item:
            return {"map_ordinal": item["map_ordinal"], "status": "skipped_completed", "map_manifest": record(manifest_path)}
        raise RuntimeError("incompatible_completed_map")
    if directory.exists() or directory.with_name(directory.name + ".partial").exists():
        raise RuntimeError("incomplete_or_existing_map_directory")
    drive = output_root.drive + "\\"
    if shutil.disk_usage(drive).free < MINIMUM_FREE_BYTES:
        raise RuntimeError("output_drive_below_80_gib_floor")
    archive = Path(str(item["archive"]["path"]))
    if not archive.is_file() or record(archive) != item["archive"]:
        raise RuntimeError("frozen_archive_changed_or_missing")

    staged = extract_member(archive, str(item["demo_member"]), output_root / "staging")
    started = time.perf_counter()
    try:
        parser = DemoParser(str(staged))
        header = parser.parse_header()
        available = set(str(field) for field in parser.list_updated_fields())
        missing = sorted(set(STATE_FIELDS + WEAPON_FIELDS) - available)
        if missing:
            raise RuntimeError("required_parser_fields_missing:" + ",".join(missing))
        states, crosswalk = hmac_state(to_pandas(parser.parse_ticks(list(STATE_FIELDS))), salt)
        states = states.sort_values(["tick", "player_id_hash"], kind="stable").reset_index(drop=True)
        weapon, weapon_crosswalk = hmac_state(to_pandas(parser.parse_ticks(list(WEAPON_FIELDS))), salt)
        weapon = weapon.sort_values(["tick", "player_id_hash"], kind="stable").reset_index(drop=True)
        fire = public_event(event_frame(parser, "weapon_fire"), "weapon_fire", salt)
        context = pd.concat([public_event(event_frame(parser, name), name, salt)[["tick", "event_type"]] for name in CONTEXT_EVENTS], ignore_index=True)
        scope, duel_audit = scoped_duels(pilot, staged, item, salt)
        partial = directory.with_name(directory.name + ".partial")
        partial.mkdir(parents=True)
        outputs: dict[str, dict[str, object]] = {}
        for name, frame in (("full_tick_player_state.parquet", states), ("weapon_inventory.parquet", weapon), ("event_stream.parquet", fire), ("context.parquet", context), ("outcome_free_duel_scope.parquet", scope)):
            path = partial / name
            frame.to_parquet(path, index=False, compression="zstd")
            outputs[name] = record(path)
        controlled = pd.concat([crosswalk, weapon_crosswalk], ignore_index=True).drop_duplicates().sort_values("player_id_hash")
        access_path = output_root / "access_control" / f"map_{int(item['map_ordinal']):04d}_identity_crosswalk.parquet"
        access_path.parent.mkdir(parents=True, exist_ok=True)
        if access_path.exists():
            raise RuntimeError("identity_crosswalk_overwrite")
        controlled.to_parquet(access_path, index=False, compression="zstd")
        manifest = {
            "stage": "r1_70_core_preoutcome_v1", "status": "completed", "created_at_utc": now(),
            "purpose": "full-tick outcome-free state, weapon, fire/context and canonical HMAC duel scope; no labels",
            "selection": item, "archive": record(archive), "raw_demo": {"bytes": staged.stat().st_size, "sha256": sha256(staged)},
            "header": {str(key): str(value) for key, value in header.items()}, "outputs": outputs,
            "access_control": {"identity_crosswalk": record(access_path), "raw_identity_persisted_outside_access_control": False},
            "outcome_boundary": {"player_death_read_transiently_for_strict_scope": True, "death_direction_persisted": False, "focal_wins_materialized": False, "p_hat_or_residual_or_history_state_computed": False},
            "duel_audit": duel_audit,
            "rows": {"state": int(len(states)), "weapon": int(len(weapon)), "weapon_fire": int(len(fire)), "context": int(len(context)), "duel_scope": int(len(scope))},
            "script_sha256": sha256(Path(__file__).resolve()), "pilot_source_sha256": sha256(PILOT_SOURCE), "completed_at_utc": now(),
        }
        write_json(partial / "map_manifest.json", manifest)
        partial.replace(directory)
        return {"map_ordinal": item["map_ordinal"], "status": "completed", "map_manifest": record(directory / "map_manifest.json"), "rows": manifest["rows"], "elapsed_seconds": round(time.perf_counter() - started, 2)}
    finally:
        staged.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--max-maps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_maps < 0:
        raise SystemExit("max_maps_must_be_nonnegative")
    if not SEVEN_ZIP.is_file() or not PILOT_SOURCE.is_file():
        raise SystemExit("required_runtime_input_missing")
    final = args.output_root / FINAL_NAME
    if final.is_file():
        raise SystemExit("completed_output_is_immutable")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit("nonempty_output_requires_resume")
    maps = validate_freeze()
    args.output_root.mkdir(parents=True, exist_ok=True)
    salt = ensure_salt(args.output_root, args.resume)
    pilot = load_pilot()
    progress_path = args.output_root / PROGRESS_NAME
    prior = {int(row["map_ordinal"]): row for row in json.loads(progress_path.read_text(encoding="utf-8")).get("runs", [])} if progress_path.is_file() else {}
    runs: list[dict[str, Any]] = []
    processed = 0
    for item in maps:
        ordinal = int(item["map_ordinal"])
        previous = prior.get(ordinal)
        if previous and previous.get("status") in {"completed", "skipped_completed"}:
            runs.append(previous)
            continue
        if args.max_maps and processed >= args.max_maps:
            continue
        try:
            run = process_map(pilot, item, args.output_root, salt.read_bytes())
        except Exception as exc:
            run = {"map_ordinal": ordinal, "status": "failed", "error": str(exc)[-1000:], "failed_at_utc": now()}
        runs.append(run); processed += 1
        write_json(progress_path, {"stage": "r1_70_core_preoutcome_v1", "r1_freeze": record(FREEZE), "script": record(Path(__file__).resolve()), "pilot_source": record(PILOT_SOURCE), "selection_count": len(maps), "runs": runs, "updated_at_utc": now()}, replace=True)
        if run["status"] == "failed":
            raise RuntimeError("map_failed:" + run.get("error", "unknown"))
        print(f"[{processed}] {run['status']} map={ordinal}", flush=True)
    complete = len(runs) == len(maps) and all(run.get("status") in {"completed", "skipped_completed"} for run in runs)
    if complete:
        write_json(final, {"stage": "r1_70_core_preoutcome_v1", "status": "passed_outcome_withheld", "r1_freeze": record(FREEZE), "script": record(Path(__file__).resolve()), "pilot_source": record(PILOT_SOURCE), "hmac_salt_sha256": sha256(salt), "maps": len(maps), "runs": runs, "completed_at_utc": now()})


if __name__ == "__main__":
    main()
