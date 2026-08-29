"""Build sparse, HMAC-keyed position/velocity tables for frozen R1 core maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd


CORE = Path(r"D:\CS2_60s_confirmation_r1_70_preoutcome_v1\core_tables_v1")
OUTPUT = Path(r"D:\CS2_60s_confirmation_r1_70_preoutcome_v1\sparse_positions_v1")
SEVEN_ZIP = Path(r"C:\Program Files\7-Zip\7z.exe")
POSITION_EXE = Path(r"C:\Users\Admin\Desktop\information flow for hot-hand-effect\CS\code\position_remedy\bin\Release\net9.0\DemoFilePositionExtractor.exe")
SALT = CORE / "access_control" / "r1_70_player_hmac_salt.bin"
MINIMUM_FREE_BYTES = 50 * 1024**3
LOOKBACK_TICKS = 640
ANCHOR_LAG_TICKS = 32


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


def planned_scope(scope: pd.DataFrame, events: pd.DataFrame, context: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    fires = events.loc[events["event_type"].eq("weapon_fire")].copy()
    fires["tick"] = pd.to_numeric(fires["tick"], errors="coerce")
    fire_by_player = {str(key): np.sort(group["tick"].dropna().astype("int64").to_numpy()) for key, group in fires.dropna(subset=["user_player_key"]).groupby("user_player_key", sort=False)}
    freeze = np.sort(pd.to_numeric(context.loc[context["event_type"].eq("round_freeze_end"), "tick"], errors="coerce").dropna().astype("int64").to_numpy())
    rows: list[dict[str, Any]] = []
    excluded = {"no_pair_weapon_fire_in_10s_window": 0, "no_round_freeze_end_before_anchor": 0}
    for row in scope.itertuples(index=False):
        event_tick = int(row.event_tick)
        candidates: list[int] = []
        for key in (str(row.focal_player_key), str(row.opponent_player_key)):
            values = fire_by_player.get(key, np.array([], dtype=np.int64))
            position = np.searchsorted(values, event_tick - LOOKBACK_TICKS, side="left")
            if position < len(values) and values[position] <= event_tick:
                candidates.append(int(values[position]))
        if not candidates:
            excluded["no_pair_weapon_fire_in_10s_window"] += 1
            continue
        first_fire = min(candidates)
        anchor = first_fire - ANCHOR_LAG_TICKS
        position = np.searchsorted(freeze, anchor, side="right") - 1
        if position < 0:
            excluded["no_round_freeze_end_before_anchor"] += 1
            continue
        rows.append({**row._asdict(), "first_fire_tick": first_fire, "anchor_tick": anchor, "post_buy_snapshot_tick": int(freeze[position])})
    return pd.DataFrame(rows), excluded


def extract_demo(archive: Path, member: str, staging: Path) -> Path:
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="active-", suffix=".dem", dir=staging, delete=False) as handle:
        demo = Path(handle.name)
        result = subprocess.run([str(SEVEN_ZIP), "e", "-so", "--", str(archive), member], stdout=handle, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0 or not demo.is_file() or demo.stat().st_size == 0:
        demo.unlink(missing_ok=True)
        raise RuntimeError("archive_member_extract_failed")
    return demo


def process(core_map: Path, output_root: Path) -> dict[str, Any]:
    source = json.loads((core_map / "map_manifest.json").read_text(encoding="utf-8"))
    selection = source["selection"]
    ordinal = int(selection["map_ordinal"])
    target = output_root / "maps" / core_map.name
    manifest_path = target / "map_manifest.json"
    if manifest_path.is_file():
        return {"map_ordinal": ordinal, "status": "skipped_completed", "manifest": record(manifest_path)}
    if target.exists() or target.with_name(target.name + ".partial").exists():
        raise RuntimeError("existing_incomplete_position_map")
    if shutil.disk_usage(output_root.drive + "\\").free < MINIMUM_FREE_BYTES:
        raise RuntimeError("output_drive_below_50_gib_floor")
    scope = pd.read_parquet(core_map / "outcome_free_duel_scope.parquet")
    events = pd.read_parquet(core_map / "event_stream.parquet")
    context = pd.read_parquet(core_map / "context.parquet")
    planned, excluded = planned_scope(scope, events, context)
    if planned.empty:
        raise RuntimeError("no_planned_rows_after_fixed_anchor_rule")
    ticks = set(planned["anchor_tick"].astype(int)) | set((planned["anchor_tick"] - 1).astype(int))
    archive = Path(str(selection["archive"]["path"]))
    if not archive.is_file() or record(archive) != selection["archive"]:
        raise RuntimeError("frozen_archive_changed_or_missing")
    demo = extract_demo(archive, str(selection["demo_member"]), output_root / "staging")
    csv = output_root / "staging" / f"positions_{ordinal:04d}.csv"
    meta = output_root / "staging" / f"positions_{ordinal:04d}.json"
    started = time.perf_counter()
    try:
        result = subprocess.run([str(POSITION_EXE), "--demo", str(demo), "--player-id-salt-file", str(SALT), "--output-csv", str(csv), "--output-meta", str(meta)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=1800)
        if result.returncode != 0 or not csv.is_file():
            raise RuntimeError("position_extractor_failed:" + result.stderr.decode("utf-8", errors="replace")[-500:])
        chunks: list[pd.DataFrame] = []
        for frame in pd.read_csv(csv, chunksize=500_000, dtype={"tick": "int64", "player_id_hash": "string"}):
            keep = frame.loc[frame["tick"].isin(ticks)].copy()
            if not keep.empty:
                chunks.append(keep)
        positions = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        if positions.empty:
            raise RuntimeError("no_sparse_positions_at_required_ticks")
        for axis in ("x", "y", "z"):
            positions[f"pos_{axis}"] = pd.to_numeric(positions[f"cell_{axis}"], errors="coerce") * 1024.0 + pd.to_numeric(positions[f"local_{axis}"], errors="coerce")
        positions = positions[["tick", "player_id_hash", "team_id", "is_alive", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"]].drop_duplicates(["tick", "player_id_hash"], keep="last").sort_values(["tick", "player_id_hash"], kind="stable")
        partial = target.with_name(target.name + ".partial")
        partial.mkdir(parents=True)
        planned_path = partial / "planned_scope.parquet"; position_path = partial / "position_remedy.parquet"
        planned.to_parquet(planned_path, index=False, compression="zstd"); positions.to_parquet(position_path, index=False, compression="zstd")
        manifest = {"stage": "r1_70_sparse_positions_v1", "status": "completed", "created_at_utc": now(), "source_core_map_manifest": record(core_map / "map_manifest.json"), "selection": selection, "planned_scope": record(planned_path), "position_remedy": record(position_path), "hmac_salt_sha256": sha256(SALT), "anchor_rule": "first weapon_fire by either strict participant in [event_tick-640,event_tick], minus 32 ticks", "anchor_strictly_before_event": bool((planned["anchor_tick"] < planned["event_tick"]).all()), "exclusions": excluded, "rows": {"input_scope": int(len(scope)), "planned_scope": int(len(planned)), "sparse_position": int(len(positions))}, "outcome_boundary": {"labels_read": False, "death_direction_persisted": False, "p_hat_or_residual_or_history_state_computed": False}, "position_extractor": record(POSITION_EXE), "elapsed_seconds": round(time.perf_counter() - started, 2)}
        write_json(partial / "map_manifest.json", manifest)
        partial.replace(target)
        return {"map_ordinal": ordinal, "status": "completed", "manifest": record(target / "map_manifest.json"), "rows": manifest["rows"], "elapsed_seconds": manifest["elapsed_seconds"]}
    finally:
        demo.unlink(missing_ok=True); csv.unlink(missing_ok=True); meta.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-root", type=Path, default=CORE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--max-maps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not (args.core_root / "manifest.json").is_file():
        raise SystemExit("core_preoutcome_manifest_required")
    if not all(path.is_file() for path in (SEVEN_ZIP, POSITION_EXE, args.core_root / "access_control" / SALT.name)):
        raise SystemExit("position_runtime_input_missing")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit("nonempty_output_requires_resume")
    args.output_root.mkdir(parents=True, exist_ok=True)
    core_maps = sorted((args.core_root / "maps").iterdir())
    progress_path = args.output_root / "progress.json"; prior = {}
    if progress_path.is_file():
        prior = {int(row["map_ordinal"]): row for row in json.loads(progress_path.read_text(encoding="utf-8")).get("runs", [])}
    runs: list[dict[str, Any]] = []; processed = 0
    for core_map in core_maps:
        source = json.loads((core_map / "map_manifest.json").read_text(encoding="utf-8")); ordinal = int(source["selection"]["map_ordinal"])
        if prior.get(ordinal, {}).get("status") in {"completed", "skipped_completed"}:
            runs.append(prior[ordinal]); continue
        if args.max_maps and processed >= args.max_maps:
            continue
        try: run = process(core_map, args.output_root)
        except Exception as exc: run = {"map_ordinal": ordinal, "status": "failed", "error": str(exc)[-1000:], "failed_at_utc": now()}
        runs.append(run); processed += 1
        write_json(progress_path, {"stage": "r1_70_sparse_positions_v1", "core_manifest": record(args.core_root / "manifest.json"), "script": record(Path(__file__).resolve()), "runs": runs, "updated_at_utc": now()}, replace=True)
        if run["status"] == "failed": raise RuntimeError("map_failed:" + run.get("error", "unknown"))
        print(f"[{processed}] {run['status']} map={ordinal}", flush=True)
    if len(runs) == len(core_maps) and all(run.get("status") in {"completed", "skipped_completed"} for run in runs):
        write_json(args.output_root / "manifest.json", {"stage": "r1_70_sparse_positions_v1", "status": "passed_outcome_withheld", "core_manifest": record(args.core_root / "manifest.json"), "script": record(Path(__file__).resolve()), "hmac_salt_sha256": sha256(SALT), "maps": len(core_maps), "runs": runs, "completed_at_utc": now()})


if __name__ == "__main__":
    main()
