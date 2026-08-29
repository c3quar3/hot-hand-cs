"""Stream full C# positions to preserve coordinate continuity, then retain anchors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

import extract_r1_70_sparse_positions_v1 as base


CELL_WIDTH = 1024.0


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: dict[str, Any], replace: bool = False) -> None:
    if path.exists() and not replace:
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sparse_continuous_positions(csv: Path, required_ticks: set[int]) -> pd.DataFrame:
    """Apply the historical cell-unwrapping rule before discarding non-anchor rows."""
    previous: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
    kept: list[pd.DataFrame] = []
    columns = ["tick", "player_id_hash", "team_id", "is_alive", "cell_x", "cell_y", "cell_z", "local_x", "local_y", "local_z", "vel_x", "vel_y", "vel_z"]
    for raw in pd.read_csv(csv, usecols=columns, chunksize=500_000, dtype={"player_id_hash": "string"}):
        raw = raw.dropna(subset=["tick", "player_id_hash"]).copy().reset_index(drop=True)
        raw["tick"] = pd.to_numeric(raw["tick"], errors="raise").astype("int64")
        raw["player_id_hash"] = raw["player_id_hash"].astype(str)
        for column in ("cell_x", "cell_y", "cell_z", "local_x", "local_y", "local_z", "vel_x", "vel_y", "vel_z"):
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
        keys = raw["player_id_hash"]
        prior_tick = raw.groupby("player_id_hash", sort=False)["tick"].shift(1)
        coordinate = raw[["cell_x", "cell_y", "cell_z"]].to_numpy(float) * CELL_WIDTH + raw[["local_x", "local_y", "local_z"]].to_numpy(float)
        prior_coordinate = pd.DataFrame(coordinate).groupby(keys, sort=False).shift(1).to_numpy(float).copy()
        first = prior_tick.isna()
        prior_tick.loc[first] = keys.loc[first].map(lambda key: previous.get(str(key), (-10**18, np.full(3, np.nan), np.zeros(3)))[0])
        for axis in range(3):
            prior_coordinate[first.to_numpy(), axis] = keys.loc[first].map(lambda key: previous.get(str(key), (-10**18, np.full(3, np.nan), np.zeros(3)))[1][axis]).to_numpy(float)
        adjacent = raw["tick"].sub(prior_tick).eq(1).to_numpy()
        adjustment = CELL_WIDTH * np.rint((coordinate - prior_coordinate) / CELL_WIDTH)
        adjustment[~adjacent | ~np.isfinite(adjustment).all(axis=1)] = 0.0
        segment = pd.Series((~adjacent).astype("int64")).groupby(keys, sort=False).cumsum()
        groups = pd.MultiIndex.from_arrays([keys, segment])
        cumulative = pd.DataFrame(adjustment).groupby(groups, sort=False).cumsum().to_numpy(float)
        seed = np.zeros_like(cumulative)
        for index in np.flatnonzero(first.to_numpy() & adjacent):
            seed[index] = previous[str(keys.iat[index])][2]
        seed_frame = pd.DataFrame(seed).groupby(groups, sort=False).transform("first").to_numpy(float)
        continuous = coordinate - cumulative - seed_frame
        for index in raw.groupby("player_id_hash", sort=False).tail(1).index:
            previous[str(raw.at[index, "player_id_hash"])] = (int(raw.at[index, "tick"]), coordinate[index], cumulative[index] + seed_frame[index])
        selected = raw["tick"].isin(required_ticks).to_numpy()
        if selected.any():
            frame = raw.loc[selected, ["tick", "player_id_hash", "team_id", "is_alive", "vel_x", "vel_y", "vel_z"]].copy()
            frame[["pos_x", "pos_y", "pos_z"]] = continuous[selected]
            kept.append(frame[["tick", "player_id_hash", "team_id", "is_alive", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"]])
    if not kept:
        return pd.DataFrame(columns=["tick", "player_id_hash", "team_id", "is_alive", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"])
    return pd.concat(kept, ignore_index=True).drop_duplicates(["tick", "player_id_hash"], keep="last").sort_values(["tick", "player_id_hash"], kind="stable")


def process(core_map: Path, output_root: Path) -> dict[str, Any]:
    source = json.loads((core_map / "map_manifest.json").read_text(encoding="utf-8"))
    selection = source["selection"]
    ordinal = int(selection["map_ordinal"])
    target = output_root / "maps" / core_map.name
    manifest_path = target / "map_manifest.json"
    if manifest_path.is_file():
        return {"map_ordinal": ordinal, "status": "skipped_completed", "manifest": base.record(manifest_path)}
    if target.exists() or target.with_name(target.name + ".partial").exists():
        raise RuntimeError("existing_incomplete_position_map")
    if shutil.disk_usage(output_root.drive + "\\").free < base.MINIMUM_FREE_BYTES:
        raise RuntimeError("output_drive_below_50_gib_floor")
    scope = pd.read_parquet(core_map / "outcome_free_duel_scope.parquet")
    events = pd.read_parquet(core_map / "event_stream.parquet")
    context = pd.read_parquet(core_map / "context.parquet")
    planned, excluded = base.planned_scope(scope, events, context)
    if planned.empty:
        partial = target.with_name(target.name + ".partial")
        partial.mkdir(parents=True)
        planned_path, position_path = partial / "planned_scope.parquet", partial / "position_remedy.parquet"
        planned.to_parquet(planned_path, index=False, compression="zstd")
        pd.DataFrame(columns=["tick", "player_id_hash", "team_id", "is_alive", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"]).to_parquet(position_path, index=False, compression="zstd")
        manifest = {
            "stage": "r1_70_sparse_positions_v2", "status": "completed_structural_zero", "created_at_utc": now(),
            "source_core_map_manifest": base.record(core_map / "map_manifest.json"), "selection": selection,
            "planned_scope": base.record(planned_path), "position_remedy": base.record(position_path),
            "hmac_salt_sha256": base.sha256(base.SALT),
            "anchor_rule": "first weapon_fire by either strict participant in [event_tick-640,event_tick], minus 32 ticks",
            "coordinate_rule": "not_run: no row passed the fixed anchor rule",
            "anchor_strictly_before_event": True, "exclusions": excluded,
            "rows": {"input_scope": int(len(scope)), "planned_scope": 0, "sparse_position": 0},
            "outcome_boundary": {"labels_read": False, "death_direction_persisted": False, "p_hat_or_residual_or_history_state_computed": False},
            "position_extractor": None, "elapsed_seconds": 0.0,
        }
        write_json(partial / "map_manifest.json", manifest)
        partial.replace(target)
        return {"map_ordinal": ordinal, "status": "completed_structural_zero", "manifest": base.record(target / "map_manifest.json"), "rows": manifest["rows"], "elapsed_seconds": 0.0}
    ticks = set(planned["anchor_tick"].astype(int)) | set((planned["anchor_tick"] - 1).astype(int))
    archive = Path(str(selection["archive"]["path"]))
    if not archive.is_file() or base.record(archive) != selection["archive"]:
        raise RuntimeError("frozen_archive_changed_or_missing")
    demo = base.extract_demo(archive, str(selection["demo_member"]), output_root / "staging")
    csv = output_root / "staging" / f"positions_{ordinal:04d}.csv"
    meta = output_root / "staging" / f"positions_{ordinal:04d}.json"
    started = time.perf_counter()
    try:
        result = base.subprocess.run([str(base.POSITION_EXE), "--demo", str(demo), "--player-id-salt-file", str(base.SALT), "--output-csv", str(csv), "--output-meta", str(meta)], stdout=base.subprocess.PIPE, stderr=base.subprocess.PIPE, check=False, timeout=1800)
        if result.returncode != 0 or not csv.is_file():
            raise RuntimeError("position_extractor_failed:" + result.stderr.decode("utf-8", errors="replace")[-500:])
        positions = sparse_continuous_positions(csv, ticks)
        if positions.empty:
            raise RuntimeError("no_sparse_positions_at_required_ticks")
        partial = target.with_name(target.name + ".partial")
        partial.mkdir(parents=True)
        planned_path, position_path = partial / "planned_scope.parquet", partial / "position_remedy.parquet"
        planned.to_parquet(planned_path, index=False, compression="zstd")
        positions.to_parquet(position_path, index=False, compression="zstd")
        manifest = {
            "stage": "r1_70_sparse_positions_v2", "status": "completed", "created_at_utc": now(),
            "source_core_map_manifest": base.record(core_map / "map_manifest.json"), "selection": selection,
            "planned_scope": base.record(planned_path), "position_remedy": base.record(position_path),
            "hmac_salt_sha256": base.sha256(base.SALT),
            "anchor_rule": "first weapon_fire by either strict participant in [event_tick-640,event_tick], minus 32 ticks",
            "coordinate_rule": "full C# tick stream cell-unwrapped before retaining anchor_tick and anchor_tick-1",
            "anchor_strictly_before_event": bool((planned["anchor_tick"] < planned["event_tick"]).all()), "exclusions": excluded,
            "rows": {"input_scope": int(len(scope)), "planned_scope": int(len(planned)), "sparse_position": int(len(positions))},
            "outcome_boundary": {"labels_read": False, "death_direction_persisted": False, "p_hat_or_residual_or_history_state_computed": False},
            "position_extractor": base.record(base.POSITION_EXE), "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        write_json(partial / "map_manifest.json", manifest)
        partial.replace(target)
        return {"map_ordinal": ordinal, "status": "completed", "manifest": base.record(target / "map_manifest.json"), "rows": manifest["rows"], "elapsed_seconds": manifest["elapsed_seconds"]}
    finally:
        demo.unlink(missing_ok=True)
        csv.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-root", type=Path, default=base.CORE)
    parser.add_argument("--output-root", type=Path, default=base.OUTPUT.with_name("sparse_positions_v2"))
    parser.add_argument("--max-maps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not (args.core_root / "manifest.json").is_file():
        raise SystemExit("core_preoutcome_manifest_required")
    base.SALT = args.core_root / "access_control" / base.SALT.name
    if not all(path.is_file() for path in (base.SEVEN_ZIP, base.POSITION_EXE, base.SALT)):
        raise SystemExit("position_runtime_input_missing")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit("nonempty_output_requires_resume")
    args.output_root.mkdir(parents=True, exist_ok=True)
    core_maps = sorted((args.core_root / "maps").iterdir())
    progress_path = args.output_root / "progress.json"
    prior = {int(row["map_ordinal"]): row for row in json.loads(progress_path.read_text(encoding="utf-8")).get("runs", [])} if progress_path.is_file() else {}
    runs: list[dict[str, Any]] = []
    processed = 0
    for core_map in core_maps:
        source = json.loads((core_map / "map_manifest.json").read_text(encoding="utf-8"))
        ordinal = int(source["selection"]["map_ordinal"])
        if prior.get(ordinal, {}).get("status") in {"completed", "completed_structural_zero", "skipped_completed"}:
            runs.append(prior[ordinal])
            continue
        if args.max_maps and processed >= args.max_maps:
            continue
        try:
            run = process(core_map, args.output_root)
        except Exception as exc:
            run = {"map_ordinal": ordinal, "status": "failed", "error": str(exc)[-1000:], "failed_at_utc": now()}
        runs.append(run)
        processed += 1
        write_json(progress_path, {"stage": "r1_70_sparse_positions_v2", "core_manifest": base.record(args.core_root / "manifest.json"), "script": base.record(Path(__file__).resolve()), "runs": runs, "updated_at_utc": now()}, replace=True)
        if run["status"] == "failed":
            raise RuntimeError("map_failed:" + run.get("error", "unknown"))
        print(f"[{processed}] {run['status']} map={ordinal}", flush=True)
    if len(runs) == len(core_maps) and all(run.get("status") in {"completed", "completed_structural_zero", "skipped_completed"} for run in runs):
        write_json(args.output_root / "manifest.json", {"stage": "r1_70_sparse_positions_v2", "status": "passed_outcome_withheld", "core_manifest": base.record(args.core_root / "manifest.json"), "script": base.record(Path(__file__).resolve()), "hmac_salt_sha256": base.sha256(base.SALT), "maps": len(core_maps), "runs": runs, "completed_at_utc": now()})


if __name__ == "__main__":
    main()
