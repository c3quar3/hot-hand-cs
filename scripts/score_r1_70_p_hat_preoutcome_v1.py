"""Score frozen D1-A p_hat on R1 70 maps without reading outcomes."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_p_hat_external_candidate_v1 as features
import score_p_hat_external_v3 as scorer


CORE = Path(r"D:\CS2_60s_confirmation_r1_70_preoutcome_v1\core_tables_v1")
POSITIONS = Path(r"D:\CS2_60s_confirmation_r1_70_preoutcome_v1\sparse_positions_v2")
OUTPUT = Path(r"D:\CS2_60s_confirmation_r1_70_preoutcome_v1\p_hat_preoutcome_v1")
MODEL = Path(r"C:\Users\Admin\Desktop\main\results\d1a_c_selection_training_v1\full_model.joblib")
MODEL_MANIFEST = Path(r"C:\Users\Admin\Desktop\main\results\d1a_c_selection_training_v1\manifest.json")


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: dict[str, Any], replace: bool = False) -> None:
    if path.exists() and not replace:
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sparse_motion(path: Path) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray, bool, bool]]:
    raw = pd.read_parquet(path).drop_duplicates(["tick", "player_id_hash"], keep="last").copy()
    raw["tick"] = pd.to_numeric(raw["tick"], errors="raise").astype("int64")
    raw["player_id_hash"] = raw["player_id_hash"].astype(str)
    for column in ("pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    keys = set(zip(raw["tick"].astype(int), raw["player_id_hash"].astype(str)))
    result: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, bool, bool]] = {}
    for row in raw.itertuples(index=False):
        tick, key = int(row.tick), str(row.player_id_hash)
        result[(tick, key)] = (
            np.array((row.pos_x, row.pos_y, row.pos_z), dtype=float),
            np.array((row.vel_x, row.vel_y, row.vel_z), dtype=float),
            (tick - 1, key) in keys,
            features.bool_value(row.is_alive),
        )
    return result


def map_candidate(core_map: Path, position_map: Path, artifact: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = json.loads((core_map / "map_manifest.json").read_text(encoding="utf-8"))
    position_manifest = json.loads((position_map / "map_manifest.json").read_text(encoding="utf-8"))
    if position_manifest.get("source_core_map_manifest", {}).get("sha256") != features.sha256(core_map / "map_manifest.json"):
        raise RuntimeError("position_core_provenance_mismatch")
    scope = pd.read_parquet(core_map / "outcome_free_duel_scope.parquet")
    planned = pd.read_parquet(position_map / "planned_scope.parquet")
    if planned.empty:
        return pd.DataFrame(), {"map_ordinal": int(source["selection"]["map_ordinal"]), "input_scope": int(len(scope)), "planned_rows": 0, "scored_rows": 0, "exclusions": position_manifest.get("exclusions", {}), "structural_zero": True}
    if planned["duel_id"].duplicated().any() or not (planned["anchor_tick"] < planned["event_tick"]).all():
        raise RuntimeError("planned_scope_key_or_time_failure")
    ticks = set(planned["anchor_tick"].astype(int)) | set(planned["post_buy_snapshot_tick"].astype(int))
    states = features.load_states([core_map], ticks)
    motion = sparse_motion(position_map / "position_remedy.parquet")
    weapon_raw = features.concat_tables([core_map], "weapon_inventory.parquet", ["tick", "player_id_hash", "Weapon.m_iItemDefinitionIndex", "Weapon.m_zoomLevel"], [("tick", "in", sorted(set(planned["anchor_tick"].astype(int))))])
    weapons = {(int(tick), str(key)): (int(item) if pd.notna(item) else 0, float(zoom) if pd.notna(zoom) else 0.0) for tick, key, item, zoom in weapon_raw.drop_duplicates(["tick", "player_id_hash"], keep="last").itertuples(index=False, name=None)}
    crosswalk = pd.read_parquet(core_map.parent.parent / "access_control" / f"map_{int(source['selection']['map_ordinal']):04d}_identity_crosswalk.parquet")
    identities = {str(key): str(int(value)) for key, value in crosswalk[["player_id_hash", "steamid"]].drop_duplicates("player_id_hash").itertuples(index=False, name=None)}
    participants = set(planned["focal_player_key"].astype(str)) | set(planned["opponent_player_key"].astype(str))
    if participants - set(identities):
        raise RuntimeError("unresolved_model_identity")
    context = pd.read_parquet(core_map / "context.parquet")
    enriched = planned.copy()
    enriched["map_name"] = str(source["header"]["map_name"]).removeprefix("de_")
    enriched["partition"] = "R1-70"
    enriched["queue_order"] = int(source["selection"]["map_ordinal"])
    rows, excluded = features.candidate_features(enriched, states, motion, weapons, identities, context)
    candidate = pd.DataFrame(rows)
    if candidate.empty:
        return candidate, {"map_ordinal": int(source["selection"]["map_ordinal"]), "input_scope": int(len(scope)), "planned_rows": int(len(planned)), "scored_rows": 0, "exclusions": dict(excluded), "structural_zero": False}
    matrix, unknown = scorer.design_matrix(candidate, artifact["design"])
    candidate["p_hat"] = artifact["model"].predict_proba(matrix)[:, 1]
    if not np.isfinite(candidate["p_hat"]).all() or not candidate["p_hat"].between(0, 1, inclusive="neither").all():
        raise RuntimeError("invalid_probability")
    keep = ["duel_id", "match_id", "map_id", "round", "event_tick", "pre_event_tick", "first_fire_tick", "anchor_tick", "post_buy_snapshot_tick", "focal_player_key", "opponent_player_key", "atomic_tick_batch_id", "atomic_tick_batch_size", "atomic_tick_participant_count", "atomic_tick_is_multidual", "atomic_tick_shared_participant", "atomic_tick_history_policy"]
    scored = planned[keep].merge(candidate[["duel_id", "p_hat"]], on="duel_id", how="inner", validate="one_to_one")
    return scored, {"map_ordinal": int(source["selection"]["map_ordinal"]), "input_scope": int(len(scope)), "planned_rows": int(len(planned)), "scored_rows": int(len(scored)), "exclusions": dict(excluded), "unknown_categories": unknown, "structural_zero": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-root", type=Path, default=CORE)
    parser.add_argument("--positions-root", type=Path, default=POSITIONS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--max-maps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not all(path.is_file() for path in (args.core_root / "manifest.json", args.positions_root / "manifest.json", MODEL, MODEL_MANIFEST)):
        raise SystemExit("required_outcome_free_input_missing")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit("nonempty_output_requires_resume")
    artifact = scorer.verify_model(MODEL, MODEL_MANIFEST)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.json"
    prior = {int(row["map_ordinal"]): row for row in json.loads(progress_path.read_text(encoding="utf-8")).get("runs", [])} if progress_path.is_file() else {}
    runs: list[dict[str, Any]] = []
    processed = 0
    for core_map in sorted((args.core_root / "maps").iterdir()):
        source = json.loads((core_map / "map_manifest.json").read_text(encoding="utf-8"))
        ordinal = int(source["selection"]["map_ordinal"])
        if prior.get(ordinal, {}).get("status") in {"completed", "skipped_completed"}:
            runs.append(prior[ordinal]); continue
        if args.max_maps and processed >= args.max_maps:
            continue
        target = args.output_root / "maps" / core_map.name
        try:
            if target.exists():
                raise RuntimeError("existing_incomplete_map")
            frame, audit = map_candidate(core_map, args.positions_root / "maps" / core_map.name, artifact)
            partial = target.with_name(target.name + ".partial")
            partial.mkdir(parents=True)
            frame.to_parquet(partial / "p_hat_preoutcome.parquet", index=False, compression="zstd")
            audit["stage"] = "r1_70_p_hat_preoutcome_v1"; audit["outcome_boundary"] = {"labels_read": False, "outcomes_persisted": False, "residual_or_history_state_computed": False}
            audit["p_hat_preoutcome"] = {"path": str((partial / "p_hat_preoutcome.parquet").resolve()), "sha256": features.sha256(partial / "p_hat_preoutcome.parquet")}
            write_json(partial / "map_manifest.json", audit)
            partial.replace(target)
            run = {"map_ordinal": ordinal, "status": "completed", "manifest": {"path": str((target / "map_manifest.json").resolve()), "sha256": features.sha256(target / "map_manifest.json")}, "rows": int(len(frame))}
        except Exception as exc:
            run = {"map_ordinal": ordinal, "status": "failed", "error": str(exc)[-1000:], "failed_at_utc": now()}
        runs.append(run); processed += 1
        write_json(progress_path, {"stage": "r1_70_p_hat_preoutcome_v1", "core_manifest": {"sha256": features.sha256(args.core_root / "manifest.json")}, "positions_manifest": {"sha256": features.sha256(args.positions_root / "manifest.json")}, "model": {"sha256": features.sha256(MODEL), "manifest_sha256": features.sha256(MODEL_MANIFEST)}, "script_sha256": features.sha256(Path(__file__).resolve()), "runs": runs, "updated_at_utc": now()}, replace=True)
        if run["status"] == "failed":
            raise RuntimeError("map_failed:" + run.get("error", "unknown"))
        print(f"[{processed}] completed map={ordinal} rows={run['rows']}", flush=True)
    if len(runs) == len(list((args.core_root / "maps").iterdir())) and all(run.get("status") in {"completed", "skipped_completed"} for run in runs):
        tables = [pd.read_parquet(path / "p_hat_preoutcome.parquet") for path in sorted((args.output_root / "maps").iterdir())]
        final = pd.concat(tables, ignore_index=True).sort_values(["match_id", "map_id", "event_tick", "duel_id"], kind="stable")
        if final["duel_id"].duplicated().any():
            raise RuntimeError("duplicate_scored_duel_id")
        final.to_parquet(args.output_root / "p_hat_preoutcome.parquet", index=False, compression="zstd")
        write_json(args.output_root / "manifest.json", {"stage": "r1_70_p_hat_preoutcome_v1", "status": "passed_outcome_withheld", "core_manifest_sha256": features.sha256(args.core_root / "manifest.json"), "positions_manifest_sha256": features.sha256(args.positions_root / "manifest.json"), "model_sha256": features.sha256(MODEL), "model_manifest_sha256": features.sha256(MODEL_MANIFEST), "script_sha256": features.sha256(Path(__file__).resolve()), "rows": int(len(final)), "maps": int(final["map_id"].nunique()), "matches": int(final["match_id"].nunique()), "p_hat_preoutcome_sha256": features.sha256(args.output_root / "p_hat_preoutcome.parquet"), "outcome_boundary": {"labels_read": False, "outcomes_persisted": False, "residual_or_history_state_computed": False}, "runs": runs, "completed_at_utc": now()})


if __name__ == "__main__":
    main()
