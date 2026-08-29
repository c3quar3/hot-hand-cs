"""Rebuild an outcome-free external p_hat candidate from locked rows and raw inputs.

This is a versioned adaptation of the historical v7/v8 feature builders.  It
never opens outcome_withheld.parquet and never copies a historical candidate's
features or anchor.  The locked scope supplies only the row universe and
participant keys; all candidate features are rebuilt from non-outcome files.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


MAIN = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"D:\CS2_p_hat_v3_d1_extracted_v1")
SCOPE_ROOT = MAIN / "data" / "confirmation_scopes_v1"
OUTPUT_ROOT = MAIN / "results" / "external_candidates_v1"
TICK_RATE = 64.0
LOOKBACK_TICKS = 640
ANCHOR_LAG_TICKS = 32
CELL_WIDTH = 1024.0
MAX_FALLBACK_SPEED = 1000.0
RAW_SOURCE = "raw_vel_continuous_position_v1"
FALLBACK_SOURCE = "backward_position_continuous_position_v1"

WEAPON_FAMILY = {
    1: "DEAGLE", 2: "ELITE", 3: "FIVESEVEN", 4: "GLOCK", 7: "AK47", 8: "AUG", 9: "AWP", 10: "FAMAS",
    11: "G3SG1", 13: "GALIL", 14: "M249", 16: "M4A4", 17: "MAC10", 19: "P90", 23: "MP5SD", 24: "UMP45",
    25: "XM1014", 26: "BIZON", 27: "MAG7", 28: "NEGEV", 29: "SAWEDOFF", 30: "TEC9", 31: "ZEUS", 32: "P2000",
    33: "MP7", 34: "MP9", 35: "NOVA", 36: "P250", 38: "SCAR20", 39: "SG553", 40: "SSG08", 60: "M4A1S",
    61: "USPS", 63: "CZ75", 64: "REVOLVER",
}
STATE_COLUMNS = [
    "tick", "player_id_hash", "CCSPlayerPawn.m_iHealth", "CCSPlayerPawn.m_ArmorValue", "CCSPlayerPawn.m_iTeamNum",
    "CCSPlayerController.m_bPawnIsAlive", "CCSPlayerController.CCSPlayerController_InGameMoneyServices.m_iAccount",
    "CCSPlayerPawn.m_unFreezetimeEndEquipmentValue", "CCSPlayerPawn.CCSPlayer_ItemServices.m_bHasHelmet",
    "CCSPlayerPawn.CCSPlayer_ItemServices.m_bHasDefuser", "CCSPlayerPawn.m_bIsDefusing", "CCSPlayerPawn.m_flFlashDuration",
    "CCSPlayerPawn.m_angEyeAngles",
]
MOTION_COLUMNS = ["tick", "player_id_hash", "is_alive", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"]
NUMERIC = [
    "focal_minus_opponent_health", "focal_minus_opponent_team_alive", "focal_speed", "opponent_speed", "position_distance",
    "focal_has_armor", "opponent_has_armor", "focal_has_helmet", "opponent_has_helmet", "focal_has_defuser", "opponent_has_defuser",
    "focal_minus_opponent_post_buy_money", "focal_minus_opponent_post_buy_equipment_value", "focal_minus_opponent_team_post_buy_equipment_value",
    "absolute_vertical_position_difference", "relative_radial_speed", "relative_transverse_speed", "relative_angular_speed",
    "focal_facing_error_radians", "opponent_facing_error_radians", "focal_nearest_living_teammate_distance",
    "opponent_nearest_living_teammate_distance", "focal_nearest_living_enemy_distance", "opponent_nearest_living_enemy_distance",
    "focal_is_scoped", "opponent_is_scoped", "focal_is_alone", "opponent_is_alone", "focal_is_flashed", "opponent_is_flashed",
    "focal_flash_remaining_seconds", "opponent_flash_remaining_seconds", "bomb_planted", "bomb_time_remaining_seconds",
    "focal_is_defusing", "opponent_is_defusing",
]
CATEGORICAL = ["focal_side", "map_name", "focal_speed_source", "opponent_speed_source", "focal_weapon", "opponent_weapon", "weapon_pair", "focal_steamid", "opponent_steamid"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def player_key(salt: bytes, value: object) -> str:
    try:
        steamid = int(value)
    except (TypeError, ValueError):
        return ""
    return hmac.new(salt, str(steamid).encode("ascii"), hashlib.sha256).hexdigest() if steamid > 0 else ""


def bool_value(value: object) -> bool:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return bool(numeric) if pd.notna(numeric) else False


def map_directories(match_id: str, map_name: str) -> list[Path]:
    result = []
    for directory in sorted((DATA_ROOT / "maps" / match_id).iterdir()):
        manifest = directory / "map_manifest.json"
        if not manifest.is_file():
            continue
        name = str(json.loads(manifest.read_text(encoding="utf-8")).get("header", {}).get("map_name", "")).removeprefix("de_")
        if name == map_name:
            result.append(directory)
    if not result:
        raise RuntimeError(f"map_directory_not_found:{match_id}:{map_name}")
    return result


def concat_tables(paths: list[Path], name: str, columns: list[str], filters: list[tuple[str, str, list[int]]] | None = None) -> pd.DataFrame:
    frames = [pq.read_table(path / name, columns=columns, filters=filters).to_pandas() for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def map_events(paths: list[Path], salt: bytes) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    event = concat_tables(paths, "event_stream.parquet", ["tick", "event_type", "user_steamid"])
    event["tick"] = pd.to_numeric(event["tick"], errors="coerce")
    event = event.dropna(subset=["tick", "user_steamid"]).copy()
    event["key"] = event["user_steamid"].map(lambda value: player_key(salt, value))
    event = event.loc[event["key"].ne("")]
    identity = {key: str(int(value)) for key, value in event[["key", "user_steamid"]].drop_duplicates("key").itertuples(index=False, name=None)}
    fire = event.loc[event["event_type"].eq("weapon_fire"), ["tick", "key"]].drop_duplicates().sort_values("tick")
    return {key: group["tick"].to_numpy(dtype=np.int64) for key, group in fire.groupby("key", sort=False)}, identity


def planned_rows(scope: pd.DataFrame, fires: dict[str, np.ndarray], context: pd.DataFrame) -> tuple[pd.DataFrame, Counter[str]]:
    freeze = np.sort(pd.to_numeric(context.loc[context["event_type"].eq("round_freeze_end"), "tick"], errors="coerce").dropna().astype("int64").to_numpy())
    kept, excluded = [], Counter()
    for row in scope.itertuples(index=False):
        event_tick = int(row.event_tick)
        first_candidates = []
        for key in (str(row.focal_player_key), str(row.opponent_player_key)):
            values = fires.get(key, np.array([], dtype=np.int64))
            pos = np.searchsorted(values, event_tick - LOOKBACK_TICKS, side="left")
            if pos < len(values) and values[pos] <= event_tick:
                first_candidates.append(int(values[pos]))
        if not first_candidates:
            excluded["no_pair_weapon_fire_in_10s_window"] += 1
            continue
        first_fire = min(first_candidates)
        anchor = first_fire - ANCHOR_LAG_TICKS
        snapshot_pos = np.searchsorted(freeze, anchor, side="right") - 1
        if snapshot_pos < 0:
            excluded["no_round_freeze_end_before_anchor"] += 1
            continue
        kept.append({**row._asdict(), "first_fire_tick": first_fire, "anchor_tick": anchor, "post_buy_snapshot_tick": int(freeze[snapshot_pos])})
    return pd.DataFrame(kept), excluded


def load_states(paths: list[Path], ticks: set[int]) -> dict[tuple[int, str], dict[str, Any]]:
    raw = concat_tables(paths, "full_tick_player_state.parquet", STATE_COLUMNS, [("tick", "in", sorted(ticks))])
    raw = raw.drop_duplicates(["tick", "player_id_hash"], keep="last")
    rename = {
        "CCSPlayerPawn.m_iHealth": "health", "CCSPlayerPawn.m_ArmorValue": "armor", "CCSPlayerPawn.m_iTeamNum": "team",
        "CCSPlayerController.m_bPawnIsAlive": "alive", "CCSPlayerController.CCSPlayerController_InGameMoneyServices.m_iAccount": "money",
        "CCSPlayerPawn.m_unFreezetimeEndEquipmentValue": "post_buy_equipment", "CCSPlayerPawn.CCSPlayer_ItemServices.m_bHasHelmet": "helmet",
        "CCSPlayerPawn.CCSPlayer_ItemServices.m_bHasDefuser": "defuser", "CCSPlayerPawn.m_bIsDefusing": "defusing",
        "CCSPlayerPawn.m_flFlashDuration": "flash", "CCSPlayerPawn.m_angEyeAngles": "angles",
    }
    raw = raw.rename(columns=rename)
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for row in raw.itertuples(index=False):
        value = row._asdict()
        value["tick"] = int(value["tick"]); value["player_id_hash"] = str(value["player_id_hash"])
        for column in ("health", "armor", "team", "money", "post_buy_equipment", "flash"):
            value[column] = float(pd.to_numeric(pd.Series([value[column]]), errors="coerce").iloc[0])
        for column in ("alive", "helmet", "defuser", "defusing"):
            value[column] = bool_value(value[column])
        result[(value["tick"], value["player_id_hash"])] = value
    return result


def continuous_motion(paths: list[Path], anchors: set[int]) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray, bool, bool]]:
    # position_remedy is tick-sorted.  Stream it to avoid retaining a full map's
    # multi-million-row position table while preserving the historical v8 cell
    # unwrapping rule within each player trajectory.
    needed_ticks = anchors | {tick - 1 for tick in anchors}
    result: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, bool, bool]] = {}
    for directory in paths:
        previous: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
        records: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, bool, bool]] = {}
        parquet = pq.ParquetFile(directory / "position_remedy.parquet")
        for batch in parquet.iter_batches(batch_size=200_000, columns=MOTION_COLUMNS):
            raw = batch.to_pandas().dropna(subset=["tick", "player_id_hash"]).copy()
            raw["tick"] = pd.to_numeric(raw["tick"], errors="coerce").astype("int64")
            raw["player_id_hash"] = raw["player_id_hash"].astype(str)
            for column in ("pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"):
                raw[column] = pd.to_numeric(raw[column], errors="coerce")
            keys = raw["player_id_hash"]
            prior_tick = raw.groupby("player_id_hash", sort=False)["tick"].shift(1)
            prior_position = raw.groupby("player_id_hash", sort=False)[["pos_x", "pos_y", "pos_z"]].shift(1)
            first = prior_tick.isna()
            prior_tick.loc[first] = keys.loc[first].map(lambda key: previous.get(key, (float("nan"), None, None))[0])
            for column, offset in zip(("pos_x", "pos_y", "pos_z"), range(3)):
                prior_position.loc[first, column] = keys.loc[first].map(lambda key: previous.get(key, (None, np.full(3, np.nan), None))[1][offset])
            coordinates = raw[["pos_x", "pos_y", "pos_z"]].to_numpy(float)
            prior = prior_position.to_numpy(float)
            adjacent = raw["tick"].sub(prior_tick).eq(1).to_numpy()
            adjustment = CELL_WIDTH * np.rint((coordinates - prior) / CELL_WIDTH)
            adjustment[~adjacent | ~np.isfinite(adjustment).all(axis=1)] = 0.0
            segment = pd.Series((~adjacent).astype("int64")).groupby(keys, sort=False).cumsum()
            groups = pd.MultiIndex.from_arrays([keys, segment])
            cumulative = pd.DataFrame(adjustment).groupby(groups, sort=False).cumsum().to_numpy(float)
            seed = np.zeros_like(cumulative)
            for index in np.flatnonzero(first.to_numpy() & adjacent):
                seed[index] = previous[str(keys.iat[index])][2]
            seed_frame = pd.DataFrame(seed).groupby(groups, sort=False).transform("first").to_numpy(float)
            continuous = coordinates - cumulative - seed_frame
            # Keep state needed by the next batch for every observed player.
            for key, index in raw.groupby("player_id_hash", sort=False).tail(1).index.to_series().items():
                previous[str(raw.at[index, "player_id_hash"])] = (int(raw.at[index, "tick"]), coordinates[index], cumulative[index] + seed_frame[index])
            selected = raw["tick"].isin(needed_ticks).to_numpy()
            for index in np.flatnonzero(selected):
                records[(int(raw.at[index, "tick"]), str(raw.at[index, "player_id_hash"]))] = (
                    continuous[index], raw.loc[index, ["vel_x", "vel_y", "vel_z"]].to_numpy(float), bool(adjacent[index]), bool_value(raw.at[index, "is_alive"]),
                )
        # Later split parts supersede overlapping rows, matching the old builder.
        result.update(records)
    return result


def yaw(value: object) -> float:
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 1:
        try:
            return float(value[1])
        except (TypeError, ValueError):
            pass
    return float("nan")


def facing(source: np.ndarray, target: np.ndarray, angles: object) -> float:
    current = yaw(angles)
    if not (np.isfinite(source).all() and np.isfinite(target).all() and math.isfinite(current)):
        return float("nan")
    diff = abs(math.atan2(target[1] - source[1], target[0] - source[0]) - math.radians(current))
    return min(diff, 2 * math.pi - diff)


def candidate_features(rows: pd.DataFrame, states: dict[tuple[int, str], dict[str, Any]], motion: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, bool, bool]], weapons: dict[tuple[int, str], tuple[int, float]], identities: dict[str, str], context: pd.DataFrame) -> tuple[list[dict[str, Any]], Counter[str]]:
    context["tick"] = pd.to_numeric(context["tick"], errors="coerce")
    plants = context.loc[context["event_type"].eq("bomb_planted"), "tick"].dropna().astype("int64").to_numpy()
    round_starts = np.sort(context.loc[context["event_type"].eq("round_start"), "tick"].dropna().astype("int64").to_numpy())
    result, excluded = [], Counter()
    for row in rows.itertuples(index=False):
        anchor, focal, opponent = int(row.anchor_tick), str(row.focal_player_key), str(row.opponent_player_key)
        fstate, ostate = states.get((anchor, focal)), states.get((anchor, opponent))
        fnow, onow, fprior, oprior = motion.get((anchor, focal)), motion.get((anchor, opponent)), motion.get((anchor - 1, focal)), motion.get((anchor - 1, opponent))
        if not all((fstate, ostate, fnow, onow, fprior, oprior)):
            excluded["missing_anchor_state_or_position"] += 1; continue
        if not (fstate["alive"] and ostate["alive"] and fnow[2] and onow[2] and fnow[3] and onow[3] and fprior[3] and oprior[3]):
            excluded["participants_not_continuously_alive_at_anchor"] += 1; continue
        team_f, team_o = int(fstate["team"]), int(ostate["team"])
        if team_f not in (2, 3) or team_o not in (2, 3) or team_f == team_o:
            excluded["invalid_anchor_teams"] += 1; continue
        live = []
        unreliable = False
        for (tick, key), state in states.items():
            if tick != anchor or not state["alive"] or int(state["team"]) not in (2, 3):
                continue
            current, prior = motion.get((anchor, key)), motion.get((anchor - 1, key))
            if current is None or prior is None or not current[2] or not current[3] or not prior[3] or not np.isfinite(current[0]).all():
                unreliable = True; break
            live.append((key, int(state["team"]), current[0], state["angles"]))
        if unreliable:
            excluded["unreliable_living_geometry"] += 1; continue
        def vector(now: tuple[np.ndarray, np.ndarray, bool, bool], prior: tuple[np.ndarray, np.ndarray, bool, bool]) -> tuple[np.ndarray, str] | None:
            raw = now[1]
            if np.isfinite(raw).all():
                return raw, RAW_SOURCE
            fallback = (now[0] - prior[0]) * TICK_RATE
            return (fallback, FALLBACK_SOURCE) if np.isfinite(fallback).all() and np.linalg.norm(fallback) <= MAX_FALLBACK_SPEED else None
        fv, ov = vector(fnow, fprior), vector(onow, oprior)
        if fv is None or ov is None:
            excluded["invalid_velocity"] += 1; continue
        fpos, opos = fnow[0], onow[0]
        direction = opos - fpos; distance = float(np.linalg.norm(direction))
        if not np.isfinite(direction).all() or distance <= 1e-6:
            excluded["invalid_position_distance"] += 1; continue
        relative = fv[0] - ov[0]; radial = float(np.dot(relative, direction) / distance)
        def nearest(owner: str, team: int, pos: np.ndarray) -> tuple[float, float, int]:
            mates, enemies = [], []
            for key, other_team, other_pos, _ in live:
                if key != owner:
                    (mates if other_team == team else enemies).append(float(np.linalg.norm(pos - other_pos)))
            return (0.0 if not mates else min(mates), float("nan") if not enemies else min(enemies), int(not mates))
        fm, fe, fa = nearest(focal, team_f, fpos); om, oe, oa = nearest(opponent, team_o, opos)
        snapshot = int(row.post_buy_snapshot_tick)
        fsnap, osnap = states.get((snapshot, focal)), states.get((snapshot, opponent))
        if fsnap is None or osnap is None:
            excluded["missing_post_buy_snapshot_state"] += 1; continue
        snapshot_states = [state for (tick, _), state in states.items() if tick == snapshot and int(state["team"]) in (2, 3)]
        team_equip = {team: sum(state["post_buy_equipment"] for state in snapshot_states if int(state["team"]) == team) for team in (2, 3)}
        fw = weapons.get((anchor, focal), (0, 0.0)); ow = weapons.get((anchor, opponent), (0, 0.0))
        fweapon, oweapon = WEAPON_FAMILY.get(fw[0], "OTHER"), WEAPON_FAMILY.get(ow[0], "OTHER")
        last_round = round_starts[np.searchsorted(round_starts, anchor, side="right") - 1] if len(round_starts) and round_starts[0] <= anchor else -1
        planted = plants[(plants > last_round) & (plants <= anchor)]
        plant_tick = int(planted[-1]) if len(planted) else None
        result.append({
            "duel_id": row.duel_id, "match_id": row.match_id, "map_id": row.map_id, "partition": row.partition, "queue_order": int(row.queue_order),
            "event_tick": int(row.event_tick), "first_fire_tick": int(row.first_fire_tick), "anchor_tick": anchor, "post_buy_snapshot_tick": snapshot,
            "focal_player_key": focal, "opponent_player_key": opponent, "focal_steamid": identities.get(focal, "unresolved:" + focal), "opponent_steamid": identities.get(opponent, "unresolved:" + opponent),
            "focal_minus_opponent_health": fstate["health"] - ostate["health"], "focal_minus_opponent_team_alive": sum(s["alive"] for _, t, _, s in [(k,t,p,states[(anchor,k)]) for k,t,p,_ in live] if int(s["team"]) == team_f) - sum(s["alive"] for _, t, _, s in [(k,t,p,states[(anchor,k)]) for k,t,p,_ in live] if int(s["team"]) == team_o),
            "focal_speed": float(np.linalg.norm(fv[0])), "opponent_speed": float(np.linalg.norm(ov[0])), "position_distance": distance,
            "focal_has_armor": int(fstate["armor"] > 0), "opponent_has_armor": int(ostate["armor"] > 0), "focal_has_helmet": int(fstate["helmet"]), "opponent_has_helmet": int(ostate["helmet"]), "focal_has_defuser": int(fstate["defuser"]), "opponent_has_defuser": int(ostate["defuser"]),
            "focal_minus_opponent_post_buy_money": fsnap["money"] - osnap["money"], "focal_minus_opponent_post_buy_equipment_value": fsnap["post_buy_equipment"] - osnap["post_buy_equipment"], "focal_minus_opponent_team_post_buy_equipment_value": team_equip[team_f] - team_equip[team_o],
            "absolute_vertical_position_difference": abs(float(direction[2])), "relative_radial_speed": radial, "relative_transverse_speed": float(math.sqrt(max(0.0, float(np.dot(relative, relative)) - radial * radial))), "relative_angular_speed": float(np.linalg.norm(np.cross(direction, relative)) / (distance * distance)),
            "focal_facing_error_radians": facing(fpos, opos, fstate["angles"]), "opponent_facing_error_radians": facing(opos, fpos, ostate["angles"]), "focal_nearest_living_teammate_distance": fm, "opponent_nearest_living_teammate_distance": om, "focal_nearest_living_enemy_distance": fe, "opponent_nearest_living_enemy_distance": oe, "focal_is_scoped": int(fw[1] > 0), "opponent_is_scoped": int(ow[1] > 0), "focal_is_alone": fa, "opponent_is_alone": oa,
            "focal_is_flashed": int(fstate["flash"] > 0), "opponent_is_flashed": int(ostate["flash"] > 0), "focal_flash_remaining_seconds": max(0.0, fstate["flash"]), "opponent_flash_remaining_seconds": max(0.0, ostate["flash"]), "bomb_planted": int(plant_tick is not None), "bomb_time_remaining_seconds": 0.0 if plant_tick is None else max(0.0, 40.0 - (anchor - plant_tick) / TICK_RATE), "focal_is_defusing": int(fstate["defusing"]), "opponent_is_defusing": int(ostate["defusing"]),
            "focal_side": "T" if team_f == 2 else "CT", "map_name": row.map_name, "focal_speed_source": fv[1], "opponent_speed_source": ov[1], "focal_weapon": fweapon, "opponent_weapon": oweapon, "weapon_pair": "_".join(sorted((fweapon, oweapon))),
        })
    return result, excluded


def build(dataset: str, output_root: Path, max_maps: int) -> None:
    if dataset != "D1-B":
        raise RuntimeError("v1_is_authorized_for_D1-B_only")
    scope_path = SCOPE_ROOT / dataset / "candidate_scope_outcome_free.parquet"
    output = output_root / dataset
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    scope = pd.read_parquet(scope_path)
    if scope["duel_id"].duplicated().any():
        raise RuntimeError("scope_duplicate_duel_id")
    salt = (DATA_ROOT / "access_control" / "d1_player_hmac_salt.bin").read_bytes()
    all_rows, exclusions, map_audits = [], Counter(), []
    groups = list(scope.groupby(["match_id", "map_name"], sort=True))
    for index, ((match_id, map_name), map_scope) in enumerate(groups, start=1):
        if max_maps and index > max_maps:
            break
        paths = map_directories(str(match_id), str(map_name))
        context = concat_tables(paths, "context.parquet", ["tick", "event_type"])
        fires, identities = map_events(paths, salt)
        planned, removed = planned_rows(map_scope, fires, context); exclusions.update(removed)
        if planned.empty:
            map_audits.append({"map_id": str(map_scope.iloc[0].map_id), "input_rows": len(map_scope), "output_rows": 0, "exclusions": dict(removed)}); continue
        ticks = set(planned["anchor_tick"].astype(int)) | set(planned["post_buy_snapshot_tick"].astype(int))
        states = load_states(paths, ticks)
        motion = continuous_motion(paths, set(planned["anchor_tick"].astype(int)))
        weapon_raw = concat_tables(paths, "weapon_inventory.parquet", ["tick", "player_id_hash", "Weapon.m_iItemDefinitionIndex", "Weapon.m_zoomLevel"], [("tick", "in", sorted(set(planned["anchor_tick"].astype(int))))])
        weapons = {(int(tick), str(key)): (int(item) if pd.notna(item) else 0, float(zoom) if pd.notna(zoom) else 0.0) for tick, key, item, zoom in weapon_raw.drop_duplicates(["tick", "player_id_hash"], keep="last").itertuples(index=False, name=None)}
        rebuilt, removed = candidate_features(planned, states, motion, weapons, identities, context); exclusions.update(removed); all_rows.extend(rebuilt)
        map_audits.append({"map_id": str(map_scope.iloc[0].map_id), "input_rows": int(len(map_scope)), "output_rows": int(len(rebuilt)), "exclusions": dict(removed)})
        print(f"maps={index}/{min(len(groups), max_maps or len(groups))} rows={len(all_rows)}", flush=True)
    partial = bool(max_maps and max_maps < len(groups))
    candidate = pd.DataFrame(all_rows)
    if candidate.empty or candidate["duel_id"].duplicated().any():
        raise RuntimeError("invalid_rebuilt_candidate")
    required = set(NUMERIC) | set(CATEGORICAL) | {"duel_id", "match_id", "map_id", "event_tick", "anchor_tick"}
    if required - set(candidate):
        raise RuntimeError("candidate_contract_missing")
    if not np.isfinite(candidate[NUMERIC].to_numpy(float)).all() or not candidate["anchor_tick"].lt(candidate["event_tick"]).all():
        raise RuntimeError("candidate_feature_or_timing_gate_failed")
    output.mkdir(parents=True)
    candidate.to_parquet(output / "candidate.parquet", index=False, compression="zstd")
    manifest = {"stage": "p_hat_external_raw_candidate_v1", "created_at_utc": datetime.now(UTC).isoformat(), "status": "partial_smoke" if partial else "passed", "outcome_access": "not_read", "dataset": dataset, "source_scope_sha256": sha256(scope_path), "candidate_sha256": sha256(output / "candidate.parquet"), "builder_sha256": sha256(Path(__file__).resolve()), "anchor_rule": "first weapon_fire by either locked participant in [event_tick-640,event_tick], minus 32 ticks", "temporal_audit": {"status": "passed", "anchor_strictly_before_event": True}, "input_rows": int(len(scope)), "processed_maps": int(len(map_audits)), "output_rows": int(len(candidate)), "excluded_rows": int(sum(exclusions.values())), "exclusions": dict(exclusions), "map_audits": map_audits, "feature_contract": {"numeric": NUMERIC, "categorical": CATEGORICAL}}
    write_new_json(output / "candidate_manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "dataset", "processed_maps", "output_rows", "excluded_rows")}, ensure_ascii=True, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("D1-B",))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-maps", type=int, default=0)
    args = parser.parse_args()
    build(args.dataset, args.output_root, args.max_maps)


if __name__ == "__main__":
    main()
