"""Create reproducible descriptive player, factor, and map side checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "results" / "conditional_residual_asset_d1_v2"
SCORED = ROOT / "results" / "external_generalization_v3" / "D1-B" / "scored.parquet"
MODEL = ROOT / "results" / "d1a_c_selection_training_v1" / "full_model.joblib"
RAW_MAP_ROOT = Path(r"D:\CS2_p_hat_v3_d1_extracted_v1\maps")
MIN_PLAYER_PARTICIPATIONS = 500

FACTOR_GROUPS = {
    "health_advantage": ["focal_minus_opponent_health"],
    "personal_equipment_value_advantage": ["focal_minus_opponent_post_buy_equipment_value"],
    "alive_teammate_advantage": ["focal_minus_opponent_team_alive"],
    "armor_state": ["focal_has_armor", "opponent_has_armor"],
    "weapon_loadout": ["focal_weapon", "opponent_weapon", "weapon_pair"],
    "scope_state": ["focal_is_scoped", "opponent_is_scoped"],
    "movement_speed": ["focal_speed", "opponent_speed"],
    "team_equipment_value_advantage": ["focal_minus_opponent_team_post_buy_equipment_value"],
    "flash_exposure": [
        "focal_flash_remaining_seconds",
        "opponent_flash_remaining_seconds",
        "focal_is_flashed",
        "opponent_is_flashed",
    ],
    "isolation": ["focal_is_alone", "opponent_is_alone"],
    "nearest_enemy_distance": [
        "focal_nearest_living_enemy_distance",
        "opponent_nearest_living_enemy_distance",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def input_paths() -> dict[str, Path]:
    return {
        "d1a_residual_asset": ASSET_ROOT / "D1-A" / "residual_asset.parquet",
        "d1b_residual_asset": ASSET_ROOT / "D1-B" / "residual_asset.parquet",
        "d1b_scored": SCORED,
        "frozen_model": MODEL,
        "script": Path(__file__).resolve(),
    }


def lock_inputs(output: Path) -> dict[str, Any]:
    paths = input_paths()
    if not RAW_MAP_ROOT.is_dir():
        raise RuntimeError(f"raw_map_root_missing:{RAW_MAP_ROOT}")
    if not all(path.is_file() for path in paths.values()):
        raise RuntimeError("analysis_input_missing")
    fingerprints = {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()}
    lock = {
        "stage": "exploratory_duel_side_checks_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "classification": "post_hoc_descriptive_exploratory_only",
        "inputs": fingerprints,
        "raw_alias_source": str(RAW_MAP_ROOT),
        "player_metric": "For every physical duel, focal receives Y-p_hat and opponent receives -(Y-p_hat); player value is the participant-record mean.",
        "player_partitions": ["D1-A forward OOF", "D1-B fixed external score"],
        "minimum_player_participations": MIN_PLAYER_PARTICIPATIONS,
        "factor_metric": "Standard deviation across previously released D1-B rows of each non-overlapping feature group's frozen-model logit contribution.",
        "map_metric": "D1-B participant-expanded observed duel win rate by map and side.",
        "forbidden_interpretations": [
            "player skill rating",
            "causal factor ranking",
            "round win probability",
            "confirmation evidence for the hot-hand hypothesis",
        ],
    }
    path = output / "checkpoint_01_input_lock.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["inputs"] != fingerprints:
            raise RuntimeError("input_hash_changed_after_checkpoint")
        return saved
    write_json_new(path, lock)
    return lock


def load_player_participants() -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    columns = ["focal_player_id", "opponent_player_id", "r_t", "primary_analysis_eligible"]
    for partition in ("D1-A", "D1-B"):
        frame = pd.read_parquet(ASSET_ROOT / partition / "residual_asset.parquet", columns=columns)
        frame = frame.loc[frame["primary_analysis_eligible"]].copy()
        focal = frame[["focal_player_id", "r_t"]].rename(
            columns={"focal_player_id": "player_id", "r_t": "adjusted_value"}
        )
        opponent = frame[["opponent_player_id", "r_t"]].rename(
            columns={"opponent_player_id": "player_id", "r_t": "adjusted_value"}
        )
        focal["player_id"] = focal["player_id"].astype(str)
        opponent["player_id"] = opponent["player_id"].astype(str)
        opponent["adjusted_value"] *= -1
        records.extend([focal, opponent])
    return pd.concat(records, ignore_index=True)


def scan_player_names(player_ids: set[str]) -> pd.DataFrame:
    names: dict[str, Counter[str]] = defaultdict(Counter)
    source_files = 0
    for path in RAW_MAP_ROOT.glob("*/*/event_stream.parquet"):
        try:
            table = pq.read_table(path, columns=["user_steamid", "user_name"]).to_pandas()
        except (KeyError, OSError):
            continue
        table = table.dropna(subset=["user_steamid", "user_name"])
        for steam_id, name in zip(table["user_steamid"], table["user_name"], strict=False):
            player_id = str(int(steam_id))
            if player_id in player_ids and str(name).strip():
                names[player_id][str(name).strip()] += 1
                source_files += 1
    rows = []
    for player_id in sorted(player_ids):
        ranked = names[player_id].most_common()
        rows.append(
            {
                "player_id": player_id,
                "player_name": ranked[0][0] if ranked else "unresolved",
                "name_event_rows": ranked[0][1] if ranked else 0,
                "observed_names_json": json.dumps(ranked, ensure_ascii=True),
            }
        )
    if any(row["player_name"] == "unresolved" for row in rows):
        raise RuntimeError("top_player_name_unresolved")
    return pd.DataFrame(rows)


def player_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    participant = load_player_participants()
    summary = (
        participant.groupby("player_id", sort=True)["adjusted_value"]
        .agg(participations="size", adjusted_value="mean", residual_sd="std")
        .reset_index()
    )
    summary["naive_se"] = summary["residual_sd"] / np.sqrt(summary["participations"])
    summary["naive_ci_low"] = summary["adjusted_value"] - 1.96 * summary["naive_se"]
    summary["naive_ci_high"] = summary["adjusted_value"] + 1.96 * summary["naive_se"]
    eligible = summary.loc[summary["participations"] >= MIN_PLAYER_PARTICIPATIONS].copy()
    top = eligible.sort_values(["adjusted_value", "participations", "player_id"], ascending=[False, False, True]).head(20)
    aliases = scan_player_names(set(top["player_id"]))
    top = top.merge(aliases, on="player_id", validate="one_to_one")
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    return top, aliases


def categorical_contribution(
    frame: pd.DataFrame, design: dict[str, Any], coefficients: np.ndarray
) -> dict[str, np.ndarray]:
    offset = len(design["numeric"])
    values: dict[str, np.ndarray] = {}
    for column in design["categorical"]:
        categories = [str(value) for value in design["categories"][column]]
        reference = str(design["references"][column])
        lookup = {value: index for index, value in enumerate(value for value in categories if value != reference)}
        contribution = np.zeros(len(frame), dtype=float)
        for row, value in enumerate(frame[column].astype(str).to_numpy()):
            if value in lookup:
                contribution[row] = coefficients[offset + lookup[value]]
        values[column] = contribution
        offset += len(categories) - 1
    return values


def factor_summary(frame: pd.DataFrame) -> pd.DataFrame:
    artifact = joblib.load(MODEL)
    design, coefficients = artifact["design"], artifact["model"].coef_.ravel()
    expected = set(design["numeric"]) | set(design["categorical"])
    if not expected.issubset(frame.columns):
        raise RuntimeError("scored_feature_contract_missing")
    contributions: dict[str, np.ndarray] = {}
    for index, column in enumerate(design["numeric"]):
        standardized = (pd.to_numeric(frame[column], errors="raise").to_numpy(float) - float(design["means"][column])) / float(design["stds"][column])
        contributions[column] = standardized * coefficients[index]
    contributions.update(categorical_contribution(frame, design, coefficients))
    rows = []
    for group, columns in FACTOR_GROUPS.items():
        contribution = sum((contributions[column] for column in columns), np.zeros(len(frame), dtype=float))
        rows.append(
            {
                "factor_group": group,
                "source_columns_json": json.dumps(columns, ensure_ascii=True),
                "sd_logit_contribution": float(np.std(contribution)),
                "mean_abs_logit_contribution": float(np.mean(np.abs(contribution))),
            }
        )
    return pd.DataFrame(rows).sort_values(["sd_logit_contribution", "factor_group"], ascending=[False, True]).reset_index(drop=True)


def map_side_summary(frame: pd.DataFrame) -> pd.DataFrame:
    focal = frame[["map_name", "focal_side", "focal_wins"]].rename(columns={"focal_side": "side", "focal_wins": "win"})
    opponent = frame[["map_name", "focal_side", "focal_wins"]].copy()
    opponent["side"] = opponent["focal_side"].map({"T": "CT", "CT": "T"})
    opponent["win"] = 1 - opponent["focal_wins"]
    participant = pd.concat([focal, opponent[["map_name", "side", "win"]]], ignore_index=True)
    return (
        participant.groupby(["map_name", "side"], sort=True)
        .agg(observed_duels=("win", "size"), wins=("win", "sum"), win_rate=("win", "mean"))
        .reset_index()
        .sort_values(["side", "win_rate", "map_name"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def write_results(output: Path, lock: dict[str, Any]) -> None:
    player_top, aliases = player_summary()
    scored = pd.read_parquet(SCORED)
    factors = factor_summary(scored)
    maps = map_side_summary(scored)
    top_maps = maps.groupby("side", sort=True).head(1).reset_index(drop=True)
    player_top.to_parquet(output / "top_20_player_adjusted_values.parquet", index=False, compression="zstd")
    aliases.to_parquet(output / "top_20_player_name_audit.parquet", index=False, compression="zstd")
    factors.to_parquet(output / "factor_group_contributions.parquet", index=False, compression="zstd")
    maps.to_parquet(output / "map_side_observed_win_rates.parquet", index=False, compression="zstd")
    top_maps.to_parquet(output / "top_map_by_side.parquet", index=False, compression="zstd")
    summary = {
        "stage": "exploratory_duel_side_checks_v1",
        "status": "passed",
        "classification": lock["classification"],
        "player_metric": lock["player_metric"],
        "player_scope": {
            "physical_duels": 57501 + 32786,
            "participant_records": 2 * (57501 + 32786),
            "minimum_participations": MIN_PLAYER_PARTICIPATIONS,
            "ranked_players": int(len(player_top)),
        },
        "top_5_factor_groups": factors.head(5).to_dict(orient="records"),
        "top_map_by_side": top_maps.to_dict(orient="records"),
        "interpretation": "Descriptive post-hoc summaries only. The player residual uses a model that already includes player identity; factor contribution is model reliance, not a causal decomposition; map rates are observed duel rates, not round win rates.",
    }
    write_json_new(output / "summary.json", summary)
    file_hashes = {path.name: sha256(path) for path in output.glob("*.parquet")}
    write_json_new(output / "file_sha256.json", file_hashes)
    write_json_new(
        output / "manifest.json",
        summary
        | {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "input_lock_sha256": sha256(output / "checkpoint_01_input_lock.json"),
            "summary_sha256": sha256(output / "summary.json"),
            "file_sha256_sha256": sha256(output / "file_sha256.json"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "exploratory_duel_side_checks_v1")
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        print(json.dumps(json.loads((output / "manifest.json").read_text(encoding="utf-8")), ensure_ascii=True, indent=2))
        return
    lock = lock_inputs(output)
    write_results(output, lock)
    print((output / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
