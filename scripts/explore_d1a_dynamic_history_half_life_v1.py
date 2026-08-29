"""Select a D1-A dynamic residual-history probability model, then score it once on D1-B/C."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
HIST = Path(r"C:\Users\Admin\Desktop\information flow for hot-hand-effect\history\2026-08-25_main_before_restart_20260825\results\p_hat_final_v8_position_continuity\candidate.parquet")
D1B = ROOT / "results" / "external_generalization_v3" / "D1-B" / "scored.parquet"
D1C = ROOT / "results" / "external_candidates_v3" / "D1-C" / "candidate.parquet"
ASSET = ROOT / "results" / "conditional_residual_asset_d1_v2"
LAG1 = ROOT / "results" / "first_order_residual_round_boundary_interaction_d1_v1"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "71_D1A动态历史残差概率模型半衰期开发执行锁_v1_2026-08-29.md"
OUT = ROOT / "results" / "d1a_dynamic_history_half_life_v1"
DATASETS = ("D1-A", "D1-B", "D1-C")
HALF_LIVES = (2, 5, 10, 20, 30, 45, 60, 90, 120, 180, 300)
FOLDS = (("F1", 1, 100, 101, 150), ("F2", 1, 150, 151, 200), ("F3", 1, 200, 201, 250), ("F4", 1, 250, 251, 300))
C, SEED, MAX_ITER, TOL, EPS, DELTA = 0.01, 20260829, 5000, 1e-8, 1e-12, 0.0001
NUMERIC = (
    "focal_minus_opponent_health", "focal_minus_opponent_team_alive", "focal_speed", "opponent_speed", "position_distance",
    "focal_has_armor", "opponent_has_armor", "focal_has_helmet", "opponent_has_helmet", "focal_has_defuser", "opponent_has_defuser",
    "focal_minus_opponent_post_buy_money", "focal_minus_opponent_post_buy_equipment_value", "focal_minus_opponent_team_post_buy_equipment_value",
    "absolute_vertical_position_difference", "relative_radial_speed", "relative_transverse_speed", "relative_angular_speed", "focal_facing_error_radians", "opponent_facing_error_radians", "focal_nearest_living_teammate_distance", "opponent_nearest_living_teammate_distance", "focal_nearest_living_enemy_distance", "opponent_nearest_living_enemy_distance", "focal_is_scoped", "opponent_is_scoped", "focal_is_alone", "opponent_is_alone",
    "focal_is_flashed", "opponent_is_flashed", "focal_flash_remaining_seconds", "opponent_flash_remaining_seconds", "bomb_planted", "bomb_time_remaining_seconds", "focal_is_defusing", "opponent_is_defusing",
)
CATEGORICAL = ("focal_side", "map_name", "focal_speed_source", "opponent_speed_source", "focal_weapon", "opponent_weapon", "weapon_pair", "focal_steamid", "opponent_steamid")


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    require(not path.exists(), f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def verify_tree(root: Path) -> dict[str, Any]:
    expected = read_json(root / "file_sha256.json")
    bad = [name for name, value in expected.items() if not (root / name).is_file() or digest(root / name) != value]
    return {"checked_files": len(expected), "mismatches": bad, "manifest_file_sha256_matches": read_json(root / "manifest.json").get("file_sha256_sha256") == digest(root / "file_sha256.json")}


def raw_path(dataset: str) -> Path:
    return {"D1-A": HIST, "D1-B": D1B, "D1-C": D1C}[dataset]


def input_paths() -> list[Path]:
    paths = [PROTOCOL, Path(__file__).resolve(), HIST, D1B, D1C, ASSET / "manifest.json", ASSET / "file_sha256.json", LAG1 / "manifest.json", LAG1 / "file_sha256.json", LAG1 / "checkpoint_04_gate_audit.json"]
    for dataset in DATASETS:
        paths += [ASSET / dataset / "residual_asset.parquet", ASSET / dataset / "checkpoint.json", LAG1 / "state" / dataset / "analysis_matrix.parquet", LAG1 / "state" / dataset / f"checkpoint_03_analysis_matrix_{dataset}.json"]
    return paths


def input_lock() -> tuple[dict[str, Any], str]:
    lock_path = OUT / "checkpoint_01_input_lock.json"
    asset_audit, lag1_audit = verify_tree(ASSET), verify_tree(LAG1)
    require(not asset_audit["mismatches"] and asset_audit["manifest_file_sha256_matches"], "conditional_asset_hash_chain_mismatch")
    require(not lag1_audit["mismatches"] and lag1_audit["manifest_file_sha256_matches"], "lag1_hash_chain_mismatch")
    records = {str(path): digest(path) for path in input_paths()}
    if lock_path.exists():
        saved = read_json(lock_path)
        require(saved.get("inputs") == records, "input_lock_source_hash_changed")
        return saved, digest(lock_path)
    value = {
        "stage": "d1a_dynamic_history_half_life_v1_input_lock", "created_at_utc": now(),
        "classification": "exploratory_nonconfirmatory_result_informed_d1_only", "inputs": records,
        "conditional_asset_hash_audit": asset_audit, "lag1_hash_audit": lag1_audit,
        "half_life_seconds": list(HALF_LIVES), "delta_match_equal_log_loss": DELTA,
        "folds": [{"name": name, "train_queue_order": [start, end], "evaluation_queue_order": [valid_start, valid_end]} for name, start, end, valid_start, valid_end in FOLDS],
        "regularization": {"penalty": "l2", "C": C, "solver": "lbfgs", "max_iter": MAX_ITER, "tol": TOL}, "e1_path_access_count": 0,
    }
    write_json(lock_path, value)
    return value, digest(lock_path)


def raw_candidate(dataset: str) -> pd.DataFrame:
    frame = pd.read_parquet(raw_path(dataset))
    if dataset == "D1-A":
        frame = frame.loc[frame["partition"].astype(str).eq("D1-A")].copy()
    required = {"duel_id", "match_id", "map_id", "event_tick", *NUMERIC, *CATEGORICAL}
    if dataset == "D1-A":
        required.add("queue_order")
    require(required.issubset(frame.columns) and not frame["duel_id"].duplicated().any(), f"raw_candidate_contract_failure:{dataset}")
    for field in NUMERIC:
        require(np.isfinite(frame[field].to_numpy(float)).all(), f"nonfinite_raw_numeric:{dataset}:{field}")
    for field in CATEGORICAL:
        values = frame[field].astype("string")
        require(not values.isna().any() and not values.str.strip().eq("").any(), f"invalid_raw_category:{dataset}:{field}")
    return frame


def player_events(asset: pd.DataFrame) -> pd.DataFrame:
    base = asset.loc[asset["residual_valid"].astype(bool) & asset["primary_analysis_eligible"].astype(bool)].copy()
    require(not base["physical_event_id"].duplicated().any(), "duplicate_physical_event_in_asset")
    focal = base[["physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id", "e_t"]].rename(columns={"focal_player_id": "player_id", "e_t": "signed_e"})
    opponent = base[["physical_event_id", "match_id", "map_id", "event_tick", "opponent_player_id", "e_t"]].rename(columns={"opponent_player_id": "player_id", "e_t": "signed_e"})
    opponent["signed_e"] = -opponent["signed_e"].to_numpy(float)
    events = pd.concat([focal, opponent], ignore_index=True)
    for field in ("physical_event_id", "match_id", "map_id", "player_id"):
        events[field] = events[field].astype(str)
    require(not events.duplicated(["physical_event_id", "player_id"]).any(), "duplicate_player_view_event")
    return events.sort_values(["match_id", "map_id", "player_id", "event_tick", "physical_event_id"], kind="stable").reset_index(drop=True)


def construct_states(asset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    events = player_events(asset)
    records: list[dict[str, Any]] = []
    max_direct_error = {h: 0.0 for h in HALF_LIVES}
    same_tick_batches = 0
    for (match_id, map_id, player_id), group in events.groupby(["match_id", "map_id", "player_id"], sort=False, dropna=False):
        prior_tick: float | None = None
        running = np.zeros(len(HALF_LIVES), dtype=float)
        earlier: list[tuple[float, float]] = []
        lag_value: float | None = None
        lag_tick: float | None = None
        for tick, batch in group.groupby("event_tick", sort=False, dropna=False):
            tick_value = float(tick)
            if prior_tick is None:
                before = np.full(len(HALF_LIVES), np.nan)
            else:
                elapsed_tick = tick_value - prior_tick
                q = np.exp(-math.log(2.0) * elapsed_tick / (np.asarray(HALF_LIVES, dtype=float) * 64.0))
                before = q * running
                for index, h in enumerate(HALF_LIVES):
                    direct = sum(value * math.exp(-math.log(2.0) * (tick_value - old_tick) / (h * 64.0)) for old_tick, value in earlier)
                    max_direct_error[h] = max(max_direct_error[h], abs(float(before[index]) - direct))
            if len(batch) > 1:
                same_tick_batches += 1
            elapsed_seconds = np.nan if lag_tick is None else (tick_value - lag_tick) / 64.0
            for row in batch.itertuples(index=False):
                record: dict[str, Any] = {
                    "physical_event_id": str(row.physical_event_id), "match_id": str(match_id), "map_id": str(map_id), "focal_player_id": str(player_id), "event_tick": tick_value,
                    "reconstructed_H_t": np.nan if lag_value is None else float(lag_value), "history_event_tick": np.nan if lag_tick is None else float(lag_tick), "T_seconds": elapsed_seconds,
                }
                for index, h in enumerate(HALF_LIVES):
                    record[f"S_h{h}"] = float(before[index])
                records.append(record)
            if prior_tick is None:
                running = np.zeros(len(HALF_LIVES), dtype=float)
            else:
                running = before
            for row in batch.itertuples(index=False):
                earlier.append((tick_value, float(row.signed_e)))
            running += float(batch["signed_e"].sum())
            prior_tick = tick_value
            if len(batch) == 1:
                lag_value, lag_tick = float(batch.iloc[0]["signed_e"]), tick_value
            else:
                lag_value, lag_tick = None, None
    return pd.DataFrame.from_records(records), {"player_view_events": int(len(events)), "same_tick_batches": same_tick_batches, "max_direct_recursion_abs_error_by_half_life": {str(h): value for h, value in max_direct_error.items()}}


def build_matrix(dataset: str, lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrix_path = OUT / "state" / dataset / "model_matrix.parquet"
    checkpoint_path = OUT / "state" / dataset / f"checkpoint_02_matrix_{dataset}.json"
    if matrix_path.exists():
        checkpoint = read_json(checkpoint_path)
        require(checkpoint.get("input_lock_sha256") == lock_hash and checkpoint.get("matrix_sha256") == digest(matrix_path), f"matrix_checkpoint_mismatch:{dataset}")
        return pd.read_parquet(matrix_path), checkpoint
    raw = raw_candidate(dataset)
    asset = pd.read_parquet(ASSET / dataset / "residual_asset.parquet")
    asset = asset.loc[asset["residual_valid"].astype(bool) & asset["primary_analysis_eligible"].astype(bool)].copy()
    states, audit = construct_states(asset)
    joined = raw.merge(asset[["duel_id", "physical_event_id", "match_id", "map_id", "focal_player_id", "event_tick", "e_t", "Y_t"]], on="duel_id", how="inner", validate="one_to_one")
    if "focal_wins" in joined:
        require(np.array_equal(joined["focal_wins"].to_numpy(int), joined["Y_t"].to_numpy(int)), f"raw_asset_outcome_mismatch:{dataset}")
    for frame in (joined, states):
        for field in ("physical_event_id", "match_id", "map_id", "focal_player_id"):
            frame[field] = frame[field].astype(str)
    merged = joined.merge(states, on=["physical_event_id", "match_id", "map_id", "focal_player_id", "event_tick"], how="inner", validate="one_to_one")
    merged = merged.loc[merged["reconstructed_H_t"].notna()].copy()
    lag1 = pd.read_parquet(LAG1 / "state" / dataset / "analysis_matrix.parquet", columns=["match_id", "map_id", "focal_player_id", "event_tick", "e_t", "H_t", "history_source_event_tick", "time_since_history_tick"])
    for field in ("match_id", "map_id", "focal_player_id"):
        lag1[field] = lag1[field].astype(str)
    keys = ["match_id", "map_id", "focal_player_id", "event_tick", "e_t"]
    require(not lag1.duplicated(keys).any(), f"nonunique_lag1_signature:{dataset}")
    merged = merged.merge(lag1, on=keys, how="inner", validate="one_to_one")
    require(len(merged) == len(lag1), f"lag1_state_join_loss:{dataset}")
    h_error = float(np.max(np.abs(merged["reconstructed_H_t"].to_numpy(float) - merged["H_t"].to_numpy(float))))
    tick_error = float(np.max(np.abs(merged["history_event_tick"].to_numpy(float) - merged["history_source_event_tick"].to_numpy(float))))
    T_error = float(np.max(np.abs(merged["T_seconds"].to_numpy(float) - merged["time_since_history_tick"].to_numpy(float) / 64.0)))
    require(h_error <= 1e-10 and tick_error <= 1e-10 and T_error <= 1e-10, f"lag1_reconstruction_error:{dataset}:{h_error}:{tick_error}:{T_error}")
    merged["z_time"] = np.log2(1.0 + merged["T_seconds"].to_numpy(float))
    for h in HALF_LIVES:
        merged[f"M2_h{h}"] = merged["H_t"].to_numpy(float) * np.exp(-math.log(2.0) * merged["T_seconds"].to_numpy(float) / h)
    all_numeric = [*NUMERIC, "z_time", "H_t", *[f"M2_h{h}" for h in HALF_LIVES], *[f"S_h{h}" for h in HALF_LIVES]]
    require(np.isfinite(merged[all_numeric].to_numpy(float)).all() and merged["Y_t"].isin([0, 1]).all(), f"nonfinite_or_nonbinary_matrix:{dataset}")
    output_fields = ["duel_id", "match_id", "Y_t", *NUMERIC, *CATEGORICAL, "z_time", "H_t", *[f"M2_h{h}" for h in HALF_LIVES], *[f"S_h{h}" for h in HALF_LIVES]]
    if dataset == "D1-A":
        output_fields.insert(3, "queue_order")
        queue = merged["queue_order"].to_numpy(int)
        require(queue.min() == 1 and queue.max() == 300 and merged.groupby("match_id")["queue_order"].nunique().eq(1).all(), "d1a_queue_integrity_failure")
    output = merged[output_fields].copy()
    direct_errors = audit["max_direct_recursion_abs_error_by_half_life"]
    require(all(value <= 1e-10 for value in direct_errors.values()), f"state_direct_recursion_failure:{dataset}:{direct_errors}")
    write_parquet(matrix_path, output)
    metrics = {"rows": int(len(output)), "matches": int(output["match_id"].nunique()), "players": int(merged["focal_player_id"].nunique()), "max_abs_H_error": h_error, "max_abs_history_tick_error": tick_error, "max_abs_T_seconds_error": T_error, **audit}
    checkpoint = {"stage": "d1a_dynamic_history_half_life_v1_matrix", "created_at_utc": now(), "dataset": dataset, "input_lock_sha256": lock_hash, "matrix_sha256": digest(matrix_path), "G1": {"status": "pass", "reason": "strict_signed_history_and_lag1_reconstruction", "metrics": metrics}}
    write_json(checkpoint_path, checkpoint)
    return output, checkpoint


class Design:
    def __init__(self, extra: str | None) -> None:
        self.numeric = [*NUMERIC, "z_time"] + ([] if extra is None else [extra])
        self.means: dict[str, float] = {}
        self.sds: dict[str, float] = {}
        self.categories: dict[str, list[str]] = {}
        self.names: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "Design":
        for field in self.numeric:
            values = frame[field].to_numpy(float)
            self.means[field], self.sds[field] = float(values.mean()), float(values.std(ddof=0))
            require(np.isfinite(values).all() and math.isfinite(self.sds[field]) and self.sds[field] > 0, f"invalid_training_scale:{field}")
        self.names = [f"num_{field}" for field in self.numeric]
        for field in CATEGORICAL:
            levels = sorted(frame[field].astype(str).unique().tolist())
            require(levels, f"empty_category:{field}")
            self.categories[field] = levels
            self.names.extend(f"cat_{field}={value}" for value in levels[1:])
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[sparse.csr_matrix, dict[str, int]]:
        numeric = np.column_stack([(frame[field].to_numpy(float) - self.means[field]) / self.sds[field] for field in self.numeric])
        blocks: list[sparse.csr_matrix] = [sparse.csr_matrix(numeric)]
        unknown: dict[str, int] = {}
        for field in CATEGORICAL:
            lookup = {value: index for index, value in enumerate(self.categories[field][1:])}
            values = frame[field].astype(str).to_numpy()
            rows = np.fromiter((index for index, value in enumerate(values) if value in lookup), dtype=np.int64)
            cols = np.fromiter((lookup[value] for value in values if value in lookup), dtype=np.int64)
            blocks.append(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(frame), len(lookup))))
            unknown[field] = int(sum(value not in self.categories[field] for value in values))
        return sparse.hstack(blocks, format="csr"), unknown


def candidates() -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = [{"name": "M0", "kind": "M0", "half_life_seconds": None, "feature": None}, {"name": "M1", "kind": "M1", "half_life_seconds": None, "feature": "H_t"}]
    for h in HALF_LIVES:
        values.append({"name": f"M2_h{h}", "kind": "M2", "half_life_seconds": h, "feature": f"M2_h{h}"})
        values.append({"name": f"M3_h{h}", "kind": "M3", "half_life_seconds": h, "feature": f"S_h{h}"})
    return values


def fit_predict(train: pd.DataFrame, evaluation: pd.DataFrame, candidate: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    design = Design(candidate["feature"]).fit(train)
    x_train, unknown_train = design.transform(train)
    x_eval, unknown_eval = design.transform(evaluation)
    require(x_train.shape[0] > x_train.shape[1] and x_train.shape[1] > 0, f"design_dimension_failure:{candidate['name']}")
    fitted = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=MAX_ITER, tol=TOL, random_state=SEED)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fitted.fit(x_train, train["Y_t"].to_numpy(int))
    require(not any(issubclass(item.category, ConvergenceWarning) for item in caught), f"convergence_warning:{candidate['name']}")
    probability = fitted.predict_proba(x_eval)[:, 1]
    require(np.isfinite(probability).all() and np.all((probability > 0.0) & (probability < 1.0)), f"invalid_probability:{candidate['name']}")
    coefficient = None if candidate["feature"] is None else float(fitted.coef_[0][design.numeric.index(candidate["feature"])])
    return probability, {"features": design.numeric, "means": design.means, "sds": design.sds, "intercept": float(fitted.intercept_[0]), "state_coefficient": coefficient, "state_coefficient_direction": None if coefficient is None else ("positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero"), "n_iter": int(fitted.n_iter_[0]), "unknown_categories_training": unknown_train, "unknown_categories_evaluation": unknown_eval, "design_shape": [int(x_train.shape[0]), int(x_train.shape[1])]} 


def smoke(matrices: dict[str, pd.DataFrame], lock_hash: str) -> None:
    path = OUT / "checkpoint_03_smoke.json"
    if path.exists():
        require(read_json(path).get("input_lock_sha256") == lock_hash, "smoke_input_lock_mismatch")
        return
    train = matrices["D1-A"].loc[matrices["D1-A"]["queue_order"].between(1, 100)].copy()
    matches = sorted(matrices["D1-B"]["match_id"].astype(str).unique())[:30]
    evaluation = matrices["D1-B"].loc[matrices["D1-B"]["match_id"].astype(str).isin(matches)].copy()
    require(len(matches) == 30 and not train.empty and not evaluation.empty, "smoke_support_failure")
    details = {}
    for candidate in candidates():
        probability, fitted = fit_predict(train, evaluation, candidate)
        details[candidate["name"]] = {"n_iter": fitted["n_iter"], "probability_range": [float(probability.min()), float(probability.max())], "design_shape": fitted["design_shape"]}
    write_json(path, {"stage": "d1a_dynamic_history_half_life_v1_smoke", "created_at_utc": now(), "input_lock_sha256": lock_hash, "G0": {"status": "pass"}, "G1": {"status": "pass"}, "G2": {"status": "pass", "training_queue_order": [1, 100], "evaluation_matches": len(matches)}, "G3": {"status": "pass", "all_candidates": details}})


def d1a_fold(name: str, train_start: int, train_end: int, valid_start: int, valid_end: int, matrix: pd.DataFrame, lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = OUT / "state" / name / "oof_predictions.parquet"
    checkpoint_path = OUT / "state" / name / "checkpoint_04_oof.json"
    if path.exists():
        checkpoint = read_json(checkpoint_path)
        require(checkpoint.get("input_lock_sha256") == lock_hash and checkpoint.get("oof_sha256") == digest(path), f"oof_checkpoint_mismatch:{name}")
        return pd.read_parquet(path), checkpoint
    train = matrix.loc[matrix["queue_order"].between(train_start, train_end)].copy()
    evaluation = matrix.loc[matrix["queue_order"].between(valid_start, valid_end)].copy()
    require(not train.empty and not evaluation.empty and set(train["match_id"]).isdisjoint(set(evaluation["match_id"])), f"forward_fold_failure:{name}")
    require(int(train["queue_order"].max()) < int(evaluation["queue_order"].min()), f"nonforward_fold:{name}")
    output = evaluation[["duel_id", "match_id", "Y_t"]].copy()
    fits: dict[str, Any] = {}
    for candidate in candidates():
        output[candidate["name"]], fits[candidate["name"]] = fit_predict(train, evaluation, candidate)
    write_parquet(path, output)
    fit_path = OUT / "state" / name / "training_fit.json"
    write_json(fit_path, {"stage": "d1a_dynamic_history_half_life_v1_training_fit", "created_at_utc": now(), "input_lock_sha256": lock_hash, "fold": name, "train_queue_order": [train_start, train_end], "evaluation_queue_order": [valid_start, valid_end], "train_rows": int(len(train)), "train_matches": int(train["match_id"].nunique()), "evaluation_rows": int(len(evaluation)), "evaluation_matches": int(evaluation["match_id"].nunique()), "models": fits})
    checkpoint = {"stage": "d1a_dynamic_history_half_life_v1_oof", "created_at_utc": now(), "input_lock_sha256": lock_hash, "fold": name, "oof_sha256": digest(path), "training_fit_sha256": digest(fit_path), "G2": {"status": "pass", "train_matches": int(train["match_id"].nunique()), "evaluation_matches": int(evaluation["match_id"].nunique())}, "G3": {"status": "pass", "reason": "all_candidates_converged_with_finite_probabilities"}}
    write_json(checkpoint_path, checkpoint)
    return output, checkpoint


def match_metrics(oof: pd.DataFrame, fold: str) -> pd.DataFrame:
    values: dict[str, Any] = {"match_id": oof["match_id"].astype(str)}
    y = oof["Y_t"].to_numpy(float)
    for candidate in candidates():
        name = candidate["name"]
        probability = np.clip(oof[name].to_numpy(float), EPS, 1.0 - EPS)
        values[f"{name}_log_loss"] = -(y * np.log(probability) + (1.0 - y) * np.log1p(-probability))
        values[f"{name}_brier"] = (probability - y) ** 2
    output = pd.DataFrame(values).groupby("match_id", sort=True).mean().reset_index()
    output.insert(0, "fold", fold)
    return output


def select_model(metrics: pd.DataFrame, lock_hash: str) -> dict[str, Any]:
    summary_path = OUT / "d1a_candidate_summary.parquet"
    selection_path = OUT / "checkpoint_05_selection.json"
    if selection_path.exists():
        value = read_json(selection_path)
        require(value.get("input_lock_sha256") == lock_hash and value.get("candidate_summary_sha256") == digest(summary_path), "selection_checkpoint_mismatch")
        return value
    records = []
    for candidate in candidates():
        name = candidate["name"]
        by_fold = metrics.groupby("fold", sort=True)[[f"{name}_log_loss", f"{name}_brier"]].mean()
        require(len(by_fold) == len(FOLDS), f"missing_fold_metric:{name}")
        records.append({**candidate, "mean_match_equal_log_loss": float(by_fold[f"{name}_log_loss"].mean()), "mean_match_equal_brier": float(by_fold[f"{name}_brier"].mean()), "fold_log_loss_sd": float(by_fold[f"{name}_log_loss"].std(ddof=1)), "fold_brier_sd": float(by_fold[f"{name}_brier"].std(ddof=1))})
    summary = pd.DataFrame(records)
    minimum = float(summary["mean_match_equal_log_loss"].min())
    summary["within_delta"] = summary["mean_match_equal_log_loss"] <= minimum + DELTA + 1e-15
    summary["complexity_priority"] = summary["kind"].map({"M0": 0, "M1": 1, "M3": 2, "M2": 3}).astype(int)
    write_parquet(summary_path, summary.sort_values(["mean_match_equal_log_loss", "name"], kind="stable"))
    tied = summary.loc[summary["within_delta"]].sort_values(["complexity_priority", "fold_log_loss_sd", "half_life_seconds", "name"], kind="stable", na_position="first")
    selected = tied.iloc[0].to_dict()
    selection = {"stage": "d1a_dynamic_history_half_life_v1_selection", "created_at_utc": now(), "input_lock_sha256": lock_hash, "candidate_summary_sha256": digest(summary_path), "selection_metric": "mean_match_equal_log_loss_over_four_forward_folds", "delta_match_equal_log_loss": DELTA, "minimum_log_loss": minimum, "within_delta_candidates": tied[["name", "kind", "half_life_seconds", "mean_match_equal_log_loss", "fold_log_loss_sd", "complexity_priority"]].to_dict(orient="records"), "selected": {key: (None if pd.isna(value) else value) for key, value in selected.items()}, "G4": {"status": "pass", "reason": "complete_candidate_grid_and_deterministic_selection"}}
    write_json(selection_path, selection)
    return selection


def fixed_score(name: str, train: pd.DataFrame, evaluation: pd.DataFrame, selected: dict[str, Any], lock_hash: str) -> dict[str, Any]:
    path = OUT / "state" / name / "fixed_spec_predictions.parquet"
    checkpoint_path = OUT / "state" / name / "checkpoint_06_fixed_score.json"
    if path.exists():
        checkpoint = read_json(checkpoint_path)
        require(checkpoint.get("input_lock_sha256") == lock_hash and checkpoint.get("predictions_sha256") == digest(path), f"fixed_score_checkpoint_mismatch:{name}")
        return checkpoint
    candidate = next(item for item in candidates() if item["name"] == selected["name"])
    require(set(train["match_id"]).isdisjoint(set(evaluation["match_id"])), f"fixed_score_match_overlap:{name}")
    probability, fit = fit_predict(train, evaluation, candidate)
    output = evaluation[["duel_id", "match_id", "Y_t"]].copy()
    output["probability"] = probability
    write_parquet(path, output)
    y, p = output["Y_t"].to_numpy(float), np.clip(output["probability"].to_numpy(float), EPS, 1.0 - EPS)
    per_row = pd.DataFrame({"match_id": output["match_id"].astype(str), "log_loss": -(y * np.log(p) + (1.0 - y) * np.log1p(-p)), "brier": (p - y) ** 2})
    per_match = per_row.groupby("match_id", sort=True).mean()
    checkpoint = {"stage": "d1a_dynamic_history_half_life_v1_fixed_score", "created_at_utc": now(), "input_lock_sha256": lock_hash, "score_name": name, "selected_model": candidate, "train_rows": int(len(train)), "train_matches": int(train["match_id"].nunique()), "evaluation_rows": int(len(evaluation)), "evaluation_matches": int(evaluation["match_id"].nunique()), "match_equal_log_loss": float(per_match["log_loss"].mean()), "match_equal_brier": float(per_match["brier"].mean()), "fit": fit, "predictions_sha256": digest(path), "G5": {"status": "pass", "reason": "single_fixed_specification_score"}}
    write_json(checkpoint_path, checkpoint)
    return checkpoint


def output_hashes() -> dict[str, str]:
    return {str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"} and not any(part.startswith("failed_") for part in path.relative_to(OUT).parts)}


def run(phase: str) -> None:
    _, lock_hash = input_lock()
    matrices, matrix_checks = {}, {}
    for dataset in DATASETS:
        matrices[dataset], matrix_checks[dataset] = build_matrix(dataset, lock_hash)
    smoke(matrices, lock_hash)
    if phase == "smoke":
        print(json.dumps({"status": "smoke_passed", "output_root": str(OUT)}, ensure_ascii=True))
        return
    oof_parts, fold_checks = [], {}
    for fold in FOLDS:
        name, train_start, train_end, valid_start, valid_end = fold
        oof, fold_checks[name] = d1a_fold(name, train_start, train_end, valid_start, valid_end, matrices["D1-A"], lock_hash)
        oof_parts.append(match_metrics(oof, name))
    metrics_path = OUT / "d1a_match_metrics.parquet"
    if metrics_path.exists():
        metrics = pd.read_parquet(metrics_path)
    else:
        metrics = pd.concat(oof_parts, ignore_index=True)
        write_parquet(metrics_path, metrics)
    selection = select_model(metrics, lock_hash)
    selected = selection["selected"]
    d1b = fixed_score("D1-B", matrices["D1-A"], matrices["D1-B"], selected, lock_hash)
    d1c = fixed_score("D1-C", pd.concat([matrices["D1-A"], matrices["D1-B"]], ignore_index=True), matrices["D1-C"], selected, lock_hash)
    gates = {"G0": {"status": "pass", "reason": "hash_locked_d1_only_inputs"}, "G1": {"status": "pass", "metrics": {dataset: check["G1"] for dataset, check in matrix_checks.items()}}, "G2": {"status": "pass", "metrics": {name: check["G2"] for name, check in fold_checks.items()}}, "G3": {"status": "pass", "metrics": {name: check["G3"] for name, check in fold_checks.items()}}, "G4": selection["G4"], "G5": {"status": "pass", "D1-B": d1b["G5"], "D1-C": d1c["G5"]}}
    gate_path = OUT / "checkpoint_07_gate_audit.json"
    if not gate_path.exists():
        write_json(gate_path, {"stage": "d1a_dynamic_history_half_life_v1_gates", "created_at_utc": now(), "input_lock_sha256": lock_hash, "gates": gates})
    hashes_path = OUT / "file_sha256.json"
    if not hashes_path.exists():
        write_json(hashes_path, output_hashes())
    else:
        for relative, expected in read_json(hashes_path).items():
            require((OUT / relative).is_file() and digest(OUT / relative) == expected, f"output_hash_mismatch:{relative}")
    manifest_path = OUT / "manifest.json"
    if not manifest_path.exists():
        write_json(manifest_path, {"stage": "d1a_dynamic_history_half_life_v1", "status": "passed", "created_at_utc": now(), "classification": "exploratory_nonconfirmatory_result_informed_d1_only", "input_lock_sha256": lock_hash, "selected_model": selected, "gate_statuses": {name: value["status"] for name, value in gates.items()}, "interpretation": "D1-A forward-CV dynamic residual-history probability model selection followed by fixed-specification D1-B/C scores; exploratory result-informed evidence only, not independent validation, p_hat replacement, hot hand, state, causal, deployable, or confirmatory evidence.", "file_sha256_sha256": digest(hashes_path)})
    print(json.dumps({"status": "passed", "manifest": str(manifest_path), "selected": selected}, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    try:
        run(args.phase)
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True)
        failed = OUT / f"failed_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        failed.mkdir(exist_ok=False)
        write_json(failed / "failure_manifest.json", {"stage": "d1a_dynamic_history_half_life_v1", "status": "failed", "created_at_utc": now(), "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
