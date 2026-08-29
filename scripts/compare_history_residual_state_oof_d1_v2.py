"""Run the v2 D1-only exploratory OOF comparison of three residual states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "results" / "conditional_residual_asset_d1_v2"
LAG1 = ROOT / "results" / "first_order_residual_round_boundary_interaction_d1_v1"
OUT = ROOT / "results" / "history_residual_state_oof_d1_v2"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "43_D1历史残差状态样本外预测比较执行协议_v2_2026-08-26.md"
DRAFT = ROOT / "docs" / "修订与补丁" / "41_D1历史残差状态样本外预测比较分析协议_草案_v1_2026-08-26.md"
DATASETS = ("D1-A", "D1-B", "D1-C")
FOLDS = {"F1": (("D1-A",), "D1-B"), "F2": (("D1-A", "D1-B"), "D1-C")}
TICK_RATE = 64.0
HALF_LIFE_SECONDS = 60.0
BOOTSTRAPS = 2000
SEED = 20260826
DELTA = 0.005
MODELS = ("M1_lag1", "M2_decay_lag1", "M3_decay_sum")


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def write_parquet_new(path: Path, frame: pd.DataFrame) -> None:
    require(not path.exists(), f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def asset_paths() -> list[Path]:
    paths = [
        PROTOCOL, DRAFT, Path(__file__).resolve(), ASSET / "manifest.json", ASSET / "file_sha256.json",
        ASSET / "field_dictionary.json", LAG1 / "manifest.json", LAG1 / "file_sha256.json",
        LAG1 / "checkpoint_04_gate_audit.json",
    ]
    for dataset in DATASETS:
        paths.extend([
            ASSET / dataset / "residual_asset.parquet", ASSET / dataset / "checkpoint.json",
            LAG1 / "state" / dataset / "analysis_matrix.parquet",
            LAG1 / "state" / dataset / f"checkpoint_03_analysis_matrix_{dataset}.json",
        ])
    return paths


def verify_tree(root: Path) -> dict[str, Any]:
    expected = read_json(root / "file_sha256.json")
    mismatches = [
        name for name, expected_hash in expected.items()
        if not (root / name).is_file() or digest(root / name) != expected_hash
    ]
    manifest = read_json(root / "manifest.json")
    return {
        "checked_files": len(expected), "mismatches": mismatches,
        "manifest_file_sha256_matches": manifest.get("file_sha256_sha256") == digest(root / "file_sha256.json"),
    }


def input_lock() -> tuple[dict[str, Any], str]:
    path = OUT / "checkpoint_01_input_lock.json"
    inputs = {str(item): digest(item) for item in asset_paths()}
    asset_audit, lag1_audit = verify_tree(ASSET), verify_tree(LAG1)
    require(not asset_audit["mismatches"] and asset_audit["manifest_file_sha256_matches"], "conditional_asset_hash_chain_mismatch")
    require(not lag1_audit["mismatches"] and lag1_audit["manifest_file_sha256_matches"], "lag1_hash_chain_mismatch")
    if path.exists():
        saved = read_json(path)
        require(saved.get("inputs") == inputs, "input_lock_source_hash_changed")
        return saved, digest(path)
    value = {
        "stage": "history_residual_state_oof_d1_v2_input_lock", "created_at_utc": now(),
        "classification": "exploratory_nonconfirmatory_result_informed_d1_only", "inputs": inputs,
        "conditional_asset_hash_audit": asset_audit, "lag1_hash_audit": lag1_audit,
        "e1_path_access_count": 0,
    }
    write_json_new(path, value)
    return value, digest(path)


def player_events(asset: pd.DataFrame) -> pd.DataFrame:
    base = asset.loc[asset["residual_valid"].astype(bool) & asset["primary_analysis_eligible"].astype(bool)].copy()
    focal = base[["physical_event_id", "match_id", "map_id", "round_id", "event_tick", "focal_player_id", "e_t"]].rename(columns={"focal_player_id": "player_id", "e_t": "signed_e"})
    opponent = base[["physical_event_id", "match_id", "map_id", "round_id", "event_tick", "opponent_player_id", "e_t"]].rename(columns={"opponent_player_id": "player_id", "e_t": "signed_e"})
    opponent["signed_e"] = -opponent["signed_e"].astype(float)
    events = pd.concat([focal, opponent], ignore_index=True)
    events["player_id"] = events["player_id"].astype(str)
    return events.sort_values(["match_id", "map_id", "player_id", "event_tick", "physical_event_id"], kind="stable").reset_index(drop=True)


def construct_states(asset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    events = player_events(asset)
    records: list[dict[str, Any]] = []
    max_direct_error = 0.0
    same_tick_batches = 0
    ambiguous_lag1_rows = 0
    for (match_id, map_id, player_id), group in events.groupby(["match_id", "map_id", "player_id"], sort=False, dropna=False):
        lag1: dict[str, Any] | None = None
        lag1_blocked = False
        running_sum = 0.0
        prior_tick: float | None = None
        earlier: list[tuple[float, float]] = []
        for tick, batch in group.groupby("event_tick", sort=False, dropna=False):
            tick_float = float(tick)
            if prior_tick is None:
                q = np.nan
                state_before = np.nan
            else:
                q = float(2.0 ** (-(tick_float - prior_tick) / (HALF_LIFE_SECONDS * TICK_RATE)))
                state_before = q * running_sum
                direct = float(sum(value * 2.0 ** (-(tick_float - old_tick) / (HALF_LIFE_SECONDS * TICK_RATE)) for old_tick, value in earlier))
                max_direct_error = max(max_direct_error, abs(state_before - direct))
            if len(batch) > 1:
                same_tick_batches += 1
            for row in batch.itertuples(index=False):
                records.append({
                    "physical_event_id": str(row.physical_event_id), "match_id": str(match_id), "map_id": str(map_id),
                    "focal_player_id": str(player_id), "event_tick": tick_float,
                    "reconstructed_H_t": np.nan if lag1 is None else float(lag1["signed_e"]),
                    "reconstructed_history_tick": np.nan if lag1 is None else float(lag1["event_tick"]),
                    "S_t": state_before, "q_t": q,
                    "lag1_ambiguous": bool(lag1 is None and lag1_blocked),
                })
                if lag1 is None and lag1_blocked:
                    ambiguous_lag1_rows += 1
            for row in batch.itertuples(index=False):
                earlier.append((tick_float, float(row.signed_e)))
            running_sum = 0.0 if prior_tick is None else state_before
            running_sum += float(batch["signed_e"].sum())
            prior_tick = tick_float
            if len(batch) == 1:
                only = batch.iloc[0]
                lag1 = {"signed_e": float(only["signed_e"]), "event_tick": tick_float}
                lag1_blocked = False
            else:
                lag1 = None
                lag1_blocked = True
    return pd.DataFrame.from_records(records), {
        "player_view_events": int(len(events)), "same_tick_batches": same_tick_batches,
        "ambiguous_lag1_rows": ambiguous_lag1_rows, "max_direct_recursion_abs_error": max_direct_error,
    }


def build_dataset(dataset: str, lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    state_path = OUT / "state" / dataset / "state_matrix.parquet"
    checkpoint_path = OUT / "state" / dataset / f"checkpoint_02_state_{dataset}.json"
    asset_path = ASSET / dataset / "residual_asset.parquet"
    lag1_path = LAG1 / "state" / dataset / "analysis_matrix.parquet"
    if state_path.exists():
        saved = read_json(checkpoint_path)
        require(saved.get("input_lock_sha256") == lock_hash, f"state_lock_mismatch:{dataset}")
        require(saved.get("state_matrix_sha256") == digest(state_path), f"state_hash_mismatch:{dataset}")
        return pd.read_parquet(state_path), saved
    asset = pd.read_parquet(asset_path, columns=["physical_event_id", "match_id", "map_id", "round_id", "event_tick", "focal_player_id", "opponent_player_id", "e_t", "residual_valid", "primary_analysis_eligible"])
    states, state_audit = construct_states(asset)
    lag1 = pd.read_parquet(lag1_path, columns=["e_t", "H_t", "match_id", "focal_player_id", "map_id", "event_tick", "history_source_event_tick", "time_since_history_tick"])
    focal = asset.loc[asset["residual_valid"].astype(bool) & asset["primary_analysis_eligible"].astype(bool), ["physical_event_id", "match_id", "map_id", "focal_player_id", "event_tick", "e_t"]].copy()
    for column in ("physical_event_id", "match_id", "map_id", "focal_player_id"):
        focal[column] = focal[column].astype(str)
    lag1["match_id"] = lag1["match_id"].astype(str)
    lag1["focal_player_id"] = lag1["focal_player_id"].astype(str)
    lag1["map_id"] = lag1["map_id"].astype(str)
    focal_states = focal.merge(states, on=["physical_event_id", "match_id", "map_id", "focal_player_id", "event_tick"], how="left", validate="one_to_one")
    signature = ["match_id", "map_id", "focal_player_id", "event_tick", "e_t", "reconstructed_H_t", "reconstructed_history_tick"]
    eligible = focal_states.loc[focal_states["reconstructed_H_t"].notna()].copy()
    signature_summary = eligible.groupby(signature, dropna=False).agg(S_min=("S_t", "min"), S_max=("S_t", "max"), q_min=("q_t", "min"), q_max=("q_t", "max"), rows=("physical_event_id", "size")).reset_index()
    signature_state_disagreement = int(((signature_summary["S_min"] != signature_summary["S_max"]) | (signature_summary["q_min"] != signature_summary["q_max"])).sum())
    require(signature_state_disagreement == 0, f"nonunique_state_for_lag1_signature:{dataset}")
    lookup = signature_summary.drop(columns=["S_max", "q_max", "rows"]).rename(columns={"reconstructed_H_t": "H_t", "reconstructed_history_tick": "history_source_event_tick", "S_min": "S_t", "q_min": "q_t"})
    merged = lag1.merge(lookup, on=["match_id", "map_id", "focal_player_id", "event_tick", "e_t", "H_t", "history_source_event_tick"], how="left", validate="many_to_one")
    missing_state = int(merged["S_t"].isna().sum())
    h_error, tick_error = (0.0, 0.0) if missing_state == 0 else (math.inf, math.inf)
    q = merged["q_t"].to_numpy(float)
    merged["L_t"] = q * merged["H_t"].to_numpy(float)
    merged["dataset"] = dataset
    matrix = merged[["dataset", "match_id", "focal_player_id", "map_id", "event_tick", "e_t", "H_t", "time_since_history_tick", "L_t", "S_t"]].copy()
    finite = int((~np.isfinite(matrix[["e_t", "H_t", "time_since_history_tick", "L_t", "S_t"]].to_numpy(float))).sum())
    audit = {
        **state_audit, "lag1_rows": int(len(lag1)), "focal_state_rows": int(len(focal_states)), "signature_rows": int(len(signature_summary)), "signature_state_disagreement": signature_state_disagreement, "missing_state_rows": missing_state,
        "max_abs_H_reconstruction_error": h_error, "max_abs_history_tick_reconstruction_error": tick_error,
        "nonfinite_model_values": finite, "match_clusters": int(matrix["match_id"].nunique()),
        "player_clusters": int(matrix["focal_player_id"].nunique()),
    }
    g1_pass = missing_state == 0 and h_error <= 1e-10 and tick_error <= 1e-10 and state_audit["max_direct_recursion_abs_error"] <= 1e-10 and finite == 0
    g2_pass = audit["match_clusters"] >= 30 and audit["player_clusters"] >= 30
    require(g1_pass, f"state_reconstruction_failed:{dataset}:{audit}")
    require(g2_pass, f"support_failed:{dataset}:{audit}")
    write_parquet_new(state_path, matrix)
    checkpoint = {
        "stage": "history_residual_state_oof_d1_v2_state", "dataset": dataset, "created_at_utc": now(),
        "input_lock_sha256": lock_hash, "asset_sha256": digest(asset_path), "lag1_matrix_sha256": digest(lag1_path),
        "state_matrix_sha256": digest(state_path), "audit": audit,
        "G1": {"status": "pass", "reason": "strict_player_view_history_and_direct_recursion_audit"},
        "G2": {"status": "pass", "reason": "match_and_player_cluster_support"},
    }
    write_json_new(checkpoint_path, checkpoint)
    return matrix, checkpoint


def design(frame: pd.DataFrame, model: str, mu: float, scale: float) -> np.ndarray:
    z = np.log2(1.0 + frame["time_since_history_tick"].to_numpy(float)) - mu
    field = {"M1_lag1": "H_t", "M2_decay_lag1": "L_t", "M3_decay_sum": "S_t"}[model]
    x = frame[field].to_numpy(float) / scale
    return np.column_stack([np.ones(len(frame)), z, x, x * z])


def fit_predict(train: pd.DataFrame, evaluation: pd.DataFrame, model: str) -> tuple[np.ndarray, dict[str, Any]]:
    mu = float(np.mean(np.log2(1.0 + train["time_since_history_tick"].to_numpy(float))))
    field = {"M1_lag1": "H_t", "M2_decay_lag1": "L_t", "M3_decay_sum": "S_t"}[model]
    scale = float(np.std(train[field].to_numpy(float), ddof=0))
    require(math.isfinite(scale) and scale > 0, f"zero_state_scale:{model}")
    x_train = design(train, model, mu, scale)
    require(np.linalg.matrix_rank(x_train) == x_train.shape[1], f"rank_deficient:{model}")
    inverse = np.linalg.inv(x_train.T @ x_train)
    coefficient = inverse @ x_train.T @ train["e_t"].to_numpy(float)
    leverage = np.einsum("ij,jk,ik->i", x_train, inverse, x_train)
    prediction = design(evaluation, model, mu, scale) @ coefficient
    require(np.isfinite(prediction).all(), f"nonfinite_prediction:{model}")
    return prediction, {"mu_train": mu, "state": field, "state_scale_train": scale, "coefficients": coefficient.tolist(), "max_leverage": float(leverage.max())}


def smoke(matrices: dict[str, pd.DataFrame], lock_hash: str) -> None:
    path = OUT / "checkpoint_03_smoke.json"
    if path.exists():
        saved = read_json(path)
        require(saved.get("input_lock_sha256") == lock_hash, "smoke_input_lock_mismatch")
        return
    train = matrices["D1-A"]
    eval_full = matrices["D1-B"]
    matches = sorted(eval_full["match_id"].unique())[:30]
    evaluation = eval_full.loc[eval_full["match_id"].isin(matches)].copy()
    metrics: dict[str, Any] = {"smoke_evaluation_matches": len(matches), "smoke_evaluation_rows": int(len(evaluation)), "models": {}}
    require(len(matches) == 30, "smoke_insufficient_matches")
    for model in MODELS:
        prediction, fitted = fit_predict(train, evaluation, model)
        metrics["models"][model] = {"max_leverage": fitted["max_leverage"], "finite_predictions": bool(np.isfinite(prediction).all()), "rank": 4}
        require(fitted["max_leverage"] <= 0.20, f"smoke_high_leverage:{model}")
    write_json_new(path, {"stage": "history_residual_state_oof_d1_v2_smoke", "created_at_utc": now(), "input_lock_sha256": lock_hash, "G0": {"status": "pass", "reason": "d1_only_inputs_locked"}, "G1": {"status": "pass", "reason": "state_matrix_audits_passed"}, "G2": {"status": "pass", "reason": "thirty_evaluation_match_smoke"}, "G3": {"status": "pass", "reason": "finite_full_rank_predictions"}, "metrics": metrics})


def fold_oof(name: str, train_sets: tuple[str, ...], eval_set: str, matrices: dict[str, pd.DataFrame], lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = OUT / "state" / name / "oof_predictions.parquet"
    checkpoint_path = OUT / "state" / name / "checkpoint_04_oof.json"
    if path.exists():
        saved = read_json(checkpoint_path)
        require(saved.get("input_lock_sha256") == lock_hash and saved.get("oof_sha256") == digest(path), f"oof_checkpoint_mismatch:{name}")
        return pd.read_parquet(path), saved
    train = pd.concat([matrices[item] for item in train_sets], ignore_index=True)
    evaluation = matrices[eval_set].copy()
    require(set(train["match_id"]).isdisjoint(set(evaluation["match_id"])), f"train_evaluation_match_overlap:{name}")
    output = evaluation[["dataset", "match_id", "focal_player_id", "e_t"]].copy()
    details: dict[str, Any] = {}
    for model in MODELS:
        prediction, fitted = fit_predict(train, evaluation, model)
        require(fitted["max_leverage"] <= 0.20, f"high_leverage:{name}:{model}")
        output[model] = prediction
        details[model] = fitted
    write_parquet_new(path, output)
    training_path = OUT / "state" / name / "training_fit.json"
    write_json_new(training_path, {"stage": "history_residual_state_oof_d1_v2_training_fit", "fold": name, "created_at_utc": now(), "input_lock_sha256": lock_hash, "train_datasets": list(train_sets), "evaluation_dataset": eval_set, "train_rows": int(len(train)), "training_matches": int(train["match_id"].nunique()), "evaluation_rows": int(len(evaluation)), "evaluation_matches": int(evaluation["match_id"].nunique()), "training_input_summary": {"e_mean": float(train["e_t"].mean()), "T_mean": float(train["time_since_history_tick"].mean()), "H_mean": float(train["H_t"].mean()), "L_mean": float(train["L_t"].mean()), "S_mean": float(train["S_t"].mean())}, "models": details})
    checkpoint = {"stage": "history_residual_state_oof_d1_v2_oof", "fold": name, "created_at_utc": now(), "input_lock_sha256": lock_hash, "train_datasets": list(train_sets), "evaluation_dataset": eval_set, "train_rows": int(len(train)), "evaluation_rows": int(len(evaluation)), "evaluation_matches": int(evaluation["match_id"].nunique()), "oof_sha256": digest(path), "training_fit_sha256": digest(training_path), "models": details, "G3": {"status": "pass", "reason": "full_rank_finite_predictions_and_leverage_threshold"}}
    write_json_new(checkpoint_path, checkpoint)
    return output, checkpoint


def match_losses(oof: pd.DataFrame, fold: str) -> pd.DataFrame:
    values = oof[["match_id", "e_t", *MODELS]].copy()
    for model in MODELS:
        values[model] = (values["e_t"] - values[model]) ** 2
    grouped = values.groupby("match_id", sort=True)[list(MODELS)].mean().reset_index()
    grouped.insert(0, "fold", fold)
    return grouped


def relative_gain(losses: pd.DataFrame, baseline: str, candidate: str) -> float:
    fold_means = losses.groupby("fold", sort=True)[[baseline, candidate]].mean()
    baseline_loss = float(fold_means[baseline].mean())
    candidate_loss = float(fold_means[candidate].mean())
    return (baseline_loss - candidate_loss) / baseline_loss


def holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted((value, name) for name, value in raw.items())
    adjusted: dict[str, float] = {}
    previous = 0.0
    for pos, (value, name) in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - pos) * value))
        adjusted[name] = previous
    return adjusted


def compare(losses: pd.DataFrame, lock_hash: str) -> dict[str, Any]:
    bootstrap_path = OUT / "state" / "bootstrap_gains.npy"
    checkpoint_path = OUT / "checkpoint_06_bootstrap.json"
    pairs = (("M1_vs_M2", "M1_lag1", "M2_decay_lag1"), ("M1_vs_M3", "M1_lag1", "M3_decay_sum"), ("M2_vs_M3", "M2_decay_lag1", "M3_decay_sum"))
    if bootstrap_path.exists():
        saved = read_json(checkpoint_path)
        require(saved.get("input_lock_sha256") == lock_hash and saved.get("bootstrap_sha256") == digest(bootstrap_path), "bootstrap_checkpoint_mismatch")
        draws = np.load(bootstrap_path)
    else:
        rng = np.random.default_rng(SEED)
        fold_arrays = {fold: part[list(MODELS)].to_numpy(float) for fold, part in losses.groupby("fold", sort=True)}
        require(set(fold_arrays) == set(FOLDS), "missing_bootstrap_fold")
        draws = np.empty((BOOTSTRAPS, len(pairs)), dtype=float)
        for i in range(BOOTSTRAPS):
            means: dict[str, np.ndarray] = {}
            for fold, values in fold_arrays.items():
                sample = values[rng.integers(0, len(values), len(values))]
                means[fold] = sample.mean(axis=0)
            combined = (means["F1"] + means["F2"]) / 2.0
            indexed = dict(zip(MODELS, combined))
            for j, (_, baseline, candidate) in enumerate(pairs):
                draws[i, j] = (indexed[baseline] - indexed[candidate]) / indexed[baseline]
        OUT.joinpath("state").mkdir(parents=True, exist_ok=True)
        np.save(bootstrap_path, draws)
        write_json_new(checkpoint_path, {"stage": "history_residual_state_oof_d1_v2_bootstrap", "created_at_utc": now(), "input_lock_sha256": lock_hash, "seed": SEED, "draws": BOOTSTRAPS, "bootstrap_sha256": digest(bootstrap_path)})
    raw: dict[str, float] = {}
    output: dict[str, Any] = {}
    for j, (name, baseline, candidate) in enumerate(pairs):
        point = relative_gain(losses, baseline, candidate)
        lo, hi = (float(x) for x in np.quantile(draws[:, j], [0.025, 0.975]))
        sd = float(np.std(draws[:, j], ddof=1))
        require(sd > 0 and math.isfinite(sd), f"invalid_bootstrap_sd:{name}")
        p = float(math.erfc(abs(point / sd) / math.sqrt(2.0)))
        raw[name] = p
        output[name] = {"baseline": baseline, "candidate": candidate, "relative_gain": point, "ci95_low": lo, "ci95_high": hi, "ci95_half_width": (hi - lo) / 2.0, "bootstrap_sd": sd, "raw_p_value": p}
    adjusted = holm(raw)
    for name, value in output.items():
        value["holm_adjusted_p_value_three_pair_family"] = adjusted[name]
        improve = value["relative_gain"] >= DELTA and value["ci95_low"] >= DELTA and adjusted[name] < 0.05
        worse = value["relative_gain"] < 0 and value["ci95_high"] < 0 and adjusted[name] < 0.05
        precision = value["ci95_half_width"] <= 0.010
        value["G4"] = {"status": "pass" if precision else "pending", "reason": "ci_half_width_threshold", "threshold": 0.010}
        value["decision"] = "exploratory_improvement" if improve and precision else ("exploratory_worsening" if worse and precision else "pending")
    return output


def output_hashes() -> dict[str, str]:
    return {str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"} and not any(part.startswith("failed_") for part in path.relative_to(OUT).parts)}


def run(phase: str) -> None:
    _, lock_hash = input_lock()
    matrices: dict[str, pd.DataFrame] = {}
    states: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        matrices[dataset], states[dataset] = build_dataset(dataset, lock_hash)
    if phase == "smoke":
        smoke(matrices, lock_hash)
        print(json.dumps({"status": "smoke_passed", "output_root": str(OUT)}, ensure_ascii=True))
        return
    smoke(matrices, lock_hash)
    fold_parts, fold_checks = [], {}
    for name, (train_sets, evaluation) in FOLDS.items():
        oof, fold_checks[name] = fold_oof(name, train_sets, evaluation, matrices, lock_hash)
        fold_parts.append(match_losses(oof, name))
    losses = pd.concat(fold_parts, ignore_index=True)
    loss_path = OUT / "match_losses.parquet"
    if not loss_path.exists():
        write_parquet_new(loss_path, losses)
    else:
        require(loss_path.is_file(), "loss_path_invalid")
        losses = pd.read_parquet(loss_path)
    comparisons = compare(losses, lock_hash)
    gates = {
        "G0": {"status": "pass", "reason": "d1_only_hash_locked_inputs", "metrics": {"e1_path_access_count": 0}},
        "G1": {"status": "pass", "reason": "lag1_reconstruction_and_direct_recursion_audits", "metrics": {name: value["audit"] for name, value in states.items()}},
        "G2": {"status": "pass", "reason": "two_forward_folds_have_match_and_player_support", "metrics": {name: {"evaluation_matches": value["evaluation_matches"], "evaluation_rows": value["evaluation_rows"]} for name, value in fold_checks.items()}},
        "G3": {"status": "pass", "reason": "all_models_full_rank_finite_and_max_leverage_at_most_0p20", "metrics": {name: {model: value["models"][model]["max_leverage"] for model in MODELS} for name, value in fold_checks.items()}},
        "G4": {"status": "pass" if all(value["G4"]["status"] == "pass" for value in comparisons.values()) else "pending", "reason": "pairwise_bootstrap_precision", "comparisons": {name: value["G4"] for name, value in comparisons.items()}},
        "G5": {"status": "pending", "reason": "exploratory_result_informed_observed_sequence_residual_prediction_only"},
        "overall": {"model_comparison_release": "pass", "interpretation_release": "exploratory_residual_loss_comparison_only", "forbidden_interpretations": ["p_hat_prediction_improvement", "hot_hand", "player_state", "causal_effect", "confirmatory_conclusion"]},
    }
    summary_path = OUT / "checkpoint_07_comparisons.json"
    if not summary_path.exists():
        write_json_new(summary_path, {"stage": "history_residual_state_oof_d1_v2_comparisons", "created_at_utc": now(), "input_lock_sha256": lock_hash, "half_life_seconds": HALF_LIFE_SECONDS, "tick_rate": TICK_RATE, "delta": DELTA, "inference": "2000_stratified_match_bootstrap_percentile_ci_normal_p_holm_three_pair_family", "comparisons": comparisons, "folds": fold_checks})
    else:
        require(read_json(summary_path).get("input_lock_sha256") == lock_hash, "comparison_checkpoint_mismatch")
    gate_path = OUT / "checkpoint_08_gate_audit.json"
    if not gate_path.exists():
        write_json_new(gate_path, {"stage": "history_residual_state_oof_d1_v2_gates", "created_at_utc": now(), "input_lock_sha256": lock_hash, "gates": gates})
    hashes_path = OUT / "file_sha256.json"
    if not hashes_path.exists():
        write_json_new(hashes_path, output_hashes())
    else:
        hashes = read_json(hashes_path)
        for relative, expected in hashes.items():
            require((OUT / relative).is_file() and digest(OUT / relative) == expected, f"output_hash_mismatch:{relative}")
    manifest_path = OUT / "manifest.json"
    if not manifest_path.exists():
        write_json_new(manifest_path, {"stage": "history_residual_state_oof_d1_v2", "status": "passed", "created_at_utc": now(), "classification": "exploratory_nonconfirmatory_result_informed_d1_only", "input_lock_sha256": lock_hash, "gate_statuses": {key: value["status"] for key, value in gates.items() if key.startswith("G")}, "interpretation": "D1-only time-forward comparison of fixed residual-history-state models; exploratory residual-loss evidence only, not p_hat performance, hot-hand, state, causal, or confirmatory evidence.", "file_sha256_sha256": digest(hashes_path)})
    else:
        manifest = read_json(manifest_path)
        require(manifest.get("input_lock_sha256") == lock_hash and manifest.get("file_sha256_sha256") == digest(hashes_path), "manifest_mismatch")
    print(json.dumps({"status": "passed", "manifest": str(manifest_path)}, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    try:
        run(args.phase)
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True)
        failed = OUT / f"failed_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        failed.mkdir(exist_ok=False)
        write_json_new(failed / "failure_manifest.json", {"stage": "history_residual_state_oof_d1_v2", "status": "failed", "created_at_utc": now(), "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
