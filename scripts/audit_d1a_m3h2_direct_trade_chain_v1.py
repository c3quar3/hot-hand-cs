"""Audit whether the selected D1-A M3(2 s) state coefficient concentrates in 5 s direct-trade chains."""

from __future__ import annotations

import argparse
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

import explore_d1a_dynamic_history_half_life_v1 as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "d1a_m3h2_direct_trade_chain_v1"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "76_D1A直接补枪链_M3_2秒状态诊断执行锁_v1_2026-08-29.md"
V3 = ROOT / "results" / "d1a_dynamic_history_half_life_v3"
V4 = ROOT / "results" / "d1a_dynamic_history_half_life_v4"
V5 = ROOT / "results" / "d1a_dynamic_history_half_life_v5_completion"
FOLDS = (("F1", 101, 150, 151, 200), ("F2", 101, 200, 201, 250), ("F3", 101, 250, 251, 300))
TRADE_WINDOW_TICKS = 5 * 64
MIN_TRAIN_ROWS, MIN_TRAIN_MATCHES = 50, 10
MIN_EVAL_ROWS, MIN_EVAL_MATCHES = 20, 5


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    base.require(not path.exists(), f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    base.require(not path.exists(), f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def inputs() -> list[Path]:
    paths = [
        PROTOCOL,
        Path(__file__).resolve(),
        V3 / "state" / "D1-A" / "model_matrix.parquet",
        V3 / "state" / "D1-A" / "checkpoint_02_matrix_D1-A.json",
        V4 / "checkpoint_05_selection.json",
        V5 / "checkpoint_03_completion_summary.json",
        V5 / "manifest.json",
        V5 / "file_sha256.json",
        base.HIST,
        base.ASSET / "manifest.json",
        base.ASSET / "file_sha256.json",
        base.ASSET / "D1-A" / "residual_asset.parquet",
        base.ASSET / "D1-A" / "checkpoint.json",
    ]
    for name, *_ in FOLDS:
        paths.append(V4 / "state" / name / "training_fit.json")
    return paths


def input_lock() -> tuple[dict[str, Any], str]:
    path = OUT / "checkpoint_01_input_lock.json"
    asset_audit = base.verify_tree(base.ASSET)
    v5_audit = base.verify_tree(V5)
    base.require(not asset_audit["mismatches"] and asset_audit["manifest_file_sha256_matches"], "conditional_asset_hash_chain_mismatch")
    base.require(not v5_audit["mismatches"] and v5_audit["manifest_file_sha256_matches"], "v5_completion_hash_chain_mismatch")
    v5_manifest = read_json(V5 / "manifest.json")
    v5_summary = read_json(V5 / "checkpoint_03_completion_summary.json")
    selected = v5_manifest.get("selected_model", {})
    base.require(v5_manifest.get("status") == "passed" and v5_manifest.get("v4_core_hash_status") == "passed", "v5_completion_status_failure")
    base.require(selected.get("name") == "M3_h2" and v5_summary.get("selected_model", {}).get("name") == "M3_h2", "selected_model_not_m3_h2")
    records = {str(item): base.digest(item) for item in inputs()}
    if path.exists():
        saved = read_json(path)
        base.require(saved.get("inputs") == records, "input_lock_source_hash_changed")
        return saved, base.digest(path)
    value = {
        "stage": "d1a_m3h2_direct_trade_chain_v1_input_lock",
        "created_at_utc": now(),
        "classification": "exploratory_nonconfirmatory_result_informed_d1a_only",
        "inputs": records,
        "conditional_asset_hash_audit": asset_audit,
        "v5_completion_hash_audit": v5_audit,
        "selected_model": selected,
        "state_half_life_seconds": 2,
        "direct_trade_window_seconds": 5,
        "direct_trade_window_ticks": TRADE_WINDOW_TICKS,
        "folds": [{"name": n, "train_queue_order": [a, b], "evaluation_queue_order": [c, d]} for n, a, b, c, d in FOLDS],
        "support_thresholds": {"training_rows": MIN_TRAIN_ROWS, "training_matches": MIN_TRAIN_MATCHES, "evaluation_rows": MIN_EVAL_ROWS, "evaluation_matches": MIN_EVAL_MATCHES},
        "external_partition_access_count": 0,
    }
    write_json(path, value)
    return value, base.digest(path)


def labelled_matrix(lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_path = OUT / "d1a_direct_trade_labels.parquet"
    checkpoint_path = OUT / "checkpoint_02_label_audit.json"
    if output_path.exists():
        checkpoint = read_json(checkpoint_path)
        base.require(checkpoint.get("input_lock_sha256") == lock_hash and checkpoint.get("labels_sha256") == base.digest(output_path), "label_checkpoint_mismatch")
        return pd.read_parquet(output_path), checkpoint

    matrix = pd.read_parquet(V3 / "state" / "D1-A" / "model_matrix.parquet")
    required_matrix = {"duel_id", "match_id", "queue_order", "Y_t", "H_t", "S_h2", "z_time", *base.NUMERIC, *base.CATEGORICAL}
    base.require(required_matrix.issubset(matrix.columns) and not matrix["duel_id"].duplicated().any(), "matrix_contract_failure")
    raw = base.raw_candidate("D1-A")
    required_raw = {"duel_id", "match_id", "map_id", "round_id", "event_tick", "focal_steamid", "opponent_steamid", "focal_wins", "focal_post_buy_team", "opponent_post_buy_team"}
    base.require(required_raw.issubset(raw.columns), "raw_trade_label_contract_failure")
    asset = pd.read_parquet(base.ASSET / "D1-A" / "residual_asset.parquet")
    asset = asset.loc[asset["residual_valid"].astype(bool) & asset["primary_analysis_eligible"].astype(bool)].copy()
    asset_fields = ["duel_id", "physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id", "opponent_player_id", "Y_t"]
    base.require(set(asset_fields).issubset(asset.columns) and not asset["duel_id"].duplicated().any() and not asset["physical_event_id"].duplicated().any(), "asset_trade_label_contract_failure")

    raw = raw[list(required_raw)].copy()
    event = raw.merge(asset[asset_fields], on="duel_id", how="inner", validate="one_to_one", suffixes=("_raw", "_asset"))
    base.require(len(event) == len(asset), "raw_asset_trade_join_loss")
    for field in ("match_id_raw", "map_id_raw", "focal_steamid", "opponent_steamid", "focal_player_id", "opponent_player_id", "physical_event_id"):
        event[field] = event[field].astype(str)
    base.require(np.array_equal(event["focal_wins"].to_numpy(int), event["Y_t"].to_numpy(int)), "raw_asset_outcome_mismatch")
    base.require(np.array_equal(event["match_id_raw"].to_numpy(str), event["match_id_asset"].astype(str).to_numpy()) and np.array_equal(event["map_id_raw"].to_numpy(str), event["map_id_asset"].astype(str).to_numpy()), "raw_asset_location_mismatch")
    base.require(np.array_equal(event["event_tick_raw"].to_numpy(float), event["event_tick_asset"].to_numpy(float)), "raw_asset_tick_mismatch")
    base.require(np.array_equal(event["focal_steamid"].to_numpy(str), event["focal_player_id"].to_numpy(str)) and np.array_equal(event["opponent_steamid"].to_numpy(str), event["opponent_player_id"].to_numpy(str)), "raw_asset_player_direction_mismatch")

    states, state_audit = base.construct_states(asset)
    for field in ("physical_event_id", "match_id", "map_id", "focal_player_id"):
        states[field] = states[field].astype(str)
    current = event.merge(
        states[["physical_event_id", "focal_player_id", "history_event_tick", "T_seconds"]],
        on=["physical_event_id", "focal_player_id"], how="inner", validate="one_to_one",
    )
    current = current.loc[current["history_event_tick"].notna()].copy()
    base.require(not current.duplicated("duel_id").any(), "nonunique_current_state_join")
    prior_views = states[["physical_event_id", "match_id", "map_id", "focal_player_id", "event_tick"]].rename(columns={"physical_event_id": "previous_physical_event_id", "match_id": "match_id_raw", "map_id": "map_id_raw", "event_tick": "history_event_tick"})
    base.require(not prior_views.duplicated(["match_id_raw", "map_id_raw", "focal_player_id", "history_event_tick"]).any(), "nonunique_prior_participant_event")
    current = current.merge(prior_views, on=["match_id_raw", "map_id_raw", "focal_player_id", "history_event_tick"], how="left", validate="many_to_one")
    base.require(current["previous_physical_event_id"].notna().all(), "history_tick_does_not_resolve_to_unique_event")

    previous = event[["physical_event_id", "round_id", "focal_steamid", "opponent_steamid", "focal_wins", "focal_post_buy_team", "opponent_post_buy_team"]].rename(columns={
        "physical_event_id": "previous_physical_event_id", "round_id": "previous_round_id", "focal_steamid": "previous_focal_player", "opponent_steamid": "previous_opponent_player", "focal_wins": "previous_focal_wins", "focal_post_buy_team": "previous_focal_team", "opponent_post_buy_team": "previous_opponent_team",
    })
    current = current.merge(previous, on="previous_physical_event_id", how="left", validate="many_to_one")
    base.require(current[["previous_round_id", "previous_focal_player", "previous_opponent_player", "previous_focal_wins", "previous_focal_team", "previous_opponent_team"]].notna().all().all(), "previous_raw_event_join_failure")

    previous_as_focal = current["previous_focal_player"].eq(current["focal_player_id"])
    previous_as_opponent = current["previous_opponent_player"].eq(current["focal_player_id"])
    base.require((previous_as_focal ^ previous_as_opponent).all(), "previous_player_view_direction_failure")
    previous_win = (previous_as_focal & current["previous_focal_wins"].eq(1)) | (previous_as_opponent & current["previous_focal_wins"].eq(0))
    previous_victim = np.where(previous_as_focal, current["previous_opponent_player"], current["previous_focal_player"])
    previous_victim_team = np.where(previous_as_focal, current["previous_opponent_team"], current["previous_focal_team"])
    current["previous_victim_player"] = previous_victim
    current["previous_victim_team"] = previous_victim_team
    current["interval_ticks"] = current["event_tick_raw"].to_numpy(float) - current["history_event_tick"].to_numpy(float)
    interval_error = float(np.max(np.abs(current["interval_ticks"].to_numpy(float) / 64.0 - current["T_seconds"].to_numpy(float))))
    base.require(interval_error <= 1e-10 and (current["interval_ticks"] > 0).all(), "strict_interval_reconstruction_failure")
    same_round = current["round_id"].astype(str).eq(current["previous_round_id"].astype(str))
    current_loss = current["focal_wins"].eq(0)
    same_team = current["opponent_post_buy_team"].eq(current["previous_victim_team"])
    different_player = current["opponent_steamid"].ne(current["previous_victim_player"])
    within_window = current["interval_ticks"].le(TRADE_WINDOW_TICKS)
    current["direct_trade_5s"] = (current_loss & previous_win & same_round & same_team & different_player & within_window).astype(int)
    current["predicate_current_loss"] = current_loss.astype(int)
    current["predicate_previous_focal_win"] = previous_win.astype(int)
    current["predicate_same_round"] = same_round.astype(int)
    current["predicate_same_team"] = same_team.astype(int)
    current["predicate_different_player"] = different_player.astype(int)
    current["predicate_within_5s"] = within_window.astype(int)

    labels = matrix.merge(current[["duel_id", "history_event_tick", "interval_ticks", "previous_physical_event_id", "direct_trade_5s", "predicate_current_loss", "predicate_previous_focal_win", "predicate_same_round", "predicate_same_team", "predicate_different_player", "predicate_within_5s"]], on="duel_id", how="inner", validate="one_to_one")
    base.require(len(labels) == len(matrix) and labels["history_event_tick"].notna().all(), "matrix_label_join_loss")
    base.require(np.isfinite(labels[["S_h2", "z_time", "interval_ticks"]].to_numpy(float)).all(), "nonfinite_labelled_matrix")
    output_columns = ["duel_id", "match_id", "queue_order", "Y_t", "H_t", "S_h2", "z_time", "history_event_tick", "interval_ticks", "previous_physical_event_id", "direct_trade_5s", "predicate_current_loss", "predicate_previous_focal_win", "predicate_same_round", "predicate_same_team", "predicate_different_player", "predicate_within_5s", *base.NUMERIC, *base.CATEGORICAL]
    labels = labels[output_columns].copy()
    write_parquet(output_path, labels)
    predicate_counts = {field: int(labels[field].sum()) for field in output_columns if field.startswith("predicate_")}
    checkpoint = {
        "stage": "d1a_m3h2_direct_trade_chain_v1_label_audit", "created_at_utc": now(), "input_lock_sha256": lock_hash,
        "labels_sha256": base.digest(output_path),
        "G1": {"status": "pass", "reason": "strict_unique_prior_participant_event_and_same_round_team_trade_contract", "metrics": {"labelled_rows": int(len(labels)), "labelled_matches": int(labels["match_id"].nunique()), "direct_trade_rows": int(labels["direct_trade_5s"].sum()), "direct_trade_matches": int(labels.loc[labels["direct_trade_5s"].eq(1), "match_id"].nunique()), "state_interval_max_abs_error_seconds": interval_error, "state_audit": state_audit, "predicate_counts": predicate_counts}},
    }
    write_json(checkpoint_path, checkpoint)
    return labels, checkpoint


def fold_support(labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    gates: dict[str, Any] = {}
    for name, train_start, train_end, evaluation_start, evaluation_end in FOLDS:
        for role, start, end, row_threshold, match_threshold in (("training", train_start, train_end, MIN_TRAIN_ROWS, MIN_TRAIN_MATCHES), ("evaluation", evaluation_start, evaluation_end, MIN_EVAL_ROWS, MIN_EVAL_MATCHES)):
            subset = labels.loc[labels["queue_order"].between(start, end)]
            trade = subset.loc[subset["direct_trade_5s"].eq(1)]
            record = {"fold": name, "role": role, "queue_order_start": start, "queue_order_end": end, "all_rows": int(len(subset)), "all_matches": int(subset["match_id"].nunique()), "direct_trade_rows": int(len(trade)), "direct_trade_matches": int(trade["match_id"].nunique()), "required_rows": row_threshold, "required_matches": match_threshold}
            record["support_pass"] = bool(record["direct_trade_rows"] >= row_threshold and record["direct_trade_matches"] >= match_threshold)
            records.append(record)
            gates[f"{name}_{role}"] = record
    support = pd.DataFrame(records)
    base.require(support["support_pass"].all(), f"direct_trade_support_failure:{support.loc[~support['support_pass'], ['fold', 'role', 'direct_trade_rows', 'direct_trade_matches']].to_dict(orient='records')}")
    return support, {"status": "pass", "reason": "predeclared_direct_trade_row_and_match_support", "metrics": gates}


def fit_l2(x: sparse.csr_matrix, y: np.ndarray, name: str) -> LogisticRegression:
    base.require(x.shape[0] > x.shape[1] and x.shape[1] > 0, f"design_dimension_failure:{name}")
    model = LogisticRegression(C=base.C, penalty="l2", solver="lbfgs", max_iter=base.MAX_ITER, tol=base.TOL, random_state=base.SEED)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y.astype(int))
    base.require(not any(issubclass(item.category, ConvergenceWarning) for item in caught), f"convergence_warning:{name}")
    base.require(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all(), f"nonfinite_coefficients:{name}")
    return model


def base_fit(train: pd.DataFrame, name: str) -> tuple[base.Design, LogisticRegression, sparse.csr_matrix]:
    design = base.Design("S_h2").fit(train)
    x_train, unknown = design.transform(train)
    base.require(not any(unknown.values()), f"unknown_training_categories:{name}:{unknown}")
    return design, fit_l2(x_train, train["Y_t"].to_numpy(int), name), x_train


def parent_state_coefficient(fold: str) -> float:
    saved = read_json(V4 / "state" / fold / "training_fit.json")
    return float(saved["models"]["M3_h2"]["state_coefficient"])


def fit_fold(fold: str, labels: pd.DataFrame, train_start: int, train_end: int, lock_hash: str) -> dict[str, Any]:
    path = OUT / "state" / fold / "checkpoint_04_fit.json"
    if path.exists():
        saved = read_json(path)
        base.require(saved.get("input_lock_sha256") == lock_hash, f"fold_checkpoint_input_lock_mismatch:{fold}")
        return saved
    train = labels.loc[labels["queue_order"].between(train_start, train_end)].copy()
    design, fitted, x_train = base_fit(train, f"{fold}_reproduction")
    state_index = design.numeric.index("S_h2")
    reproduction = float(fitted.coef_[0][state_index])
    expected = parent_state_coefficient(fold)
    base.require(abs(reproduction - expected) <= 1e-8, f"v4_state_coefficient_reproduction_failure:{fold}:{reproduction}:{expected}")

    state_standardized = (train["S_h2"].to_numpy(float) - design.means["S_h2"]) / design.sds["S_h2"]
    direct = train["direct_trade_5s"].to_numpy(float)
    x_interaction = sparse.hstack([x_train, sparse.csr_matrix(np.column_stack([direct, direct * state_standardized]))], format="csr")
    interaction = fit_l2(x_interaction, train["Y_t"].to_numpy(int), f"{fold}_trade_interaction")
    b_nontrade = float(interaction.coef_[0][state_index])
    b_interaction = float(interaction.coef_[0][-1])
    b_trade = b_nontrade + b_interaction

    leave_train = train.loc[train["direct_trade_5s"].eq(0)].copy()
    leave_design, leave_model, leave_x = base_fit(leave_train, f"{fold}_leave_trade_out")
    leave_coefficient = float(leave_model.coef_[0][leave_design.numeric.index("S_h2")])
    values = {
        "stage": "d1a_m3h2_direct_trade_chain_v1_fold_fit", "created_at_utc": now(), "input_lock_sha256": lock_hash,
        "fold": fold, "train_queue_order": [train_start, train_end], "train_rows": int(len(train)), "train_matches": int(train["match_id"].nunique()), "direct_trade_rows": int(train["direct_trade_5s"].sum()), "direct_trade_matches": int(train.loc[train["direct_trade_5s"].eq(1), "match_id"].nunique()),
        "reproduction_state_coefficient": reproduction, "v4_saved_state_coefficient": expected, "reproduction_abs_error": abs(reproduction - expected),
        "interaction_nontrade_state_coefficient": b_nontrade, "interaction_trade_state_coefficient": b_trade, "interaction_coefficient_difference": b_interaction, "interaction_direct_trade_intercept_shift": float(interaction.coef_[0][-2]),
        "leave_trade_out_rows": int(len(leave_train)), "leave_trade_out_state_coefficient": leave_coefficient, "leave_trade_out_minus_reproduction": leave_coefficient - reproduction,
        "reproduction_design_shape": [int(x_train.shape[0]), int(x_train.shape[1])], "interaction_design_shape": [int(x_interaction.shape[0]), int(x_interaction.shape[1])], "leave_trade_out_design_shape": [int(leave_x.shape[0]), int(leave_x.shape[1])], "reproduction_n_iter": int(fitted.n_iter_[0]), "interaction_n_iter": int(interaction.n_iter_[0]), "leave_trade_out_n_iter": int(leave_model.n_iter_[0]),
    }
    write_json(path, values)
    return values


def output_hashes() -> dict[str, str]:
    return {str(item.relative_to(OUT)): base.digest(item) for item in sorted(OUT.rglob("*")) if item.is_file() and item.name not in {"file_sha256.json", "manifest.json"} and not any(part.startswith("failed_") for part in item.relative_to(OUT).parts)}


def run(phase: str) -> None:
    _, lock_hash = input_lock()
    labels, label_gate = labelled_matrix(lock_hash)
    support, support_gate = fold_support(labels)
    support_path = OUT / "fold_trade_support.parquet"
    if not support_path.exists():
        write_parquet(support_path, support)
    else:
        base.require(base.digest(support_path) == read_json(OUT / "checkpoint_03_smoke.json").get("fold_trade_support_sha256"), "support_output_hash_mismatch")
    smoke_path = OUT / "checkpoint_03_smoke.json"
    if not smoke_path.exists():
        write_json(smoke_path, {"stage": "d1a_m3h2_direct_trade_chain_v1_smoke", "created_at_utc": now(), "input_lock_sha256": lock_hash, "fold_trade_support_sha256": base.digest(support_path), "G0": {"status": "pass", "reason": "v5_completed_m3h2_and_hash_locked_d1a_inputs_only"}, "G1": label_gate["G1"], "G2": {"status": "pass", "folds": [{"name": n, "train": [a, b], "evaluation": [c, d]} for n, a, b, c, d in FOLDS]}, "G3": support_gate})
    if phase == "smoke":
        print(json.dumps({"status": "smoke_passed", "output_root": str(OUT)}, ensure_ascii=True))
        return

    fits = []
    for name, train_start, train_end, _, _ in FOLDS:
        fits.append(fit_fold(name, labels, train_start, train_end, lock_hash))
    coefficient_rows = []
    for item in fits:
        for model, coefficient in (("reproduction", item["reproduction_state_coefficient"]), ("interaction_nontrade", item["interaction_nontrade_state_coefficient"]), ("interaction_trade", item["interaction_trade_state_coefficient"]), ("leave_trade_out", item["leave_trade_out_state_coefficient"])):
            coefficient_rows.append({"fold": item["fold"], "model": model, "state_coefficient": coefficient, "direct_trade_rows": item["direct_trade_rows"], "direct_trade_matches": item["direct_trade_matches"]})
    coefficients = pd.DataFrame(coefficient_rows)
    coefficient_path = OUT / "fold_coefficients.parquet"
    if not coefficient_path.exists():
        write_parquet(coefficient_path, coefficients)
    summary = coefficients.groupby("model", sort=True)["state_coefficient"].agg(["mean", "std", "min", "max"]).reset_index().rename(columns={"std": "fold_sd", "min": "fold_min", "max": "fold_max", "mean": "fold_mean"})
    interaction_table = pd.DataFrame([{key: item[key] for key in ("fold", "interaction_nontrade_state_coefficient", "interaction_trade_state_coefficient", "interaction_coefficient_difference", "leave_trade_out_state_coefficient", "leave_trade_out_minus_reproduction", "reproduction_abs_error")} for item in fits])
    summary_path = OUT / "coefficient_summary.parquet"
    interaction_path = OUT / "interaction_diagnostic.parquet"
    if not summary_path.exists():
        write_parquet(summary_path, summary)
    if not interaction_path.exists():
        write_parquet(interaction_path, interaction_table)
    gate = {"G0": {"status": "pass", "reason": "v5_completed_m3h2_and_hash_locked_d1a_inputs_only"}, "G1": label_gate["G1"], "G2": {"status": "pass", "folds": [item["fold"] for item in fits]}, "G3": support_gate, "G4": {"status": "pass", "reason": "all_l2_fits_converged_and_reproduced_v4_m3h2_coefficients", "max_reproduction_abs_error": float(max(item["reproduction_abs_error"] for item in fits))}, "G5": {"status": "pass", "reason": "complete_label_support_and_three_coefficient_diagnostics"}}
    gates_path = OUT / "checkpoint_05_gate_audit.json"
    if not gates_path.exists():
        write_json(gates_path, {"stage": "d1a_m3h2_direct_trade_chain_v1_gates", "created_at_utc": now(), "input_lock_sha256": lock_hash, "gates": gate})
    hashes_path = OUT / "file_sha256.json"
    if not hashes_path.exists():
        write_json(hashes_path, output_hashes())
    else:
        for relative, expected in read_json(hashes_path).items():
            base.require((OUT / relative).is_file() and base.digest(OUT / relative) == expected, f"output_hash_mismatch:{relative}")
    manifest_path = OUT / "manifest.json"
    if not manifest_path.exists():
        write_json(manifest_path, {"stage": "d1a_m3h2_direct_trade_chain_v1", "status": "passed", "created_at_utc": now(), "classification": "exploratory_nonconfirmatory_result_informed_d1a_only", "input_lock_sha256": lock_hash, "state_model": "M3_h2", "direct_trade_window_seconds": 5, "gate_statuses": {name: value["status"] for name, value in gate.items()}, "interpretation": "A five-second direct-trade-chain label diagnostic for the already selected D1-A M3(2 s) model. It is descriptive only and does not establish that M3 captures trades, nor any hot-hand, causal, state, prediction, or confirmatory claim.", "file_sha256_sha256": base.digest(hashes_path)})
    print(json.dumps({"status": "passed", "manifest": str(manifest_path)}, ensure_ascii=True))


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
        (failed / "error.txt").write_text("".join(traceback.format_exception(error)), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
