"""Versioned G1 repair for the D1-A M3(2 s) five-second direct-trade-chain diagnostic."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import audit_d1a_m3h2_direct_trade_chain_v1 as prior
import explore_d1a_dynamic_history_half_life_v1 as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "d1a_m3h2_direct_trade_chain_v2"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "77_D1A直接补枪链_M3_2秒状态诊断执行锁_v2_2026-08-29.md"
V1_SCRIPT = ROOT / "scripts" / "audit_d1a_m3h2_direct_trade_chain_v1.py"


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
    return [
        PROTOCOL, Path(__file__).resolve(), V1_SCRIPT,
        prior.V3 / "state" / "D1-A" / "model_matrix.parquet", prior.V3 / "state" / "D1-A" / "checkpoint_02_matrix_D1-A.json",
        prior.V4 / "checkpoint_05_selection.json", prior.V5 / "checkpoint_03_completion_summary.json", prior.V5 / "manifest.json", prior.V5 / "file_sha256.json",
        base.HIST, base.ASSET / "manifest.json", base.ASSET / "file_sha256.json", base.ASSET / "D1-A" / "residual_asset.parquet", base.ASSET / "D1-A" / "checkpoint.json",
        *[prior.V4 / "state" / name / "training_fit.json" for name, *_ in prior.FOLDS],
    ]


def input_lock() -> tuple[dict[str, Any], str]:
    path = OUT / "checkpoint_01_input_lock.json"
    asset_audit, v5_audit = base.verify_tree(base.ASSET), base.verify_tree(prior.V5)
    base.require(not asset_audit["mismatches"] and asset_audit["manifest_file_sha256_matches"], "conditional_asset_hash_chain_mismatch")
    base.require(not v5_audit["mismatches"] and v5_audit["manifest_file_sha256_matches"], "v5_completion_hash_chain_mismatch")
    manifest, summary = read_json(prior.V5 / "manifest.json"), read_json(prior.V5 / "checkpoint_03_completion_summary.json")
    selected = manifest.get("selected_model", {})
    base.require(manifest.get("status") == "passed" and manifest.get("v4_core_hash_status") == "passed" and selected.get("name") == "M3_h2" and summary.get("selected_model", {}).get("name") == "M3_h2", "v5_selected_m3h2_contract_failure")
    records = {str(item): base.digest(item) for item in inputs()}
    if path.exists():
        saved = read_json(path)
        base.require(saved.get("inputs") == records, "input_lock_source_hash_changed")
        return saved, base.digest(path)
    value = {"stage": "d1a_m3h2_direct_trade_chain_v2_input_lock", "created_at_utc": now(), "classification": "exploratory_nonconfirmatory_result_informed_d1a_only", "inputs": records, "conditional_asset_hash_audit": asset_audit, "v5_completion_hash_audit": v5_audit, "selected_model": selected, "state_half_life_seconds": 2, "direct_trade_window_seconds": 5, "direct_trade_window_ticks": prior.TRADE_WINDOW_TICKS, "folds": [{"name": n, "train_queue_order": [a, b], "evaluation_queue_order": [c, d]} for n, a, b, c, d in prior.FOLDS], "support_thresholds": {"training_rows": prior.MIN_TRAIN_ROWS, "training_matches": prior.MIN_TRAIN_MATCHES, "evaluation_rows": prior.MIN_EVAL_ROWS, "evaluation_matches": prior.MIN_EVAL_MATCHES}, "external_partition_access_count": 0, "v1_failure_reference": "G1_global_prior_tick_uniqueness_check_only"}
    write_json(path, value)
    return value, base.digest(path)


def labelled_matrix(lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_path, checkpoint_path = OUT / "d1a_direct_trade_labels.parquet", OUT / "checkpoint_02_label_audit.json"
    if output_path.exists():
        checkpoint = read_json(checkpoint_path)
        base.require(checkpoint.get("input_lock_sha256") == lock_hash and checkpoint.get("labels_sha256") == base.digest(output_path), "label_checkpoint_mismatch")
        return pd.read_parquet(output_path), checkpoint
    matrix = pd.read_parquet(prior.V3 / "state" / "D1-A" / "model_matrix.parquet")
    required_matrix = {"duel_id", "match_id", "queue_order", "Y_t", "H_t", "S_h2", "z_time", *base.NUMERIC, *base.CATEGORICAL}
    base.require(required_matrix.issubset(matrix.columns) and not matrix["duel_id"].duplicated().any(), "matrix_contract_failure")
    raw = base.raw_candidate("D1-A")
    fields = ["duel_id", "match_id", "map_id", "round_id", "event_tick", "focal_steamid", "opponent_steamid", "focal_wins", "focal_post_buy_team", "opponent_post_buy_team"]
    base.require(set(fields).issubset(raw.columns), "raw_trade_label_contract_failure")
    asset = pd.read_parquet(base.ASSET / "D1-A" / "residual_asset.parquet")
    asset = asset.loc[asset["residual_valid"].astype(bool) & asset["primary_analysis_eligible"].astype(bool)].copy()
    asset_fields = ["duel_id", "physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id", "opponent_player_id", "Y_t"]
    base.require(set(asset_fields).issubset(asset.columns) and not asset["duel_id"].duplicated().any() and not asset["physical_event_id"].duplicated().any(), "asset_trade_label_contract_failure")
    event = raw[fields].merge(asset[asset_fields], on="duel_id", how="inner", validate="one_to_one", suffixes=("_raw", "_asset"))
    base.require(len(event) == len(asset), "raw_asset_trade_join_loss")
    for field in ("match_id_raw", "map_id_raw", "focal_steamid", "opponent_steamid", "focal_player_id", "opponent_player_id", "physical_event_id"):
        event[field] = event[field].astype(str)
    base.require(np.array_equal(event["focal_wins"].to_numpy(int), event["Y_t"].to_numpy(int)), "raw_asset_outcome_mismatch")
    base.require(np.array_equal(event["match_id_raw"].to_numpy(str), event["match_id_asset"].astype(str).to_numpy()) and np.array_equal(event["map_id_raw"].to_numpy(str), event["map_id_asset"].astype(str).to_numpy()) and np.array_equal(event["event_tick_raw"].to_numpy(float), event["event_tick_asset"].to_numpy(float)), "raw_asset_event_contract_mismatch")
    base.require(np.array_equal(event["focal_steamid"].to_numpy(str), event["focal_player_id"].to_numpy(str)) and np.array_equal(event["opponent_steamid"].to_numpy(str), event["opponent_player_id"].to_numpy(str)), "raw_asset_player_direction_mismatch")
    states, state_audit = base.construct_states(asset)
    for field in ("physical_event_id", "match_id", "map_id", "focal_player_id"):
        states[field] = states[field].astype(str)
    current = event.merge(states[["physical_event_id", "focal_player_id", "history_event_tick", "T_seconds"]], on=["physical_event_id", "focal_player_id"], how="inner", validate="one_to_one")
    current = current.loc[current["history_event_tick"].notna()].copy()
    base.require(not current.duplicated("duel_id").any(), "nonunique_current_state_join")
    key_fields = ["match_id_raw", "map_id_raw", "focal_player_id", "history_event_tick"]
    target_keys = current[key_fields].drop_duplicates()
    prior_views = states[["physical_event_id", "match_id", "map_id", "focal_player_id", "event_tick"]].rename(columns={"physical_event_id": "previous_physical_event_id", "match_id": "match_id_raw", "map_id": "map_id_raw", "event_tick": "history_event_tick"})
    prior_views = prior_views.merge(target_keys, on=key_fields, how="inner", validate="many_to_many")
    base.require(not prior_views.duplicated(key_fields).any(), "nonunique_referenced_prior_participant_event")
    current = current.merge(prior_views, on=key_fields, how="left", validate="many_to_one")
    base.require(current["previous_physical_event_id"].notna().all(), "history_tick_does_not_resolve_to_unique_event")
    previous = event[["physical_event_id", "round_id", "focal_steamid", "opponent_steamid", "focal_wins", "focal_post_buy_team", "opponent_post_buy_team"]].rename(columns={"physical_event_id": "previous_physical_event_id", "round_id": "previous_round_id", "focal_steamid": "previous_focal_player", "opponent_steamid": "previous_opponent_player", "focal_wins": "previous_focal_wins", "focal_post_buy_team": "previous_focal_team", "opponent_post_buy_team": "previous_opponent_team"})
    current = current.merge(previous, on="previous_physical_event_id", how="left", validate="many_to_one")
    base.require(current[["previous_round_id", "previous_focal_player", "previous_opponent_player", "previous_focal_wins", "previous_focal_team", "previous_opponent_team"]].notna().all().all(), "previous_raw_event_join_failure")
    previous_as_focal, previous_as_opponent = current["previous_focal_player"].eq(current["focal_player_id"]), current["previous_opponent_player"].eq(current["focal_player_id"])
    base.require((previous_as_focal ^ previous_as_opponent).all(), "previous_player_view_direction_failure")
    previous_win = (previous_as_focal & current["previous_focal_wins"].eq(1)) | (previous_as_opponent & current["previous_focal_wins"].eq(0))
    current["previous_victim_player"] = np.where(previous_as_focal, current["previous_opponent_player"], current["previous_focal_player"])
    current["previous_victim_team"] = np.where(previous_as_focal, current["previous_opponent_team"], current["previous_focal_team"])
    current["interval_ticks"] = current["event_tick_raw"].to_numpy(float) - current["history_event_tick"].to_numpy(float)
    interval_error = float(np.max(np.abs(current["interval_ticks"].to_numpy(float) / 64.0 - current["T_seconds"].to_numpy(float))))
    base.require(interval_error <= 1e-10 and (current["interval_ticks"] > 0).all(), "strict_interval_reconstruction_failure")
    predicates = {"predicate_current_loss": current["focal_wins"].eq(0), "predicate_previous_focal_win": previous_win, "predicate_same_round": current["round_id"].astype(str).eq(current["previous_round_id"].astype(str)), "predicate_same_team": current["opponent_post_buy_team"].eq(current["previous_victim_team"]), "predicate_different_player": current["opponent_steamid"].ne(current["previous_victim_player"]), "predicate_within_5s": current["interval_ticks"].le(prior.TRADE_WINDOW_TICKS)}
    for name, value in predicates.items():
        current[name] = value.astype(int)
    current["direct_trade_5s"] = np.logical_and.reduce([value.to_numpy(bool) for value in predicates.values()]).astype(int)
    label_fields = ["duel_id", "history_event_tick", "interval_ticks", "previous_physical_event_id", "direct_trade_5s", *predicates]
    labels = matrix.merge(current[label_fields], on="duel_id", how="inner", validate="one_to_one")
    base.require(len(labels) == len(matrix) and labels["history_event_tick"].notna().all() and np.isfinite(labels[["S_h2", "z_time", "interval_ticks"]].to_numpy(float)).all(), "matrix_label_join_or_numeric_failure")
    labels = labels[["duel_id", "match_id", "queue_order", "Y_t", "H_t", "S_h2", "z_time", "history_event_tick", "interval_ticks", "previous_physical_event_id", "direct_trade_5s", *predicates, *base.NUMERIC, *base.CATEGORICAL]].copy()
    write_parquet(output_path, labels)
    checkpoint = {"stage": "d1a_m3h2_direct_trade_chain_v2_label_audit", "created_at_utc": now(), "input_lock_sha256": lock_hash, "labels_sha256": base.digest(output_path), "G1": {"status": "pass", "reason": "strict_unique_referenced_prior_participant_event_and_same_round_team_trade_contract", "metrics": {"labelled_rows": int(len(labels)), "labelled_matches": int(labels["match_id"].nunique()), "direct_trade_rows": int(labels["direct_trade_5s"].sum()), "direct_trade_matches": int(labels.loc[labels["direct_trade_5s"].eq(1), "match_id"].nunique()), "referenced_prior_keys": int(len(target_keys)), "state_interval_max_abs_error_seconds": interval_error, "state_audit": state_audit, "predicate_counts": {name: int(labels[name].sum()) for name in predicates}}}}
    write_json(checkpoint_path, checkpoint)
    return labels, checkpoint


def output_hashes() -> dict[str, str]:
    return {str(item.relative_to(OUT)): base.digest(item) for item in sorted(OUT.rglob("*")) if item.is_file() and item.name not in {"file_sha256.json", "manifest.json"} and not any(part.startswith("failed_") for part in item.relative_to(OUT).parts)}


def run(phase: str) -> None:
    _, lock_hash = input_lock()
    labels, label_gate = labelled_matrix(lock_hash)
    support, support_gate = prior.fold_support(labels)
    support_path, smoke_path = OUT / "fold_trade_support.parquet", OUT / "checkpoint_03_smoke.json"
    if not support_path.exists():
        write_parquet(support_path, support)
    if not smoke_path.exists():
        write_json(smoke_path, {"stage": "d1a_m3h2_direct_trade_chain_v2_smoke", "created_at_utc": now(), "input_lock_sha256": lock_hash, "fold_trade_support_sha256": base.digest(support_path), "G0": {"status": "pass", "reason": "v5_completed_m3h2_and_hash_locked_d1a_inputs_only"}, "G1": label_gate["G1"], "G2": {"status": "pass", "folds": [{"name": n, "train": [a, b], "evaluation": [c, d]} for n, a, b, c, d in prior.FOLDS]}, "G3": support_gate})
    if phase == "smoke":
        print(json.dumps({"status": "smoke_passed", "output_root": str(OUT)}, ensure_ascii=True))
        return
    prior.OUT = OUT
    fits = [prior.fit_fold(name, labels, start, end, lock_hash) for name, start, end, _, _ in prior.FOLDS]
    rows = [{"fold": item["fold"], "model": model, "state_coefficient": coefficient, "direct_trade_rows": item["direct_trade_rows"], "direct_trade_matches": item["direct_trade_matches"]} for item in fits for model, coefficient in (("reproduction", item["reproduction_state_coefficient"]), ("interaction_nontrade", item["interaction_nontrade_state_coefficient"]), ("interaction_trade", item["interaction_trade_state_coefficient"]), ("leave_trade_out", item["leave_trade_out_state_coefficient"]))]
    coefficients = pd.DataFrame(rows)
    coefficient_path = OUT / "fold_coefficients.parquet"
    if not coefficient_path.exists(): write_parquet(coefficient_path, coefficients)
    summary = coefficients.groupby("model", sort=True)["state_coefficient"].agg(["mean", "std", "min", "max"]).reset_index().rename(columns={"mean": "fold_mean", "std": "fold_sd", "min": "fold_min", "max": "fold_max"})
    interaction = pd.DataFrame([{key: item[key] for key in ("fold", "interaction_nontrade_state_coefficient", "interaction_trade_state_coefficient", "interaction_coefficient_difference", "leave_trade_out_state_coefficient", "leave_trade_out_minus_reproduction", "reproduction_abs_error")} for item in fits])
    if not (OUT / "coefficient_summary.parquet").exists(): write_parquet(OUT / "coefficient_summary.parquet", summary)
    if not (OUT / "interaction_diagnostic.parquet").exists(): write_parquet(OUT / "interaction_diagnostic.parquet", interaction)
    gates = {"G0": {"status": "pass", "reason": "v5_completed_m3h2_and_hash_locked_d1a_inputs_only"}, "G1": label_gate["G1"], "G2": {"status": "pass", "folds": [item["fold"] for item in fits]}, "G3": support_gate, "G4": {"status": "pass", "reason": "all_l2_fits_converged_and_reproduced_v4_m3h2_coefficients", "max_reproduction_abs_error": float(max(item["reproduction_abs_error"] for item in fits))}, "G5": {"status": "pass", "reason": "complete_label_support_and_three_coefficient_diagnostics"}}
    if not (OUT / "checkpoint_05_gate_audit.json").exists(): write_json(OUT / "checkpoint_05_gate_audit.json", {"stage": "d1a_m3h2_direct_trade_chain_v2_gates", "created_at_utc": now(), "input_lock_sha256": lock_hash, "gates": gates})
    hashes_path = OUT / "file_sha256.json"
    if not hashes_path.exists(): write_json(hashes_path, output_hashes())
    manifest_path = OUT / "manifest.json"
    if not manifest_path.exists(): write_json(manifest_path, {"stage": "d1a_m3h2_direct_trade_chain_v2", "status": "passed", "created_at_utc": now(), "classification": "exploratory_nonconfirmatory_result_informed_d1a_only", "input_lock_sha256": lock_hash, "state_model": "M3_h2", "direct_trade_window_seconds": 5, "gate_statuses": {name: value["status"] for name, value in gates.items()}, "interpretation": "A five-second direct-trade-chain label diagnostic for the already selected D1-A M3(2 s) model. It is descriptive only and does not establish that M3 captures trades, nor any hot-hand, causal, state, prediction, or confirmatory claim.", "file_sha256_sha256": base.digest(hashes_path), "v1_repair": "referenced_prior_key_uniqueness_only"})
    print(json.dumps({"status": "passed", "manifest": str(manifest_path)}, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    try: run(args.phase)
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True)
        failed = OUT / f"failed_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        failed.mkdir(exist_ok=False)
        (failed / "error.txt").write_text("".join(traceback.format_exception(error)), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
