"""Describe selected D1-A M3(2 s) state coefficients in pre-outcome five-second potential-trade contexts."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

import audit_d1a_m3h2_direct_trade_chain_v1 as helper
import explore_d1a_dynamic_history_half_life_v1 as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "d1a_m3h2_potential_trade_chain_v1"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "79_D1A潜在补枪链_M3_2秒状态诊断执行锁_v1_2026-08-29.md"
V2 = ROOT / "results" / "d1a_m3h2_direct_trade_chain_v2"
FOLDS = helper.FOLDS
PREDICATES = ("predicate_previous_focal_win", "predicate_same_round", "predicate_same_team", "predicate_different_player", "predicate_within_5s")


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


def input_paths() -> list[Path]:
    paths = [PROTOCOL, Path(__file__).resolve(), ROOT / "scripts" / "audit_d1a_m3h2_direct_trade_chain_v1.py", V2 / "manifest.json", V2 / "file_sha256.json", V2 / "checkpoint_02_label_audit.json", V2 / "d1a_direct_trade_labels.parquet"]
    paths += [helper.V4 / "state" / name / "training_fit.json" for name, *_ in FOLDS]
    return paths


def input_lock() -> tuple[dict[str, Any], str]:
    path = OUT / "checkpoint_01_input_lock.json"
    audit = base.verify_tree(V2)
    manifest = read_json(V2 / "manifest.json")
    base.require(not audit["mismatches"] and audit["manifest_file_sha256_matches"] and manifest.get("status") == "passed", "v2_label_ledger_hash_chain_failure")
    records = {str(item): base.digest(item) for item in input_paths()}
    if path.exists():
        saved = read_json(path)
        base.require(saved.get("inputs") == records, "input_lock_source_hash_changed")
        return saved, base.digest(path)
    value = {"stage": "d1a_m3h2_potential_trade_chain_v1_input_lock", "created_at_utc": now(), "classification": "exploratory_nonconfirmatory_result_informed_d1a_only", "inputs": records, "v2_label_ledger_hash_audit": audit, "state_model": "M3_h2", "state_half_life_seconds": 2, "potential_trade_window_seconds": 5, "folds": [{"name": n, "train_queue_order": [a, b], "evaluation_queue_order": [c, d]} for n, a, b, c, d in FOLDS], "external_partition_access_count": 0, "v2_reuse_boundary": "label_ledger_predicates_only_no_v2_coefficients"}
    write_json(path, value)
    return value, base.digest(path)


def labelled_matrix(lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    output, checkpoint_path = OUT / "d1a_potential_trade_labels.parquet", OUT / "checkpoint_02_label_audit.json"
    if output.exists():
        checkpoint = read_json(checkpoint_path)
        base.require(checkpoint.get("input_lock_sha256") == lock_hash and checkpoint.get("labels_sha256") == base.digest(output), "label_checkpoint_mismatch")
        return pd.read_parquet(output), checkpoint
    labels = pd.read_parquet(V2 / "d1a_direct_trade_labels.parquet")
    required = {"duel_id", "match_id", "queue_order", "Y_t", "S_h2", "z_time", "direct_trade_5s", "predicate_current_loss", *PREDICATES, *base.NUMERIC, *base.CATEGORICAL}
    base.require(required.issubset(labels.columns) and not labels["duel_id"].duplicated().any(), "v2_label_ledger_contract_failure")
    potential = np.logical_and.reduce([labels[field].to_numpy(int).astype(bool) for field in PREDICATES])
    labels["potential_trade_5s"] = potential.astype(int)
    direct_expected = potential & labels["predicate_current_loss"].to_numpy(int).astype(bool)
    base.require(np.array_equal(direct_expected.astype(int), labels["direct_trade_5s"].to_numpy(int)), "direct_to_potential_label_contract_failure")
    base.require(set(labels.loc[labels["potential_trade_5s"].eq(1), "Y_t"].astype(int).unique()) == {0, 1}, "potential_label_outcome_support_failure")
    fields = ["duel_id", "match_id", "queue_order", "Y_t", "S_h2", "z_time", "potential_trade_5s", *PREDICATES, *base.NUMERIC, *base.CATEGORICAL]
    labels = labels[fields].copy()
    write_parquet(output, labels)
    by_outcome = labels.loc[labels["potential_trade_5s"].eq(1)].groupby("Y_t").size().to_dict()
    checkpoint = {"stage": "d1a_m3h2_potential_trade_chain_v1_label_audit", "created_at_utc": now(), "input_lock_sha256": lock_hash, "labels_sha256": base.digest(output), "G1": {"status": "pass", "reason": "five_nonoutcome_predicates_and_direct_label_identity", "metrics": {"rows": int(len(labels)), "matches": int(labels["match_id"].nunique()), "potential_trade_rows": int(labels["potential_trade_5s"].sum()), "potential_trade_matches": int(labels.loc[labels["potential_trade_5s"].eq(1), "match_id"].nunique()), "potential_trade_rows_by_current_outcome": {str(k): int(v) for k, v in by_outcome.items()}}}}
    write_json(checkpoint_path, checkpoint)
    return labels, checkpoint


def fold_support(labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = []
    for fold, train_start, train_end, eval_start, eval_end in FOLDS:
        for role, start, end, min_rows, min_matches, min_per_outcome in (("training", train_start, train_end, 50, 10, 20), ("evaluation", eval_start, eval_end, 20, 5, 10)):
            subset = labels.loc[labels["queue_order"].between(start, end)]
            potential = subset.loc[subset["potential_trade_5s"].eq(1)]
            outcome_counts = potential.groupby("Y_t").size().to_dict()
            record = {"fold": fold, "role": role, "queue_order_start": start, "queue_order_end": end, "all_rows": int(len(subset)), "all_matches": int(subset["match_id"].nunique()), "potential_trade_rows": int(len(potential)), "potential_trade_matches": int(potential["match_id"].nunique()), "potential_y0_rows": int(outcome_counts.get(0, 0)), "potential_y1_rows": int(outcome_counts.get(1, 0)), "required_rows": min_rows, "required_matches": min_matches, "required_per_outcome": min_per_outcome}
            record["support_pass"] = bool(record["potential_trade_rows"] >= min_rows and record["potential_trade_matches"] >= min_matches and record["potential_y0_rows"] >= min_per_outcome and record["potential_y1_rows"] >= min_per_outcome)
            records.append(record)
    support = pd.DataFrame(records)
    base.require(support["support_pass"].all(), f"potential_trade_support_failure:{support.loc[~support['support_pass']].to_dict(orient='records')}")
    return support, {"status": "pass", "reason": "predeclared_potential_trade_rows_matches_and_both_current_outcomes", "metrics": {f"{row.fold}_{row.role}": row._asdict() for row in support.itertuples(index=False)}}


def fit_fold(name: str, labels: pd.DataFrame, start: int, end: int, lock_hash: str) -> dict[str, Any]:
    path = OUT / "state" / name / "checkpoint_04_fit.json"
    if path.exists():
        saved = read_json(path)
        base.require(saved.get("input_lock_sha256") == lock_hash, f"fold_checkpoint_input_lock_mismatch:{name}")
        return saved
    train = labels.loc[labels["queue_order"].between(start, end)].copy()
    design, reproduction_model, x_train = helper.base_fit(train, f"{name}_reproduction")
    state_index = design.numeric.index("S_h2")
    reproduction = float(reproduction_model.coef_[0][state_index])
    expected = helper.parent_state_coefficient(name)
    base.require(abs(reproduction - expected) <= 1e-8, f"v4_state_coefficient_reproduction_failure:{name}:{reproduction}:{expected}")
    state_standardized = (train["S_h2"].to_numpy(float) - design.means["S_h2"]) / design.sds["S_h2"]
    potential = train["potential_trade_5s"].to_numpy(float)
    x_interaction = sparse.hstack([x_train, sparse.csr_matrix(np.column_stack([potential, potential * state_standardized]))], format="csr")
    interaction = helper.fit_l2(x_interaction, train["Y_t"].to_numpy(int), f"{name}_potential_trade_interaction")
    b_nonpotential = float(interaction.coef_[0][state_index])
    b_difference = float(interaction.coef_[0][-1])
    b_potential = b_nonpotential + b_difference
    leave_train = train.loc[train["potential_trade_5s"].eq(0)].copy()
    leave_design, leave_model, leave_x = helper.base_fit(leave_train, f"{name}_leave_potential_trade_out")
    leave_coefficient = float(leave_model.coef_[0][leave_design.numeric.index("S_h2")])
    value = {"stage": "d1a_m3h2_potential_trade_chain_v1_fold_fit", "created_at_utc": now(), "input_lock_sha256": lock_hash, "fold": name, "train_queue_order": [start, end], "train_rows": int(len(train)), "train_matches": int(train["match_id"].nunique()), "potential_trade_rows": int(train["potential_trade_5s"].sum()), "potential_trade_matches": int(train.loc[train["potential_trade_5s"].eq(1), "match_id"].nunique()), "reproduction_state_coefficient": reproduction, "v4_saved_state_coefficient": expected, "reproduction_abs_error": abs(reproduction - expected), "interaction_nonpotential_state_coefficient": b_nonpotential, "interaction_potential_state_coefficient": b_potential, "interaction_coefficient_difference": b_difference, "interaction_potential_intercept_shift": float(interaction.coef_[0][-2]), "leave_potential_trade_out_rows": int(len(leave_train)), "leave_potential_trade_out_state_coefficient": leave_coefficient, "leave_potential_trade_out_minus_reproduction": leave_coefficient - reproduction, "reproduction_design_shape": [int(x_train.shape[0]), int(x_train.shape[1])], "interaction_design_shape": [int(x_interaction.shape[0]), int(x_interaction.shape[1])], "leave_potential_trade_out_design_shape": [int(leave_x.shape[0]), int(leave_x.shape[1])], "reproduction_n_iter": int(reproduction_model.n_iter_[0]), "interaction_n_iter": int(interaction.n_iter_[0]), "leave_potential_trade_out_n_iter": int(leave_model.n_iter_[0])}
    write_json(path, value)
    return value


def hashes() -> dict[str, str]:
    return {str(path.relative_to(OUT)): base.digest(path) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"} and not any(part.startswith("failed_") for part in path.relative_to(OUT).parts)}


def run(phase: str) -> None:
    _, lock_hash = input_lock()
    labels, label_checkpoint = labelled_matrix(lock_hash)
    support, support_gate = fold_support(labels)
    support_path, smoke_path = OUT / "fold_potential_trade_support.parquet", OUT / "checkpoint_03_smoke.json"
    if not support_path.exists(): write_parquet(support_path, support)
    if not smoke_path.exists(): write_json(smoke_path, {"stage": "d1a_m3h2_potential_trade_chain_v1_smoke", "created_at_utc": now(), "input_lock_sha256": lock_hash, "support_sha256": base.digest(support_path), "G0": {"status": "pass", "reason": "hash_locked_d1a_label_ledger_only"}, "G1": label_checkpoint["G1"], "G2": {"status": "pass", "folds": [{"name": n, "train": [a, b], "evaluation": [c, d]} for n, a, b, c, d in FOLDS]}, "G3": support_gate})
    if phase == "smoke":
        print(json.dumps({"status": "smoke_passed", "output_root": str(OUT)}, ensure_ascii=True)); return
    fits = [fit_fold(name, labels, start, end, lock_hash) for name, start, end, _, _ in FOLDS]
    rows = [{"fold": item["fold"], "model": model, "state_coefficient": coefficient, "potential_trade_rows": item["potential_trade_rows"], "potential_trade_matches": item["potential_trade_matches"]} for item in fits for model, coefficient in (("reproduction", item["reproduction_state_coefficient"]), ("interaction_nonpotential", item["interaction_nonpotential_state_coefficient"]), ("interaction_potential", item["interaction_potential_state_coefficient"]), ("leave_potential_trade_out", item["leave_potential_trade_out_state_coefficient"]))]
    coefficients = pd.DataFrame(rows)
    if not (OUT / "fold_coefficients.parquet").exists(): write_parquet(OUT / "fold_coefficients.parquet", coefficients)
    summary = coefficients.groupby("model", sort=True)["state_coefficient"].agg(["mean", "std", "min", "max"]).reset_index().rename(columns={"mean": "fold_mean", "std": "fold_sd", "min": "fold_min", "max": "fold_max"})
    diagnostic = pd.DataFrame([{key: item[key] for key in ("fold", "interaction_nonpotential_state_coefficient", "interaction_potential_state_coefficient", "interaction_coefficient_difference", "leave_potential_trade_out_state_coefficient", "leave_potential_trade_out_minus_reproduction", "reproduction_abs_error")} for item in fits])
    if not (OUT / "coefficient_summary.parquet").exists(): write_parquet(OUT / "coefficient_summary.parquet", summary)
    if not (OUT / "interaction_diagnostic.parquet").exists(): write_parquet(OUT / "interaction_diagnostic.parquet", diagnostic)
    gates = {"G0": {"status": "pass", "reason": "hash_locked_d1a_label_ledger_only"}, "G1": label_checkpoint["G1"], "G2": {"status": "pass", "folds": [item["fold"] for item in fits]}, "G3": support_gate, "G4": {"status": "pass", "reason": "all_l2_fits_converged_and_reproduced_v4_m3h2_coefficients", "max_reproduction_abs_error": float(max(item["reproduction_abs_error"] for item in fits))}, "G5": {"status": "pass", "reason": "complete_label_support_and_three_coefficient_diagnostics"}}
    if not (OUT / "checkpoint_05_gate_audit.json").exists(): write_json(OUT / "checkpoint_05_gate_audit.json", {"stage": "d1a_m3h2_potential_trade_chain_v1_gates", "created_at_utc": now(), "input_lock_sha256": lock_hash, "gates": gates})
    hashes_path = OUT / "file_sha256.json"
    if not hashes_path.exists(): write_json(hashes_path, hashes())
    manifest_path = OUT / "manifest.json"
    if not manifest_path.exists(): write_json(manifest_path, {"stage": "d1a_m3h2_potential_trade_chain_v1", "status": "passed", "created_at_utc": now(), "classification": "exploratory_nonconfirmatory_result_informed_d1a_only", "input_lock_sha256": lock_hash, "state_model": "M3_h2", "potential_trade_window_seconds": 5, "gate_statuses": {key: value["status"] for key, value in gates.items()}, "interpretation": "Pre-outcome potential-trade context diagnostic for the already selected D1-A M3(2 s) model. Descriptive only; it cannot establish that M3 captures trades or any hot-hand, causal, state, prediction, or confirmatory claim.", "file_sha256_sha256": base.digest(hashes_path)})
    print(json.dumps({"status": "passed", "manifest": str(manifest_path)}, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--phase", choices=("smoke", "full"), default="full"); args = parser.parse_args()
    try: run(args.phase)
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True); failed = OUT / f"failed_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"; failed.mkdir(exist_ok=False); (failed / "error.txt").write_text("".join(traceback.format_exception(error)), encoding="utf-8"); raise


if __name__ == "__main__": main()
