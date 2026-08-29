"""Fit resumable D1 unified pre-match and residual-history L2 logistic models."""

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
STATE = ROOT / "results" / "history_residual_state_oof_d1_v2"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "69_D1统一赛前与历史残差特征模型执行协议_v1_2026-08-29.md"
OUT = ROOT / "results" / "unified_history_feature_model_d1_v1"
DATASETS = ("D1-A", "D1-B", "D1-C")
FOLDS = {"F1": (("D1-A",), "D1-B"), "F2": (("D1-A", "D1-B"), "D1-C")}
MODELS = {"U0_base_time": (), "UH_lag1": ("H_t",), "US_decay_sum": ("S_t",), "UHS_lag1_decay_sum": ("H_t", "S_t")}
C, SEED, MAX_ITER, TOL, EPS = 0.01, 20260829, 5000, 1e-8, 1e-12
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


def input_lock() -> tuple[dict[str, Any], str]:
    path = OUT / "checkpoint_01_input_lock.json"
    asset_audit, state_audit = verify_tree(ASSET), verify_tree(STATE)
    require(not asset_audit["mismatches"] and asset_audit["manifest_file_sha256_matches"], "conditional_asset_hash_chain_mismatch")
    require(not state_audit["mismatches"] and state_audit["manifest_file_sha256_matches"], "state_hash_chain_mismatch")
    sources = [PROTOCOL, Path(__file__).resolve(), HIST, D1B, D1C, ASSET / "manifest.json", ASSET / "file_sha256.json", STATE / "manifest.json", STATE / "file_sha256.json"]
    for dataset in DATASETS:
        sources.extend([ASSET / dataset / "residual_asset.parquet", STATE / "state" / dataset / "state_matrix.parquet"])
    records = {str(item): digest(item) for item in sources}
    if path.exists():
        saved = read_json(path)
        require(saved.get("inputs") == records, "input_lock_source_hash_changed")
        return saved, digest(path)
    value = {"stage": "unified_history_feature_model_d1_v1_input_lock", "created_at_utc": now(), "classification": "exploratory_nonconfirmatory_result_informed_d1_only", "inputs": records, "conditional_asset_hash_audit": asset_audit, "state_hash_audit": state_audit, "models": {name: list(extra) for name, extra in MODELS.items()}, "regularization": {"penalty": "l2", "C": C}, "e1_path_access_count": 0}
    write_json(path, value)
    return value, digest(path)


def raw_candidate(dataset: str) -> pd.DataFrame:
    if dataset == "D1-A":
        frame = pd.read_parquet(HIST)
        frame = frame.loc[frame["partition"].astype(str).eq(dataset)].copy()
    elif dataset == "D1-B":
        frame = pd.read_parquet(D1B)
    else:
        frame = pd.read_parquet(D1C)
    required = {"duel_id", "match_id", "map_id", "event_tick", *NUMERIC, *CATEGORICAL}
    require(required.issubset(frame.columns) and not frame["duel_id"].duplicated().any(), f"raw_candidate_contract_failure:{dataset}")
    return frame


def load_matrix(dataset: str, lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = OUT / "state" / dataset / "unified_matrix.parquet"
    checkpoint_path = OUT / "state" / dataset / f"checkpoint_02_matrix_{dataset}.json"
    if path.exists():
        checkpoint = read_json(checkpoint_path)
        require(checkpoint.get("input_lock_sha256") == lock_hash and checkpoint.get("matrix_sha256") == digest(path), f"matrix_checkpoint_mismatch:{dataset}")
        return pd.read_parquet(path), checkpoint
    raw = raw_candidate(dataset)
    asset = pd.read_parquet(ASSET / dataset / "residual_asset.parquet")
    asset = asset.loc[asset["residual_valid"].astype(bool) & asset["primary_analysis_eligible"].astype(bool)].copy()
    state = pd.read_parquet(STATE / "state" / dataset / "state_matrix.parquet")
    joined = raw.merge(asset[["duel_id", "focal_player_id", "e_t", "Y_t"]], on="duel_id", how="inner", validate="one_to_one")
    if "focal_wins" in joined:
        require(np.array_equal(joined["focal_wins"].to_numpy(int), joined["Y_t"].to_numpy(int)), f"raw_asset_outcome_mismatch:{dataset}")
    for frame in (joined, state):
        for column in ("match_id", "map_id", "focal_player_id"):
            frame[column] = frame[column].astype(str)
    keys = ["match_id", "map_id", "focal_player_id", "event_tick", "e_t"]
    require(not state.duplicated(keys).any(), f"nonunique_state_signature:{dataset}")
    merged = joined.merge(state[keys + ["H_t", "S_t", "time_since_history_tick"]], on=keys, how="inner", validate="one_to_one")
    require(len(merged) == len(state) and not merged["duel_id"].duplicated().any(), f"state_join_loss:{dataset}")
    merged["log_time"] = np.log2(1.0 + merged["time_since_history_tick"].to_numpy(float))
    all_numeric = [*NUMERIC, "log_time", "H_t", "S_t"]
    require(np.isfinite(merged[all_numeric].to_numpy(float)).all() and merged["Y_t"].isin([0, 1]).all(), f"nonfinite_or_nonbinary:{dataset}")
    output = merged[["duel_id", "match_id", "Y_t", *NUMERIC, *CATEGORICAL, "log_time", "H_t", "S_t"]].copy()
    write_parquet(path, output)
    metrics = {"rows": len(output), "matches": int(output["match_id"].nunique()), "state_rows": len(state), "strict_history_source": "history_residual_state_oof_d1_v2/state_matrix"}
    checkpoint = {"stage": "unified_history_feature_model_d1_v1_matrix", "created_at_utc": now(), "dataset": dataset, "input_lock_sha256": lock_hash, "matrix_sha256": digest(path), "G1": {"status": "pass", "reason": "raw_candidate_asset_state_one_to_one_join", "metrics": metrics}}
    write_json(checkpoint_path, checkpoint)
    return output, checkpoint


class Design:
    def __init__(self, extra: tuple[str, ...]) -> None:
        self.numeric = [*NUMERIC, "log_time", *extra]
        self.means: dict[str, float] = {}
        self.sds: dict[str, float] = {}
        self.categories: dict[str, list[str]] = {}
        self.offsets: dict[str, tuple[int, int]] = {}
        self.names: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "Design":
        for field in self.numeric:
            values = frame[field].to_numpy(float)
            self.means[field], self.sds[field] = float(values.mean()), float(values.std(ddof=0))
            require(math.isfinite(self.sds[field]) and self.sds[field] > 0, f"invalid_scale:{field}")
        self.names = [f"num_{field}" for field in self.numeric]
        cursor = len(self.names)
        for field in CATEGORICAL:
            values = sorted(frame[field].astype(str).unique().tolist())
            require(values, f"empty_category:{field}")
            self.categories[field] = values
            dummy = values[1:]
            self.offsets[field] = (cursor, cursor + len(dummy))
            self.names.extend(f"cat_{field}={value}" for value in dummy)
            cursor += len(dummy)
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[sparse.csr_matrix, dict[str, int]]:
        numeric = np.column_stack([(frame[field].to_numpy(float) - self.means[field]) / self.sds[field] for field in self.numeric])
        blocks: list[sparse.csr_matrix] = [sparse.csr_matrix(numeric)]
        unknown: dict[str, int] = {}
        for field in CATEGORICAL:
            lookup = {value: index for index, value in enumerate(self.categories[field][1:])}
            raw = frame[field].astype(str).to_numpy()
            unknown[field] = int(sum(value not in self.categories[field] for value in raw))
            rows = np.fromiter((index for index, value in enumerate(raw) if value in lookup), dtype=np.int64)
            columns = np.fromiter((lookup[value] for value in raw if value in lookup), dtype=np.int64)
            blocks.append(sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(frame), len(lookup))))
        return sparse.hstack(blocks, format="csr"), unknown


def fit_model(train: pd.DataFrame, evaluation: pd.DataFrame, name: str, extra: tuple[str, ...]) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    design = Design(extra).fit(train)
    x_train, _ = design.transform(train)
    x_eval, unknown = design.transform(evaluation)
    fitted = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=MAX_ITER, tol=TOL, random_state=SEED)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fitted.fit(x_train, train["Y_t"].to_numpy(int))
    require(not any(issubclass(item.category, ConvergenceWarning) for item in caught), f"convergence_warning:{name}")
    probability = fitted.predict_proba(x_eval)[:, 1]
    require(np.isfinite(probability).all() and np.all((probability > 0) & (probability < 1)), f"invalid_probability:{name}")
    coefficient = fitted.coef_[0]
    rows = []
    for column, value in zip(design.names, coefficient, strict=True):
        if column.startswith("num_"):
            rows.append({"model": name, "measure": "standardized_coefficient", "source_column": column.removeprefix("num_"), "category": "", "value": float(value)})
        else:
            field, category = column.removeprefix("cat_").split("=", 1)
            rows.append({"model": name, "measure": "dummy_coefficient", "source_column": field, "category": category, "value": float(value)})
    for field, (start, end) in design.offsets.items():
        contribution = x_train[:, start:end] @ coefficient[start:end]
        rows.append({"model": name, "measure": "training_contribution_sd", "source_column": field, "category": "", "value": float(np.std(np.asarray(contribution).ravel(), ddof=0))})
    details = {"features": design.names, "numeric_means": design.means, "numeric_sds": design.sds, "intercept": float(fitted.intercept_[0]), "n_iter": int(fitted.n_iter_[0]), "unknown_evaluation_categories": unknown}
    return probability, details, pd.DataFrame.from_records(rows)


def smoke(matrices: dict[str, pd.DataFrame], lock_hash: str) -> None:
    path = OUT / "checkpoint_03_smoke.json"
    if path.exists():
        require(read_json(path).get("input_lock_sha256") == lock_hash, "smoke_input_lock_mismatch")
        return
    matches = sorted(matrices["D1-B"]["match_id"].unique())[:30]
    evaluation = matrices["D1-B"].loc[matrices["D1-B"]["match_id"].isin(matches)].copy()
    require(len(matches) == 30 and not evaluation.empty, "smoke_match_support_failure")
    models = {}
    for name, extra in MODELS.items():
        probability, details, _ = fit_model(matrices["D1-A"], evaluation, name, extra)
        models[name] = {"n_iter": details["n_iter"], "probability_range": [float(probability.min()), float(probability.max())], "unknown_categories": details["unknown_evaluation_categories"]}
    write_json(path, {"stage": "unified_history_feature_model_d1_v1_smoke", "created_at_utc": now(), "input_lock_sha256": lock_hash, "G0": {"status": "pass"}, "G1": {"status": "pass"}, "G2": {"status": "pass", "evaluation_matches": len(matches), "evaluation_rows": len(evaluation)}, "G3": {"status": "pass", "models": models}})


def fold_fit(name: str, train_sets: tuple[str, ...], evaluation_set: str, matrices: dict[str, pd.DataFrame], lock_hash: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prediction_path = OUT / "state" / name / "oof_predictions.parquet"
    coefficient_path = OUT / "state" / name / "coefficients.parquet"
    checkpoint_path = OUT / "state" / name / "checkpoint_04_fit.json"
    if prediction_path.exists():
        checkpoint = read_json(checkpoint_path)
        require(checkpoint.get("input_lock_sha256") == lock_hash and checkpoint.get("oof_sha256") == digest(prediction_path) and checkpoint.get("coefficient_sha256") == digest(coefficient_path), f"fit_checkpoint_mismatch:{name}")
        return pd.read_parquet(prediction_path), pd.read_parquet(coefficient_path), checkpoint
    train = pd.concat([matrices[item] for item in train_sets], ignore_index=True)
    evaluation = matrices[evaluation_set].copy()
    require(set(train["match_id"]).isdisjoint(set(evaluation["match_id"])), f"train_evaluation_match_overlap:{name}")
    output = evaluation[["duel_id", "match_id", "Y_t"]].copy()
    coefficients, details = [], {}
    for model, extra in MODELS.items():
        output[model], details[model], rows = fit_model(train, evaluation, model, extra)
        coefficients.append(rows.assign(fold=name))
    coefficient = pd.concat(coefficients, ignore_index=True)
    write_parquet(prediction_path, output)
    write_parquet(coefficient_path, coefficient)
    fit_path = OUT / "state" / name / "training_fit.json"
    write_json(fit_path, {"stage": "unified_history_feature_model_d1_v1_training_fit", "created_at_utc": now(), "input_lock_sha256": lock_hash, "fold": name, "train_datasets": list(train_sets), "evaluation_dataset": evaluation_set, "train_rows": len(train), "train_matches": int(train["match_id"].nunique()), "evaluation_rows": len(evaluation), "evaluation_matches": int(evaluation["match_id"].nunique()), "models": details})
    checkpoint = {"stage": "unified_history_feature_model_d1_v1_fit", "created_at_utc": now(), "input_lock_sha256": lock_hash, "fold": name, "oof_sha256": digest(prediction_path), "coefficient_sha256": digest(coefficient_path), "training_fit_sha256": digest(fit_path), "G2": {"status": "pass", "train_matches": int(train["match_id"].nunique()), "evaluation_matches": int(evaluation["match_id"].nunique())}, "G3": {"status": "pass", "reason": "all_l2_models_converged_with_finite_probabilities"}}
    write_json(checkpoint_path, checkpoint)
    return output, coefficient, checkpoint


def match_metrics(oof: pd.DataFrame, fold: str) -> pd.DataFrame:
    y = oof["Y_t"].to_numpy(float)
    values: dict[str, Any] = {"match_id": oof["match_id"].astype(str)}
    for model in MODELS:
        probability = np.clip(oof[model].to_numpy(float), EPS, 1 - EPS)
        values[f"{model}_log_loss"] = -(y * np.log(probability) + (1 - y) * np.log1p(-probability))
        values[f"{model}_brier"] = (probability - y) ** 2
    output = pd.DataFrame(values).groupby("match_id", sort=True).mean().reset_index()
    output.insert(0, "fold", fold)
    return output


def hashes() -> dict[str, str]:
    return {str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"} and not any(part.startswith("failed_") for part in path.relative_to(OUT).parts)}


def run(phase: str) -> None:
    _, lock_hash = input_lock()
    matrices, matrix_checks = {}, {}
    for dataset in DATASETS:
        matrices[dataset], matrix_checks[dataset] = load_matrix(dataset, lock_hash)
    smoke(matrices, lock_hash)
    if phase == "smoke":
        print(json.dumps({"status": "smoke_passed", "output_root": str(OUT)}, ensure_ascii=True))
        return
    metric_parts, coefficient_parts, fold_checks = [], [], {}
    for name, (train, evaluation) in FOLDS.items():
        oof, coefficient, check = fold_fit(name, train, evaluation, matrices, lock_hash)
        metric_parts.append(match_metrics(oof, name))
        coefficient_parts.append(coefficient)
        fold_checks[name] = check
    metrics_path, coefficient_path = OUT / "match_metrics.parquet", OUT / "coefficient_summary.parquet"
    if not metrics_path.exists():
        write_parquet(metrics_path, pd.concat(metric_parts, ignore_index=True))
    if not coefficient_path.exists():
        write_parquet(coefficient_path, pd.concat(coefficient_parts, ignore_index=True))
    gates = {"G0": {"status": "pass", "reason": "d1_only_hash_locked_inputs", "metrics": {"e1_path_access_count": 0}}, "G1": {"status": "pass", "reason": "raw_candidate_asset_state_one_to_one_join", "metrics": {dataset: check["G1"]["metrics"] for dataset, check in matrix_checks.items()}}, "G2": {"status": "pass", "reason": "two_time_forward_disjoint_match_folds", "metrics": {name: check["G2"] for name, check in fold_checks.items()}}, "G3": {"status": "pass", "reason": "all_l2_models_converged_with_finite_probabilities", "metrics": {name: check["G3"] for name, check in fold_checks.items()}}, "G4": {"status": "pass", "reason": "oof_probability_match_metric_and_coefficient_assets_complete"}, "G5": {"status": "pending", "reason": "exploratory_result_informed_conditional_coefficient_comparison_only"}}
    gate_path = OUT / "checkpoint_05_gate_audit.json"
    if not gate_path.exists():
        write_json(gate_path, {"stage": "unified_history_feature_model_d1_v1_gate_audit", "created_at_utc": now(), "input_lock_sha256": lock_hash, "gates": gates})
    hash_path = OUT / "file_sha256.json"
    if not hash_path.exists():
        write_json(hash_path, hashes())
    else:
        for relative, expected in read_json(hash_path).items():
            require((OUT / relative).is_file() and digest(OUT / relative) == expected, f"output_hash_mismatch:{relative}")
    manifest_path = OUT / "manifest.json"
    if not manifest_path.exists():
        write_json(manifest_path, {"stage": "unified_history_feature_model_d1_v1", "status": "passed", "created_at_utc": now(), "classification": "exploratory_nonconfirmatory_result_informed_d1_only", "input_lock_sha256": lock_hash, "gate_statuses": {name: value["status"] for name, value in gates.items()}, "interpretation": "Unified D1 original-pre-match-feature, time, and history-residual L2 coefficients; descriptive conditional weights only, not raw p_hat replacement, predictive contribution, hot-hand, state, causal, deployable, or confirmatory evidence.", "file_sha256_sha256": digest(hash_path)})
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
        write_json(failed / "failure_manifest.json", {"stage": "unified_history_feature_model_d1_v1", "status": "failed", "created_at_utc": now(), "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
