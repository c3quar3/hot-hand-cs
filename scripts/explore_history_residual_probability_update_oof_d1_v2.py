"""Run the three-model D1 exploratory probability-update comparison."""

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
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "results" / "conditional_residual_asset_d1_v2"
STATE = ROOT / "results" / "history_residual_state_oof_d1_v2"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "68_D1历史残差概率更新探索执行协议_v2_2026-08-29.md"
OUT = ROOT / "results" / "history_residual_probability_update_oof_d1_v2"
DATASETS = ("D1-A", "D1-B", "D1-C")
FOLDS = {"F1": (("D1-A",), "D1-B"), "F2": (("D1-A", "D1-B"), "D1-C")}
MODELS = ("M0_base_time", "M1_base_time_lag1", "M2_base_time_decay_sum")
BOOTSTRAPS, SEED, EPS = 5000, 20260829, 1e-12


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


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
    inputs = [PROTOCOL, Path(__file__).resolve(), ASSET / "manifest.json", ASSET / "file_sha256.json", STATE / "manifest.json", STATE / "file_sha256.json"]
    for dataset in DATASETS:
        inputs.extend([ASSET / dataset / "residual_asset.parquet", STATE / "state" / dataset / "state_matrix.parquet"])
    records = {str(item.relative_to(ROOT)): digest(item) for item in inputs}
    if path.exists():
        saved = read_json(path)
        require(saved.get("inputs") == records, "input_lock_source_hash_changed")
        return saved, digest(path)
    value = {"stage": "history_residual_probability_update_oof_d1_v2_input_lock", "created_at_utc": now(), "classification": "exploratory_nonconfirmatory_result_informed_d1_only", "inputs": records, "conditional_asset_hash_audit": asset_audit, "state_hash_audit": state_audit, "models": list(MODELS), "folds": {name: {"train": list(train), "evaluation": evaluation} for name, (train, evaluation) in FOLDS.items()}, "bootstrap": {"draws": BOOTSTRAPS, "seed": SEED, "unit": "match_within_fold"}, "e1_path_access_count": 0}
    write_json(path, value)
    return value, digest(path)


def load_matrix(dataset: str, lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = OUT / "state" / dataset / "model_matrix.parquet"
    checkpoint_path = OUT / "state" / dataset / f"checkpoint_02_matrix_{dataset}.json"
    if path.exists():
        checkpoint = read_json(checkpoint_path)
        require(checkpoint.get("input_lock_sha256") == lock_hash and checkpoint.get("model_matrix_sha256") == digest(path), f"matrix_checkpoint_mismatch:{dataset}")
        return pd.read_parquet(path), checkpoint
    state = pd.read_parquet(STATE / "state" / dataset / "state_matrix.parquet")
    asset = pd.read_parquet(ASSET / dataset / "residual_asset.parquet", columns=["match_id", "map_id", "focal_player_id", "event_tick", "e_t", "Y_t", "p_hat", "residual_valid", "primary_analysis_eligible", "p_support_0p01"])
    asset = asset.loc[asset["residual_valid"].astype(bool) & asset["primary_analysis_eligible"].astype(bool) & asset["p_support_0p01"].astype(bool)].copy()
    keys = ["match_id", "map_id", "focal_player_id", "event_tick", "e_t"]
    for frame in (state, asset):
        for column in ("match_id", "map_id", "focal_player_id"):
            frame[column] = frame[column].astype(str)
    require(not state.duplicated(keys).any() and not asset.duplicated(keys).any(), f"nonunique_matrix_key:{dataset}")
    frame = state.merge(asset[keys + ["Y_t", "p_hat"]], on=keys, how="inner", validate="one_to_one")
    require(len(frame) == len(state), f"state_asset_join_loss:{dataset}")
    frame["logit_p_hat"] = np.log(frame["p_hat"].to_numpy(float) / (1.0 - frame["p_hat"].to_numpy(float)))
    frame["log_time"] = np.log2(1.0 + frame["time_since_history_tick"].to_numpy(float))
    required = ["Y_t", "p_hat", "logit_p_hat", "log_time", "H_t", "S_t"]
    require(np.isfinite(frame[required].to_numpy(float)).all() and frame["Y_t"].isin([0, 1]).all(), f"invalid_values:{dataset}")
    require(frame["p_hat"].between(0.01, 0.99, inclusive="both").all(), f"probability_support_failure:{dataset}")
    frame = frame[["match_id", "map_id", "focal_player_id", "event_tick", "Y_t", "p_hat", "logit_p_hat", "log_time", "H_t", "S_t"]].copy()
    write_parquet(path, frame)
    metrics = {"rows": len(frame), "matches": int(frame["match_id"].nunique()), "players": int(frame["focal_player_id"].nunique()), "p_hat_range": [float(frame["p_hat"].min()), float(frame["p_hat"].max())], "strict_history_source": "history_residual_state_oof_d1_v2/state_matrix"}
    checkpoint = {"stage": "history_residual_probability_update_oof_d1_v2_matrix", "created_at_utc": now(), "dataset": dataset, "input_lock_sha256": lock_hash, "model_matrix_sha256": digest(path), "G1": {"status": "pass", "reason": "one_to_one_strict_history_probability_join", "metrics": metrics}}
    write_json(checkpoint_path, checkpoint)
    return frame, checkpoint


def columns(model: str) -> list[str]:
    base = ["logit_p_hat", "log_time"]
    if model == "M1_base_time_lag1":
        base.append("H_t")
    elif model == "M2_base_time_decay_sum":
        base.append("S_t")
    return base


def fit_predict(train: pd.DataFrame, evaluation: pd.DataFrame, model: str) -> tuple[np.ndarray, dict[str, Any]]:
    fields = columns(model)
    mean = train[fields].mean()
    sd = train[fields].std(ddof=0)
    require(np.isfinite(sd.to_numpy(float)).all() and (sd > 0).all(), f"invalid_scale:{model}")
    x_train = ((train[fields] - mean) / sd).to_numpy(float)
    x_eval = ((evaluation[fields] - mean) / sd).to_numpy(float)
    require(np.linalg.matrix_rank(np.column_stack([np.ones(len(x_train)), x_train])) == len(fields) + 1, f"rank_deficient:{model}")
    fitted = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, tol=1e-8, random_state=SEED)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        fitted.fit(x_train, train["Y_t"].to_numpy(int))
    require(not any(issubclass(item.category, ConvergenceWarning) for item in caught), f"convergence_warning:{model}")
    probability = fitted.predict_proba(x_eval)[:, 1]
    require(np.isfinite(probability).all() and np.all((probability > 0.0) & (probability < 1.0)), f"invalid_probability:{model}")
    details = {"features": fields, "means": {field: float(mean[field]) for field in fields}, "sds": {field: float(sd[field]) for field in fields}, "intercept": float(fitted.intercept_[0]), "coefficients": {field: float(value) for field, value in zip(fields, fitted.coef_[0], strict=True)}, "n_iter": int(fitted.n_iter_[0])}
    return probability, details


def smoke(matrices: dict[str, pd.DataFrame], lock_hash: str) -> None:
    path = OUT / "checkpoint_03_smoke.json"
    if path.exists():
        require(read_json(path).get("input_lock_sha256") == lock_hash, "smoke_input_lock_mismatch")
        return
    matches = sorted(matrices["D1-B"]["match_id"].unique())[:30]
    evaluation = matrices["D1-B"].loc[matrices["D1-B"]["match_id"].isin(matches)].copy()
    require(len(matches) == 30 and not evaluation.empty, "smoke_match_support_failure")
    model_metrics = {}
    for model in MODELS:
        probability, details = fit_predict(matrices["D1-A"], evaluation, model)
        model_metrics[model] = {"n_iter": details["n_iter"], "probability_range": [float(probability.min()), float(probability.max())]}
    write_json(path, {"stage": "history_residual_probability_update_oof_d1_v2_smoke", "created_at_utc": now(), "input_lock_sha256": lock_hash, "G0": {"status": "pass"}, "G1": {"status": "pass"}, "G2": {"status": "pass", "evaluation_matches": len(matches), "evaluation_rows": len(evaluation)}, "G3": {"status": "pass", "models": model_metrics}})


def fold_oof(name: str, train_sets: tuple[str, ...], evaluation_set: str, matrices: dict[str, pd.DataFrame], lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = OUT / "state" / name / "oof_predictions.parquet"
    checkpoint_path = OUT / "state" / name / "checkpoint_04_oof.json"
    if path.exists():
        checkpoint = read_json(checkpoint_path)
        require(checkpoint.get("input_lock_sha256") == lock_hash and checkpoint.get("oof_sha256") == digest(path), f"oof_checkpoint_mismatch:{name}")
        return pd.read_parquet(path), checkpoint
    train = pd.concat([matrices[item] for item in train_sets], ignore_index=True)
    evaluation = matrices[evaluation_set].copy()
    require(set(train["match_id"]).isdisjoint(set(evaluation["match_id"])), f"train_evaluation_match_overlap:{name}")
    output = evaluation[["match_id", "Y_t", "p_hat"]].copy()
    fits = {}
    for model in MODELS:
        output[model], fits[model] = fit_predict(train, evaluation, model)
    write_parquet(path, output)
    fit_path = OUT / "state" / name / "training_fit.json"
    write_json(fit_path, {"stage": "history_residual_probability_update_oof_d1_v2_training_fit", "created_at_utc": now(), "input_lock_sha256": lock_hash, "fold": name, "train_datasets": list(train_sets), "evaluation_dataset": evaluation_set, "train_rows": len(train), "train_matches": int(train["match_id"].nunique()), "evaluation_rows": len(evaluation), "evaluation_matches": int(evaluation["match_id"].nunique()), "models": fits})
    checkpoint = {"stage": "history_residual_probability_update_oof_d1_v2_oof", "created_at_utc": now(), "input_lock_sha256": lock_hash, "fold": name, "oof_sha256": digest(path), "training_fit_sha256": digest(fit_path), "G2": {"status": "pass", "train_matches": int(train["match_id"].nunique()), "evaluation_matches": int(evaluation["match_id"].nunique()), "evaluation_rows": len(evaluation)}, "G3": {"status": "pass", "reason": "all_models_converged_with_finite_probabilities"}}
    write_json(checkpoint_path, checkpoint)
    return output, checkpoint


def match_losses(oof: pd.DataFrame, fold: str) -> pd.DataFrame:
    y = oof["Y_t"].to_numpy(float)
    values: dict[str, Any] = {"match_id": oof["match_id"].astype(str)}
    for model in MODELS:
        probability = np.clip(oof[model].to_numpy(float), EPS, 1.0 - EPS)
        values[f"{model}_log_loss"] = -(y * np.log(probability) + (1.0 - y) * np.log1p(-probability))
        values[f"{model}_brier"] = (probability - y) ** 2
    output = pd.DataFrame(values).groupby("match_id", sort=True).mean().reset_index()
    output.insert(0, "fold", fold)
    return output


def compare(losses: pd.DataFrame, lock_hash: str) -> dict[str, Any]:
    path, checkpoint_path = OUT / "state" / "bootstrap_differences.npy", OUT / "checkpoint_05_bootstrap.json"
    cells = [(model, metric) for model in MODELS[1:] for metric in ("log_loss", "brier")]
    if path.exists():
        checkpoint = read_json(checkpoint_path)
        require(checkpoint.get("input_lock_sha256") == lock_hash and checkpoint.get("bootstrap_sha256") == digest(path), "bootstrap_checkpoint_mismatch")
        draws = np.load(path)
    else:
        rng = np.random.default_rng(SEED)
        partitions = {fold: frame for fold, frame in losses.groupby("fold", sort=True)}
        require(set(partitions) == set(FOLDS), "bootstrap_fold_mismatch")
        draws = np.empty((BOOTSTRAPS, len(cells)), dtype=float)
        for iteration in range(BOOTSTRAPS):
            for index, (model, metric) in enumerate(cells):
                fold_values = []
                for fold in sorted(partitions):
                    sample = partitions[fold].iloc[rng.integers(0, len(partitions[fold]), len(partitions[fold]))]
                    fold_values.append(float((sample[f"{model}_{metric}"] - sample[f"M0_base_time_{metric}"]).mean()))
                draws[iteration, index] = float(np.mean(fold_values))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, draws)
        write_json(checkpoint_path, {"stage": "history_residual_probability_update_oof_d1_v2_bootstrap", "created_at_utc": now(), "input_lock_sha256": lock_hash, "draws": BOOTSTRAPS, "seed": SEED, "columns": [f"{model}_minus_M0_{metric}" for model, metric in cells], "bootstrap_sha256": digest(path)})
    summary: dict[str, Any] = {}
    for index, (model, metric) in enumerate(cells):
        point = float(np.mean([frame[f"{model}_{metric}"].mean() - frame[f"M0_base_time_{metric}"].mean() for _, frame in losses.groupby("fold", sort=True)]))
        lo, hi = (float(value) for value in np.quantile(draws[:, index], [0.025, 0.975]))
        sd = float(np.std(draws[:, index], ddof=1))
        require(math.isfinite(sd) and sd > 0, f"invalid_bootstrap_sd:{model}:{metric}")
        value: dict[str, Any] = {"candidate_minus_M0": point, "ci95_low": lo, "ci95_high": hi, "ci95_half_width": (hi - lo) / 2.0, "bootstrap_sd": sd}
        if metric == "log_loss":
            value["two_sided_normal_p_value"] = math.erfc(abs(point / sd) / math.sqrt(2.0))
        summary[f"{model}_vs_M0_{metric}"] = value
    return summary


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
    parts, fold_checks = [], {}
    for name, (train, evaluation) in FOLDS.items():
        oof, check = fold_oof(name, train, evaluation, matrices, lock_hash)
        parts.append(match_losses(oof, name))
        fold_checks[name] = check
    loss_path = OUT / "match_losses.parquet"
    if loss_path.exists():
        losses = pd.read_parquet(loss_path)
    else:
        losses = pd.concat(parts, ignore_index=True)
        write_parquet(loss_path, losses)
    comparisons = compare(losses, lock_hash)
    gates = {"G0": {"status": "pass", "reason": "d1_only_hash_locked_inputs", "metrics": {"e1_path_access_count": 0}}, "G1": {"status": "pass", "reason": "strict_history_state_and_probability_join", "metrics": {dataset: check["G1"]["metrics"] for dataset, check in matrix_checks.items()}}, "G2": {"status": "pass", "reason": "two_time_forward_disjoint_match_folds", "metrics": {name: check["G2"] for name, check in fold_checks.items()}}, "G3": {"status": "pass", "reason": "all_logistic_models_converged_with_finite_probabilities", "metrics": {name: check["G3"] for name, check in fold_checks.items()}}, "G4": {"status": "pass", "reason": "5000_stratified_match_bootstrap_completed"}, "G5": {"status": "pending", "reason": "exploratory_result_informed_probability_update_only"}}
    comparison_path = OUT / "checkpoint_06_comparisons.json"
    if not comparison_path.exists():
        write_json(comparison_path, {"stage": "history_residual_probability_update_oof_d1_v2_comparisons", "created_at_utc": now(), "input_lock_sha256": lock_hash, "metrics": "match_equal_within_fold_then_equal_F1_F2", "comparisons": comparisons})
    gate_path = OUT / "checkpoint_07_gate_audit.json"
    if not gate_path.exists():
        write_json(gate_path, {"stage": "history_residual_probability_update_oof_d1_v2_gate_audit", "created_at_utc": now(), "input_lock_sha256": lock_hash, "gates": gates})
    hash_path = OUT / "file_sha256.json"
    if not hash_path.exists():
        write_json(hash_path, hashes())
    else:
        for relative, expected in read_json(hash_path).items():
            require((OUT / relative).is_file() and digest(OUT / relative) == expected, f"output_hash_mismatch:{relative}")
    manifest_path = OUT / "manifest.json"
    if not manifest_path.exists():
        write_json(manifest_path, {"stage": "history_residual_probability_update_oof_d1_v2", "status": "passed", "created_at_utc": now(), "classification": "exploratory_nonconfirmatory_result_informed_d1_only", "input_lock_sha256": lock_hash, "gate_statuses": {name: item["status"] for name, item in gates.items()}, "interpretation": "D1-only forward probability-update comparison of the three fixed models M0/M1/M2; not raw p_hat retraining, independent validation, hot-hand, state, causal, deployable, or confirmatory evidence.", "file_sha256_sha256": digest(hash_path)})
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
        write_json(failed / "failure_manifest.json", {"stage": "history_residual_probability_update_oof_d1_v2", "status": "failed", "created_at_utc": now(), "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
