"""Run the frozen D1-only residual baseline ladder comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "results" / "history_residual_state_oof_d1_v2"
OUT = ROOT / "results" / "history_residual_baseline_ladder_oof_d1_v1"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "45_D1历史残差分层基线样本外预测比较执行协议_v1_2026-08-26.md"
DRAFT = ROOT / "docs" / "修订与补丁" / "44_D1历史残差分层基线样本外预测比较分析协议_草案_v1_2026-08-26.md"
FOLDS = {"F1": (("D1-A",), "D1-B"), "F2": (("D1-A", "D1-B"), "D1-C")}
MODELS = ("B0_p_hat_residual_zero", "B1_time_only", "M1_lag1", "M2_decay_lag1", "M3_decay_sum")
PAIRS = (
    ("B0_vs_B1", "B0_p_hat_residual_zero", "B1_time_only"),
    ("B0_vs_M1", "B0_p_hat_residual_zero", "M1_lag1"),
    ("B1_vs_M1", "B1_time_only", "M1_lag1"),
    ("M1_vs_M2", "M1_lag1", "M2_decay_lag1"),
    ("M1_vs_M3", "M1_lag1", "M3_decay_sum"),
    ("M2_vs_M3", "M2_decay_lag1", "M3_decay_sum"),
)
BOOTSTRAPS = 2000
SEED = 20260826
DELTA = 0.005


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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


def verify_tree(root: Path) -> dict[str, Any]:
    expected = read_json(root / "file_sha256.json")
    mismatches = [
        name for name, expected_hash in expected.items()
        if not (root / name).is_file() or digest(root / name) != expected_hash
    ]
    manifest = read_json(root / "manifest.json")
    return {
        "checked_files": len(expected),
        "mismatches": mismatches,
        "manifest_hash_matches": manifest.get("file_sha256_sha256") == digest(root / "file_sha256.json"),
        "manifest_status": manifest.get("status"),
    }


def prior_audit() -> dict[str, Any]:
    audit = verify_tree(PREVIOUS)
    require(not audit["mismatches"], f"previous_output_hash_mismatch:{audit}")
    require(audit["manifest_hash_matches"] and audit["manifest_status"] == "passed", f"previous_manifest_invalid:{audit}")
    gates = read_json(PREVIOUS / "checkpoint_08_gate_audit.json")["gates"]
    require(all(gates[key]["status"] == "pass" for key in ("G0", "G1", "G2", "G3", "G4")), f"previous_gate_failure:{gates}")
    return {"tree": audit, "previous_gate_statuses": {key: gates[key]["status"] for key in gates if key.startswith("G")}}


def input_files() -> list[Path]:
    paths = [PROTOCOL, DRAFT, Path(__file__).resolve(), PREVIOUS / "manifest.json", PREVIOUS / "file_sha256.json", PREVIOUS / "checkpoint_08_gate_audit.json"]
    for dataset in ("D1-A", "D1-B", "D1-C"):
        paths.extend([PREVIOUS / "state" / dataset / "state_matrix.parquet", PREVIOUS / "state" / dataset / f"checkpoint_02_state_{dataset}.json"])
    for fold in FOLDS:
        paths.extend([PREVIOUS / "state" / fold / "oof_predictions.parquet", PREVIOUS / "state" / fold / "training_fit.json", PREVIOUS / "state" / fold / "checkpoint_04_oof.json"])
    return paths


def input_lock() -> tuple[dict[str, Any], str]:
    audit = prior_audit()
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in input_files()}
    path = OUT / "checkpoint_01_input_lock.json"
    if path.exists():
        saved = read_json(path)
        require(saved.get("inputs") == inputs, "input_lock_source_hash_changed")
        return saved, digest(path)
    value = {
        "stage": "history_residual_baseline_ladder_oof_d1_v1_input_lock",
        "created_at_utc": now(),
        "classification": "exploratory_nonconfirmatory_result_informed_d1_only",
        "inputs": inputs,
        "previous_audit": audit,
        "e1_path_access_count": 0,
    }
    write_json_new(path, value)
    return value, digest(path)


def load_matrix(dataset: str) -> pd.DataFrame:
    path = PREVIOUS / "state" / dataset / "state_matrix.parquet"
    frame = pd.read_parquet(path, columns=["dataset", "match_id", "focal_player_id", "e_t", "H_t", "time_since_history_tick"])
    require(np.isfinite(frame[["e_t", "H_t", "time_since_history_tick"]].to_numpy(float)).all(), f"nonfinite_matrix:{dataset}")
    return frame


def fit_b1(train: pd.DataFrame, evaluation: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    log_train = np.log2(1.0 + train["time_since_history_tick"].to_numpy(float))
    mu = float(log_train.mean())
    x_train = np.column_stack([np.ones(len(train)), log_train - mu])
    require(np.linalg.matrix_rank(x_train) == x_train.shape[1], "b1_rank_deficient")
    inverse = np.linalg.inv(x_train.T @ x_train)
    coefficients = inverse @ x_train.T @ train["e_t"].to_numpy(float)
    leverage = np.einsum("ij,jk,ik->i", x_train, inverse, x_train)
    x_eval = np.column_stack([np.ones(len(evaluation)), np.log2(1.0 + evaluation["time_since_history_tick"].to_numpy(float)) - mu])
    prediction = x_eval @ coefficients
    require(np.isfinite(prediction).all(), "b1_nonfinite_prediction")
    return prediction, {"mu_train": mu, "coefficients": coefficients.tolist(), "rank": int(x_train.shape[1]), "max_leverage": float(leverage.max())}


def reference_predictions(fold: str, evaluation: pd.DataFrame) -> pd.DataFrame:
    path = PREVIOUS / "state" / fold / "oof_predictions.parquet"
    reference = pd.read_parquet(path, columns=["dataset", "match_id", "focal_player_id", "e_t", "M1_lag1", "M2_decay_lag1", "M3_decay_sum"])
    require(len(reference) == len(evaluation), f"reference_row_count_mismatch:{fold}")
    for column in ("dataset", "match_id", "focal_player_id"):
        require((reference[column].astype(str).to_numpy() == evaluation[column].astype(str).to_numpy()).all(), f"reference_key_mismatch:{fold}:{column}")
    require(np.allclose(reference["e_t"].to_numpy(float), evaluation["e_t"].to_numpy(float), atol=1e-12, rtol=0), f"reference_response_mismatch:{fold}")
    require(np.isfinite(reference[["M1_lag1", "M2_decay_lag1", "M3_decay_sum"]].to_numpy(float)).all(), f"reference_nonfinite:{fold}")
    return reference


def smoke(matrices: dict[str, pd.DataFrame], lock_hash: str) -> None:
    path = OUT / "checkpoint_02_smoke.json"
    if path.exists():
        saved = read_json(path)
        require(saved.get("input_lock_sha256") == lock_hash, "smoke_input_lock_mismatch")
        return
    train = matrices["D1-A"]
    full = matrices["D1-B"]
    matches = sorted(full["match_id"].astype(str).unique())[:30]
    evaluation = full.loc[full["match_id"].astype(str).isin(matches)].copy()
    reference = reference_predictions("F1", full)
    reference = reference.loc[reference["match_id"].astype(str).isin(matches)].reset_index(drop=True)
    evaluation = evaluation.reset_index(drop=True)
    prediction, fitted = fit_b1(train, evaluation)
    require(len(reference) == len(evaluation) and np.isfinite(prediction).all(), "smoke_alignment_failure")
    require(fitted["max_leverage"] <= 0.20, "smoke_high_leverage")
    write_json_new(path, {"stage": "history_residual_baseline_ladder_oof_d1_v1_smoke", "created_at_utc": now(), "input_lock_sha256": lock_hash, "G0": {"status": "pass"}, "G1": {"status": "pass"}, "G2": {"status": "pass", "evaluation_matches": len(matches), "evaluation_rows": len(evaluation)}, "G3": {"status": "pass", "b1_max_leverage": fitted["max_leverage"]}})


def fold_oof(name: str, train_sets: tuple[str, ...], evaluation_set: str, matrices: dict[str, pd.DataFrame], lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = OUT / "state" / name / "oof_predictions.parquet"
    checkpoint_path = OUT / "state" / name / "checkpoint_03_oof.json"
    if path.exists():
        checkpoint = read_json(checkpoint_path)
        require(checkpoint.get("input_lock_sha256") == lock_hash and checkpoint.get("oof_sha256") == digest(path), f"oof_checkpoint_mismatch:{name}")
        return pd.read_parquet(path), checkpoint
    train = pd.concat([matrices[item] for item in train_sets], ignore_index=True)
    evaluation = matrices[evaluation_set].copy()
    require(set(train["match_id"].astype(str)).isdisjoint(set(evaluation["match_id"].astype(str))), f"train_evaluation_overlap:{name}")
    reference = reference_predictions(name, evaluation)
    prediction, fitted = fit_b1(train, evaluation)
    require(fitted["max_leverage"] <= 0.20, f"high_leverage:{name}")
    output = reference[["dataset", "match_id", "focal_player_id", "e_t", "M1_lag1", "M2_decay_lag1", "M3_decay_sum"]].copy()
    output.insert(4, "B0_p_hat_residual_zero", 0.0)
    output.insert(5, "B1_time_only", prediction)
    require(np.isfinite(output[list(MODELS)].to_numpy(float)).all(), f"nonfinite_output:{name}")
    write_parquet_new(path, output)
    training_path = OUT / "state" / name / "training_fit.json"
    write_json_new(training_path, {"stage": "history_residual_baseline_ladder_oof_d1_v1_training_fit", "fold": name, "created_at_utc": now(), "input_lock_sha256": lock_hash, "train_datasets": list(train_sets), "evaluation_dataset": evaluation_set, "train_rows": len(train), "evaluation_rows": len(evaluation), "training_matches": int(train["match_id"].nunique()), "evaluation_matches": int(evaluation["match_id"].nunique()), "evaluation_players": int(evaluation["focal_player_id"].nunique()), "B1": fitted, "reference_oof_sha256": digest(PREVIOUS / "state" / name / "oof_predictions.parquet")})
    checkpoint = {"stage": "history_residual_baseline_ladder_oof_d1_v1_oof", "fold": name, "created_at_utc": now(), "input_lock_sha256": lock_hash, "oof_sha256": digest(path), "training_fit_sha256": digest(training_path), "B0_all_zero": bool((output["B0_p_hat_residual_zero"] == 0).all()), "B1": fitted, "G2": {"status": "pass", "evaluation_matches": int(evaluation["match_id"].nunique()), "evaluation_players": int(evaluation["focal_player_id"].nunique()), "evaluation_rows": len(evaluation)}, "G3": {"status": "pass", "b1_max_leverage": fitted["max_leverage"]}}
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
    means = losses.groupby("fold", sort=True)[[baseline, candidate]].mean()
    baseline_loss = float(means[baseline].mean())
    candidate_loss = float(means[candidate].mean())
    return (baseline_loss - candidate_loss) / baseline_loss


def holm(raw: dict[str, float]) -> dict[str, float]:
    adjusted: dict[str, float] = {}
    previous = 0.0
    for position, (value, name) in enumerate(sorted((value, name) for name, value in raw.items())):
        previous = max(previous, min(1.0, (len(raw) - position) * value))
        adjusted[name] = previous
    return adjusted


def comparisons(losses: pd.DataFrame, lock_hash: str) -> dict[str, Any]:
    draw_path = OUT / "state" / "bootstrap_gains.npy"
    checkpoint_path = OUT / "checkpoint_05_bootstrap.json"
    if draw_path.exists():
        checkpoint = read_json(checkpoint_path)
        require(checkpoint.get("input_lock_sha256") == lock_hash and checkpoint.get("bootstrap_sha256") == digest(draw_path), "bootstrap_checkpoint_mismatch")
        draws = np.load(draw_path)
    else:
        rng = np.random.default_rng(SEED)
        by_fold = {fold: part[list(MODELS)].to_numpy(float) for fold, part in losses.groupby("fold", sort=True)}
        require(set(by_fold) == set(FOLDS), "bootstrap_fold_mismatch")
        draws = np.empty((BOOTSTRAPS, len(PAIRS)), dtype=float)
        for index in range(BOOTSTRAPS):
            fold_means = []
            for fold in sorted(by_fold):
                values = by_fold[fold]
                fold_means.append(values[rng.integers(0, len(values), len(values))].mean(axis=0))
            combined = np.mean(fold_means, axis=0)
            indexed = dict(zip(MODELS, combined))
            for position, (_, baseline, candidate) in enumerate(PAIRS):
                draws[index, position] = (indexed[baseline] - indexed[candidate]) / indexed[baseline]
        draw_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(draw_path, draws)
        write_json_new(checkpoint_path, {"stage": "history_residual_baseline_ladder_oof_d1_v1_bootstrap", "created_at_utc": now(), "input_lock_sha256": lock_hash, "seed": SEED, "draws": BOOTSTRAPS, "bootstrap_sha256": digest(draw_path)})
    raw: dict[str, float] = {}
    output: dict[str, Any] = {}
    for position, (name, baseline, candidate) in enumerate(PAIRS):
        point = relative_gain(losses, baseline, candidate)
        lo, hi = (float(value) for value in np.quantile(draws[:, position], [0.025, 0.975]))
        sd = float(np.std(draws[:, position], ddof=1))
        require(sd > 0 and math.isfinite(sd), f"invalid_bootstrap_sd:{name}")
        raw[name] = math.erfc(abs(point / sd) / math.sqrt(2.0))
        precision = (hi - lo) / 2.0 <= 0.010
        if point >= DELTA and lo >= DELTA:
            decision = "exploratory_substantive_improvement"
        elif point > 0 and lo > 0 and hi < DELTA:
            decision = "exploratory_detectable_below_threshold"
        elif point < 0 and hi < 0:
            decision = "exploratory_worsening"
        else:
            decision = "pending"
        output[name] = {"baseline": baseline, "candidate": candidate, "relative_gain": point, "ci95_low": lo, "ci95_high": hi, "ci95_half_width": (hi - lo) / 2.0, "bootstrap_sd": sd, "raw_p_value": raw[name], "G4": {"status": "pass" if precision else "pending", "threshold": 0.010}}
        output[name]["provisional_decision_before_holm"] = decision
    adjusted = holm(raw)
    for name, value in output.items():
        value["holm_adjusted_p_value_six_pair_family"] = adjusted[name]
        value["decision"] = value["provisional_decision_before_holm"] if value["G4"]["status"] == "pass" and adjusted[name] < 0.05 else "pending"
    return output


def output_hashes() -> dict[str, str]:
    return {str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"} and not any(part.startswith("failed_") for part in path.relative_to(OUT).parts)}


def run(phase: str) -> None:
    _, lock_hash = input_lock()
    matrices = {dataset: load_matrix(dataset) for dataset in ("D1-A", "D1-B", "D1-C")}
    smoke(matrices, lock_hash)
    if phase == "smoke":
        print(json.dumps({"status": "smoke_passed", "output_root": str(OUT)}, ensure_ascii=True))
        return
    parts, fold_checks = [], {}
    for name, (train_sets, evaluation_set) in FOLDS.items():
        oof, check = fold_oof(name, train_sets, evaluation_set, matrices, lock_hash)
        parts.append(match_losses(oof, name))
        fold_checks[name] = check
    losses = pd.concat(parts, ignore_index=True)
    loss_path = OUT / "match_losses.parquet"
    if loss_path.exists():
        require(digest(loss_path) == read_json(OUT / "file_sha256.json").get("match_losses.parquet", digest(loss_path)), "loss_hash_mismatch")
        losses = pd.read_parquet(loss_path)
    else:
        write_parquet_new(loss_path, losses)
    comparison = comparisons(losses, lock_hash)
    gates = {"G0": {"status": "pass", "reason": "d1_only_previous_hash_chain_locked", "metrics": {"e1_path_access_count": 0}}, "G1": {"status": "pass", "reason": "zero_baseline_and_previous_state_audits_verified"}, "G2": {"status": "pass", "reason": "forward_fold_support_and_shared_rows", "metrics": {name: check["G2"] for name, check in fold_checks.items()}}, "G3": {"status": "pass", "reason": "b1_full_rank_finite_and_leverage_at_most_0p20", "metrics": {name: check["G3"] for name, check in fold_checks.items()}}, "G4": {"status": "pass" if all(value["G4"]["status"] == "pass" for value in comparison.values()) else "pending", "reason": "six_pair_bootstrap_precision", "comparisons": {name: value["G4"] for name, value in comparison.items()}}, "G5": {"status": "pending", "reason": "exploratory_result_informed_residual_prediction_only"}}
    summary_path = OUT / "checkpoint_06_comparisons.json"
    if not summary_path.exists():
        write_json_new(summary_path, {"stage": "history_residual_baseline_ladder_oof_d1_v1_comparisons", "created_at_utc": now(), "input_lock_sha256": lock_hash, "delta": DELTA, "inference": "2000_stratified_match_bootstrap_percentile_ci_normal_p_holm_six_pair_family", "comparisons": comparison, "folds": fold_checks})
    gate_path = OUT / "checkpoint_07_gate_audit.json"
    if not gate_path.exists():
        write_json_new(gate_path, {"stage": "history_residual_baseline_ladder_oof_d1_v1_gates", "created_at_utc": now(), "input_lock_sha256": lock_hash, "gates": gates})
    hashes_path = OUT / "file_sha256.json"
    if not hashes_path.exists():
        write_json_new(hashes_path, output_hashes())
    else:
        for relative, expected in read_json(hashes_path).items():
            require((OUT / relative).is_file() and digest(OUT / relative) == expected, f"output_hash_mismatch:{relative}")
    manifest_path = OUT / "manifest.json"
    if not manifest_path.exists():
        write_json_new(manifest_path, {"stage": "history_residual_baseline_ladder_oof_d1_v1", "status": "passed", "created_at_utc": now(), "classification": "exploratory_nonconfirmatory_result_informed_d1_only", "input_lock_sha256": lock_hash, "gate_statuses": {name: value["status"] for name, value in gates.items()}, "interpretation": "D1-only residual baseline ladder; exploratory observed-sequence Pearson-residual loss comparison only, not p_hat probability performance, hot-hand, state, causal, or confirmatory evidence.", "file_sha256_sha256": digest(hashes_path)})
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
        write_json_new(failed / "failure_manifest.json", {"stage": "history_residual_baseline_ladder_oof_d1_v1", "status": "failed", "created_at_utc": now(), "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
