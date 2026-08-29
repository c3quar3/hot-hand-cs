"""Read-only v2 validation of saved D1-A p_hat OOF predictions."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "main"
PROTOCOL = MAIN / "docs" / "协议" / "04_D1A_p_hat内部验证范围差异与执行协议_v2_2026-08-25.md"
MAPPING = MAIN / "data" / "d1a_internal_validation_input_mapping_v1.json"
INPUT_LOCK = MAIN / "data" / "d1a_c_selection_input_lock_v1.json"
TRAINING = MAIN / "results" / "d1a_c_selection_training_v1"
V1_MANIFEST = MAIN / "results" / "d1a_internal_validation_v1" / "manifest.json"
OUTPUT = MAIN / "results" / "d1a_internal_validation_v2"
EPS = 1e-15
SEED = 20260825
BOOTSTRAPS = 2000
FOLDS = (("fold_1", 101, 150), ("fold_2", 151, 200), ("fold_3", 201, 250), ("fold_4", 251, 300))


class ValidationFailure(RuntimeError):
    pass


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationFailure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise ValidationFailure(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def fit_calibration(x: np.ndarray, y: np.ndarray, weights: np.ndarray, start: np.ndarray | None = None) -> np.ndarray:
    beta = np.zeros(2) if start is None else start.copy()
    for _ in range(100):
        probability = 1 / (1 + np.exp(-np.clip(beta[0] + beta[1] * x, -35, 35)))
        residual = weights * (y - probability)
        curvature = weights * np.maximum(probability * (1 - probability), 1e-9)
        h00, h01, h11 = curvature.sum(), np.dot(curvature, x), np.dot(curvature, x * x)
        gradient = np.array((residual.sum(), np.dot(residual, x)))
        step = np.linalg.solve(np.array(((h00, h01), (h01, h11))), gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            return beta
    raise ValidationFailure("calibration_newton_nonconvergence")


def point_metrics(frame: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probability = np.clip(frame["p_hat"].to_numpy(float), EPS, 1 - EPS)
    outcome = frame["focal_wins"].to_numpy(float)
    logit = np.log(probability / (1 - probability))
    loss = -(outcome * np.log(probability) + (1 - outcome) * np.log(1 - probability))
    brier = (outcome - probability) ** 2
    groups = pd.qcut(pd.Series(probability), q=10, duplicates="drop")
    codes = groups.cat.codes.to_numpy(int)
    bins = pd.DataFrame({"bin": groups, "p_hat": probability, "focal_wins": outcome}).groupby("bin", observed=True).agg(p_hat=("p_hat", "mean"), focal_wins=("focal_wins", "mean"), rows=("focal_wins", "size")).reset_index(drop=True)
    calibration = fit_calibration(logit, outcome, np.ones(len(frame)))
    ece = float((bins["rows"] * (bins["p_hat"] - bins["focal_wins"]).abs()).sum() / len(frame))
    by_match = pd.DataFrame({"match_id": frame["match_id"], "loss": loss, "brier": brier}).groupby("match_id", sort=True).mean()
    result = {"match_equal_log_loss": float(by_match["loss"].mean()), "match_equal_brier": float(by_match["brier"].mean()), "row_brier": float(brier.mean()), "calibration_intercept": float(calibration[0]), "calibration_slope": float(calibration[1]), "calibration_ece_10_bin": ece, "calibration_bin_count": int(len(bins))}
    return result, bins, loss, brier, logit, codes


def bootstrap(frame: pd.DataFrame, loss: np.ndarray, brier: np.ndarray, logit: np.ndarray, bins: np.ndarray, calibration_start: np.ndarray) -> pd.DataFrame:
    _, match_code = np.unique(frame["match_id"].to_numpy(), return_inverse=True)
    match_count = int(match_code.max()) + 1
    row_count = np.bincount(match_code, minlength=match_count).astype(float)
    mean_loss = np.bincount(match_code, weights=loss, minlength=match_count) / row_count
    mean_brier = np.bincount(match_code, weights=brier, minlength=match_count) / row_count
    total_brier = np.bincount(match_code, weights=brier, minlength=match_count)
    bin_count = int(bins.max()) + 1
    bin_rows = np.zeros((match_count, bin_count))
    bin_probability = np.zeros((match_count, bin_count))
    bin_outcome = np.zeros((match_count, bin_count))
    probability = frame["p_hat"].to_numpy(float)
    outcome = frame["focal_wins"].to_numpy(float)
    for code in range(bin_count):
        member = bins == code
        bin_rows[:, code] = np.bincount(match_code[member], minlength=match_count)
        bin_probability[:, code] = np.bincount(match_code[member], weights=probability[member], minlength=match_count)
        bin_outcome[:, code] = np.bincount(match_code[member], weights=outcome[member], minlength=match_count)
    rng = np.random.default_rng(SEED)
    results: list[dict[str, float | int]] = []
    for replicate in range(1, BOOTSTRAPS + 1):
        counts = np.bincount(rng.integers(0, match_count, size=match_count), minlength=match_count).astype(float)
        row_weights = counts[match_code]
        calibration = fit_calibration(logit, outcome, row_weights, calibration_start)
        rows_by_bin = counts @ bin_rows
        probability_by_bin = counts @ bin_probability
        outcome_by_bin = counts @ bin_outcome
        expected = probability_by_bin / rows_by_bin
        observed = outcome_by_bin / rows_by_bin
        ece = float((rows_by_bin * np.abs(expected - observed)).sum() / rows_by_bin.sum())
        results.append({"replicate": replicate, "match_equal_log_loss": float(np.dot(counts, mean_loss) / match_count), "match_equal_brier": float(np.dot(counts, mean_brier) / match_count), "row_brier": float(np.dot(counts, total_brier) / np.dot(counts, row_count)), "calibration_intercept": float(calibration[0]), "calibration_slope": float(calibration[1]), "calibration_ece_10_bin": ece})
    return pd.DataFrame(results)


def expected_fold(queue: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=queue.index, dtype="string")
    for name, start, end in FOLDS:
        result.loc[queue.between(start, end)] = name
    return result


def run() -> dict[str, Any]:
    ensure(not OUTPUT.exists(), f"output_exists:{OUTPUT}")
    required_active = (PROTOCOL, MAPPING, INPUT_LOCK, TRAINING / "manifest.json", TRAINING / "selected_c_oof.parquet", TRAINING / "full_model.joblib", V1_MANIFEST)
    for path in required_active:
        ensure(path.is_file(), f"missing_active_input:{path}")
    mapping, input_lock, training, v1 = read_json(MAPPING), read_json(INPUT_LOCK), read_json(TRAINING / "manifest.json"), read_json(V1_MANIFEST)
    ensure(v1.get("status") == "failed" and v1.get("error") == "ValidationFailure:candidate_queue_scope_mismatch", "v1_failure_record_mismatch")
    ensure(mapping.get("outcome_access") == "D1-A_previously_authorized_only", "invalid_outcome_access")
    ensure(training.get("selected_c") == 0.01, "unexpected_selected_c")
    mapped: dict[str, Path] = {}
    for key, lock_key in (("candidate", "candidate_sha256"), ("candidate_audit", "candidate_audit_sha256"), ("source_manifest", "source_lock_sha256")):
        item = mapping["mapping"][key]
        path = Path(item["historical_read_only_path"])
        ensure(path.is_file(), f"mapped_input_missing:{key}")
        ensure(sha256(path) == item["sha256"] == input_lock[lock_key], f"mapped_hash_mismatch:{key}")
        mapped[key] = path
    active_hashes = {"protocol": sha256(PROTOCOL), "mapping": sha256(MAPPING), "training_input_lock": sha256(INPUT_LOCK), "training_manifest": sha256(TRAINING / "manifest.json"), "selected_c_oof": sha256(TRAINING / "selected_c_oof.parquet"), "full_model": sha256(TRAINING / "full_model.joblib"), "v1_failure_manifest": sha256(V1_MANIFEST), "script": sha256(Path(__file__).resolve())}
    OUTPUT.mkdir(parents=True)
    write_new_json(OUTPUT / "input_lock.json", {"stage": "d1a_p_hat_internal_validation_v2_input_lock", "created_at_utc": datetime.now(UTC).isoformat(), "outcome_access": "D1-A_previously_authorized_only", "input_hashes": active_hashes, "mapped_historical_inputs": {key: {"path": str(path), "sha256": sha256(path)} for key, path in mapped.items()}, "reconciliation": {"input_lock_declared_queue_gap": input_lock["queue_gap"], "actual_expected_queue_orders": [1, 300], "actual_gap": [], "reason": "v1 demonstrated that queue_order=129 is present in both the hash-equivalent candidate and saved fold_1 OOF."}})
    for name, value in training["files"].items():
        path = TRAINING / name
        ensure(path.is_file() and sha256(path) == value, f"training_artifact_hash_mismatch:{name}")
    candidate_audit, source_manifest = read_json(mapped["candidate_audit"]), read_json(mapped["source_manifest"])
    ensure(candidate_audit.get("stage") == "p_hat_final_v8_position_continuity_candidate", "unexpected_candidate_audit_stage")
    ensure(candidate_audit.get("source_lock_sha256") == input_lock["source_lock_sha256"], "candidate_audit_source_lock_mismatch")
    ensure(source_manifest.get("stage") == "p_hat_final_v8_position_continuity_source_lock", "unexpected_source_manifest_stage")
    numeric = input_lock["feature_contract"]["numeric"]
    categorical = input_lock["feature_contract"]["categorical"]
    required_columns = list(dict.fromkeys(("duel_id", "match_id", "map_id", "queue_order", "partition", "focal_wins", "event_tick", "anchor_tick", *numeric, *categorical)))
    candidate = pd.read_parquet(mapped["candidate"], columns=required_columns)
    candidate = candidate.loc[candidate["partition"].eq("D1-A")].copy()
    ensure(len(candidate) == input_lock["rows"] == 86685, "candidate_row_count_mismatch")
    ensure(candidate["duel_id"].is_unique, "candidate_duplicate_duel_id")
    ensure(candidate.groupby("match_id")["queue_order"].nunique().eq(1).all(), "candidate_match_queue_nonunique")
    ensure(candidate["match_id"].nunique() == input_lock["matches"] == 300, "candidate_match_count_mismatch")
    ensure(set(candidate["queue_order"].astype(int).unique()) == set(range(1, 301)), "candidate_queue_scope_mismatch_v2")
    ensure(set(pd.to_numeric(candidate["focal_wins"], errors="raise").unique()).issubset({0, 1}), "candidate_outcome_not_binary")
    ensure((pd.to_numeric(candidate["anchor_tick"], errors="raise") < pd.to_numeric(candidate["event_tick"], errors="raise")).all(), "anchor_not_before_event")
    for column in numeric:
        ensure(np.isfinite(pd.to_numeric(candidate[column], errors="raise").to_numpy(float)).all(), f"candidate_nonfinite_numeric:{column}")
    for column in categorical:
        values = candidate[column].astype("string")
        ensure(not values.isna().any() and not values.str.strip().eq("").any(), f"candidate_missing_categorical:{column}")
    oof = pd.read_parquet(TRAINING / "selected_c_oof.parquet")
    required_oof = {"duel_id", "match_id", "map_id", "queue_order", "focal_wins", "p_hat", "c", "fold"}
    ensure(set(oof.columns) == required_oof, "oof_schema_mismatch")
    ensure(oof["duel_id"].is_unique and oof["c"].eq(0.01).all(), "oof_key_or_c_mismatch")
    ensure(np.isfinite(oof["p_hat"].to_numpy(float)).all() and oof["p_hat"].between(0, 1, inclusive="neither").all(), "oof_invalid_probability")
    expected = expected_fold(pd.to_numeric(oof["queue_order"], errors="raise"))
    ensure(expected.notna().all() and oof["fold"].astype("string").eq(expected).all(), "oof_fold_assignment_mismatch")
    ensure(oof.groupby("match_id")["fold"].nunique().eq(1).all(), "oof_match_crosses_fold")
    validation_candidate = candidate.loc[candidate["queue_order"].between(101, 300), ["duel_id", "match_id", "map_id", "queue_order", "focal_wins"]]
    merged = oof.merge(validation_candidate, on="duel_id", how="outer", suffixes=("_oof", "_candidate"), indicator=True, validate="one_to_one")
    ensure(merged["_merge"].eq("both").all(), "oof_candidate_coverage_mismatch")
    for column in ("match_id", "map_id", "queue_order", "focal_wins"):
        ensure(merged[f"{column}_oof"].eq(merged[f"{column}_candidate"]).all(), f"oof_candidate_value_mismatch:{column}")
    points, calibration_bins, loss, brier, logit, bin_codes = point_metrics(oof)
    for key in ("match_equal_log_loss", "match_equal_brier"):
        ensure(abs(points[key] - training["selected_oof_metrics"][key]) <= 1e-12, f"oof_metric_reproduction_mismatch:{key}")
    calibration_start = np.array((points["calibration_intercept"], points["calibration_slope"]))
    boot = bootstrap(oof, loss, brier, logit, bin_codes, calibration_start)
    quantiles = {column: {str(q): float(boot[column].quantile(q)) for q in (0.025, 0.5, 0.975)} for column in boot.columns if column != "replicate"}
    write_new_json(OUTPUT / "structural_audit.json", {"status": "passed", "candidate_rows": int(len(candidate)), "candidate_matches": int(candidate["match_id"].nunique()), "candidate_queue_orders": [int(candidate["queue_order"].min()), int(candidate["queue_order"].max())], "candidate_queue_gap": [], "input_lock_declared_queue_gap": input_lock["queue_gap"], "oof_rows": int(len(oof)), "oof_matches": int(oof["match_id"].nunique()), "oof_fold_rows": oof.groupby("fold", sort=True).size().to_dict(), "oof_fold_matches": oof.groupby("fold", sort=True)["match_id"].nunique().to_dict(), "queue_129": {"rows": int(oof.loc[oof["queue_order"].eq(129)].shape[0]), "matches": int(oof.loc[oof["queue_order"].eq(129), "match_id"].nunique()), "fold": "fold_1"}})
    write_new_json(OUTPUT / "temporal_feature_audit.json", {"status": "passed_with_scope_limit", "verified": ["anchor_tick_strictly_before_event_tick_for_all_D1A_rows", "candidate_audit_and_source_manifest_hashes_match_training_input_lock", "candidate_audit_records_nonoutcome_position_continuity_exclusions"], "not_independently_reconstructed": ["all_raw_event_and_state_fields", "same_map_round_post_buy_economy_snapshot_semantics"], "interpretation": "Static provenance and anchor-order audit only; no historical candidate reconstruction was run."})
    write_new_json(OUTPUT / "metrics.json", {"status": "passed", "rows": int(len(oof)), "matches": int(oof["match_id"].nunique()), **points, "bootstrap_replicates": BOOTSTRAPS, "bootstrap_match_cluster_quantiles": quantiles})
    calibration_bins.to_parquet(OUTPUT / "calibration_bins.parquet", index=False, compression="zstd")
    boot.to_parquet(OUTPUT / "match_cluster_bootstrap.parquet", index=False, compression="zstd")
    files = {path.name: sha256(path) for path in OUTPUT.iterdir() if path.is_file()}
    write_new_json(OUTPUT / "manifest.json", {"stage": "d1a_p_hat_internal_validation_v2", "created_at_utc": datetime.now(UTC).isoformat(), "status": "passed", "outcome_access": "D1-A_previously_authorized_only", "input_hashes": active_hashes, "output_hashes": files, "random_seed": SEED, "bootstrap_replicates": BOOTSTRAPS, "python": sys.version, "platform": platform.platform(), "interpretation": "Post-selection D1-A internal diagnostic only; not a holdout, external replication, hot-hand, state, information-flow, or causal result."})
    return {"status": "passed", "rows": int(len(oof)), "matches": int(oof["match_id"].nunique()), **points}


def main() -> None:
    try:
        report = run()
    except Exception as error:
        if not OUTPUT.exists():
            OUTPUT.mkdir(parents=True)
        if not (OUTPUT / "manifest.json").exists():
            write_new_json(OUTPUT / "manifest.json", {"stage": "d1a_p_hat_internal_validation_v2", "created_at_utc": datetime.now(UTC).isoformat(), "status": "failed", "outcome_access": "D1-A_previously_authorized_only", "error": f"{type(error).__name__}:{error}"})
        raise
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
