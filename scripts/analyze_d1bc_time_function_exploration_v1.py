"""Run the versioned D1-B/C exploratory state-slope function analysis."""

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
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "d1bc_time_function_exploration_v1"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "65_D1BC_60秒状态时间函数探索分析协议_v1_2026-08-28.md"
SOURCE_ROOT = ROOT / "results" / "exponential_history_state_sequence_v1"
DATASETS = ("D1-B", "D1-C")
REFERENCE_DATASET = "D1-A"
Z95 = 1.959963984540054
NORMAL = NormalDist()
CURVE_POINTS = 301


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_new(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    frame.to_csv(path, index=False, float_format="%.12g")


def state_path(dataset: str) -> Path:
    return SOURCE_ROOT / "state" / dataset / "state_matrix.parquet"


def preflight_path(dataset: str) -> Path:
    return SOURCE_ROOT / "state" / dataset / "checkpoint_02_preflight.json"


def source_paths() -> list[Path]:
    paths = [PROTOCOL, Path(__file__), SOURCE_ROOT / "reference" / "D1-A_reference.json"]
    for dataset in (REFERENCE_DATASET, *DATASETS):
        paths.extend([state_path(dataset), preflight_path(dataset)])
    return paths


def expected_lock() -> dict[str, Any]:
    return {
        "stage": "d1bc_time_function_exploration_v1_input_lock",
        "classification": "user_authorized_exploratory_function_form_analysis_on_historically_outcome_exposed_D1B_D1C",
        "datasets": list(DATASETS),
        "reference_dataset": REFERENCE_DATASET,
        "model": "OLS e_t ~ 1 + z + h1 + h2 + h3 + S_star + S_star:z + S_star:h1 + S_star:h2 + S_star:h3; h_j=max(0,z-q_j)",
        "inference": "two_way_CRV1(match_id,player_id)_with_existing_finite_sample_corrections; descriptive_3df_Wald_for_state_hinges",
        "inputs": {str(path.relative_to(ROOT)): digest(path) for path in source_paths()},
    }


def ensure_lock() -> tuple[dict[str, Any], str]:
    path = OUT / "checkpoint_01_input_lock.json"
    expected = expected_lock()
    if path.exists():
        saved = read_json(path)
        require(all(saved.get(key) == value for key, value in expected.items()), "input_lock_mismatch")
        return saved, digest(path)
    value = {**expected, "created_at_utc": now()}
    write_json_new(path, value)
    return value, digest(path)


def read_state(dataset: str) -> pd.DataFrame:
    frame = pd.read_parquet(state_path(dataset))
    required = {"match_id", "player_id", "d_tick", "S_h60", "e_t"}
    require(required.issubset(frame.columns), f"missing_state_columns:{dataset}")
    numeric = frame[["d_tick", "S_h60", "e_t"]].to_numpy(dtype=float)
    require(np.isfinite(numeric).all() and (numeric[:, 0] > 0).all(), f"nonfinite_state_values:{dataset}")
    require(frame["match_id"].notna().all() and frame["player_id"].notna().all(), f"missing_clusters:{dataset}")
    return frame.copy()


def reference_and_knots(frame: pd.DataFrame) -> dict[str, Any]:
    reference = read_json(SOURCE_ROOT / "reference" / "D1-A_reference.json")
    z = np.log2(1.0 + frame["d_tick"].to_numpy(dtype=float))
    require(math.isclose(float(z.mean()), float(reference["mu_z"]), abs_tol=1e-12), "reference_mu_z_mismatch")
    require(math.isclose(float(frame["S_h60"].mean()), float(reference["mu_S"]), abs_tol=1e-12), "reference_mu_S_mismatch")
    require(math.isclose(float(frame["S_h60"].std(ddof=1)), float(reference["sigma_S_ddof1"]), abs_tol=1e-12), "reference_sigma_S_mismatch")
    centered = z - float(reference["mu_z"])
    knots = np.quantile(centered, [0.10, 0.50, 0.90])
    require(np.isfinite(knots).all() and np.all(np.diff(knots) > 0), "invalid_d1a_time_knots")
    return {
        "mu_z": float(reference["mu_z"]),
        "mu_S": float(reference["mu_S"]),
        "sigma_S_ddof1": float(reference["sigma_S_ddof1"]),
        "knots_z": [float(value) for value in knots],
        "knots_seconds": [float((2.0 ** (value + float(reference["mu_z"])) - 1.0) / 64.0) for value in knots],
    }


def model_frame(frame: pd.DataFrame, reference: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    z = np.log2(1.0 + frame["d_tick"].to_numpy(dtype=float)) - reference["mu_z"]
    state = (frame["S_h60"].to_numpy(dtype=float) - reference["mu_S"]) / reference["sigma_S_ddof1"]
    hinges = np.column_stack([np.maximum(0.0, z - knot) for knot in reference["knots_z"]])
    base = np.column_stack([np.ones(len(frame)), z, hinges, state, state * z, state[:, None] * hinges])
    names = ["intercept", "time_linear", "time_hinge_p10", "time_hinge_p50", "time_hinge_p90", "state", "state_time_linear", "state_time_hinge_p10", "state_time_hinge_p50", "state_time_hinge_p90"]
    return base, frame["e_t"].to_numpy(dtype=float), names, z


def two_way_fit(frame: pd.DataFrame, reference: dict[str, Any], expose: bool) -> dict[str, Any]:
    design, response, names, _ = model_frame(frame, reference)
    n, k = design.shape
    require(n > k and np.linalg.matrix_rank(design) == k, "rank_deficient_design")
    bread_inverse = np.linalg.inv(design.T @ design)
    coefficient = bread_inverse @ design.T @ response
    residual = response - design @ coefficient
    leverage = np.einsum("ij,jk,ik->i", design, bread_inverse, design)

    def meat(groups: pd.Series) -> tuple[np.ndarray, int]:
        _, codes = np.unique(groups.astype(str).to_numpy(), return_inverse=True)
        summed = np.zeros((int(codes.max()) + 1, k))
        np.add.at(summed, codes, design * residual[:, None])
        return summed.T @ summed, len(summed)

    match_meat, matches = meat(frame["match_id"])
    player_meat, players = meat(frame["player_id"])
    intersection_meat, intersections = meat(frame["match_id"].astype(str) + "\x1f" + frame["player_id"].astype(str))
    correction = lambda groups: (groups / (groups - 1)) * ((n - 1) / (n - k))
    covariance = bread_inverse @ (correction(matches) * match_meat + correction(players) * player_meat - correction(intersections) * intersection_meat) @ bread_inverse
    covariance = (covariance + covariance.T) / 2.0
    g3 = bool(np.isfinite(covariance).all() and (np.diag(covariance) > 0).all() and matches >= 30 and players >= 30 and intersections >= 30 and leverage.max() <= 0.20)
    result: dict[str, Any] = {"n": int(n), "k": int(k), "match_clusters": int(matches), "player_clusters": int(players), "intersection_clusters": int(intersections), "max_leverage": float(leverage.max()), "g3_pass": g3}
    if not expose or not g3:
        return result
    standard_error = np.sqrt(np.diag(covariance))
    result.update({"coefficient_names": names, "coefficients": {name: float(value) for name, value in zip(names, coefficient)}, "standard_errors": {name: float(value) for name, value in zip(names, standard_error)}, "ci95": {name: [float(value - Z95 * standard_error[index]), float(value + Z95 * standard_error[index])] for index, (name, value) in enumerate(zip(names, coefficient))}, "covariance_matrix": covariance.tolist()})
    tested = np.asarray([7, 8, 9], dtype=int)
    contrast = coefficient[tested]
    contrast_covariance = covariance[np.ix_(tested, tested)]
    require(np.linalg.matrix_rank(contrast_covariance) == 3, "noninvertible_hinge_covariance")
    wald = float(contrast @ np.linalg.inv(contrast_covariance) @ contrast)
    result["state_slope_hinge_wald"] = {"statistic": wald, "df": 3, "p_value": float(chi2.sf(wald, 3)), "tested_coefficients": [names[index] for index in tested]}
    return result


def segment_support(frame: pd.DataFrame, reference: dict[str, Any]) -> list[dict[str, Any]]:
    z = np.log2(1.0 + frame["d_tick"].to_numpy(dtype=float)) - reference["mu_z"]
    bins = [-np.inf, *reference["knots_z"], np.inf]
    labels = ["before_p10", "p10_to_p50", "p50_to_p90", "after_p90"]
    values: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        lower, upper = bins[index], bins[index + 1]
        selected = (z <= upper) if index == 0 else ((z > lower) if index == len(labels) - 1 else ((z > lower) & (z <= upper)))
        subset = frame.loc[selected]
        values.append({"segment": label, "rows": int(len(subset)), "match_clusters": int(subset["match_id"].nunique()), "player_clusters": int(subset["player_id"].nunique()), "F4_segment_pass": bool(subset["match_id"].nunique() >= 30)})
    return values


def derive_curve(model: dict[str, Any], reference: dict[str, Any], low_z: float, high_z: float, dataset: str) -> pd.DataFrame:
    z = np.linspace(low_z, high_z, CURVE_POINTS)
    hinges = np.column_stack([np.maximum(0.0, z - knot) for knot in reference["knots_z"]])
    coefficient = model["coefficients"]
    slope = coefficient["state"] + coefficient["state_time_linear"] * z + hinges @ np.asarray([coefficient["state_time_hinge_p10"], coefficient["state_time_hinge_p50"], coefficient["state_time_hinge_p90"]])
    contrast = np.column_stack([np.ones(len(z)), z, hinges])
    covariance = np.asarray(model["covariance_matrix"], dtype=float)[np.ix_([5, 6, 7, 8, 9], [5, 6, 7, 8, 9])]
    variance = np.einsum("ij,jk,ik->i", contrast, covariance, contrast)
    require(np.isfinite(variance).all() and (variance >= 0).all(), f"invalid_curve_variance:{dataset}")
    se = np.sqrt(variance)
    seconds = (np.power(2.0, z + reference["mu_z"]) - 1.0) / 64.0
    return pd.DataFrame({"dataset": dataset, "z": z, "seconds": seconds, "state_slope": slope, "standard_error": se, "ci95_low": slope - Z95 * se, "ci95_high": slope + Z95 * se})


def smoke(frames: dict[str, pd.DataFrame], reference: dict[str, Any], lock_hash: str) -> None:
    path = OUT / "checkpoint_02_smoke.json"
    state_hashes = {dataset: digest(state_path(dataset)) for dataset in DATASETS}
    if path.exists():
        saved = read_json(path)
        require(saved.get("input_lock_sha256") == lock_hash and saved.get("state_hashes") == state_hashes, "smoke_checkpoint_mismatch")
        require(saved["F2"]["status"] == "pass", "prior_smoke_failed")
        return
    metrics: dict[str, Any] = {}
    for dataset, frame in frames.items():
        smallest = sorted(frame["match_id"].astype(str).unique())[:30]
        metrics[dataset] = two_way_fit(frame.loc[frame["match_id"].astype(str).isin(smallest)].copy(), reference, expose=False)
    passed = all(value["g3_pass"] for value in metrics.values())
    write_json_new(path, {"stage": "d1bc_time_function_exploration_v1_smoke", "created_at_utc": now(), "input_lock_sha256": lock_hash, "state_hashes": state_hashes, "subset_rule": "30_lexicographically_smallest_match_id_clusters_per_dataset", "F2": {"status": "pass" if passed else "fail", "reason": "design_estimable_without_coefficients_intervals_or_p_values", "metrics": metrics}})
    require(passed, "smoke_f2_failed")


def output_hashes() -> dict[str, str]:
    return {str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"} and not any(part.startswith("failed_") for part in path.relative_to(OUT).parts)}


def run(phase: str) -> None:
    _, lock_hash = ensure_lock()
    reference_frame = read_state(REFERENCE_DATASET)
    reference = reference_and_knots(reference_frame)
    frames = {dataset: read_state(dataset) for dataset in DATASETS}
    for dataset in DATASETS:
        preflight = read_json(preflight_path(dataset))
        require(preflight.get("state_matrix_sha256") == digest(state_path(dataset)), f"state_identity_mismatch:{dataset}")
        require(all(preflight[gate]["status"] == "pass" for gate in ("G0", "G1", "G2")), f"source_preflight_failed:{dataset}")
    smoke(frames, reference, lock_hash)
    if phase == "smoke":
        print(json.dumps({"status": "smoke_passed", "output_root": str(OUT)}, ensure_ascii=True))
        return
    full_path = OUT / "checkpoint_03_full.json"
    state_hashes = {dataset: digest(state_path(dataset)) for dataset in (REFERENCE_DATASET, *DATASETS)}
    if full_path.exists():
        full = read_json(full_path)
        require(full.get("input_lock_sha256") == lock_hash and full.get("state_hashes") == state_hashes, "full_checkpoint_mismatch")
    else:
        models = {dataset: two_way_fit(frame, reference, expose=True) for dataset, frame in frames.items()}
        require(all(model["g3_pass"] for model in models.values()), "full_f3_failed")
        supports = {dataset: segment_support(frame, reference) for dataset, frame in frames.items()}
        low_z = max(float(np.quantile(np.log2(1.0 + frames[dataset]["d_tick"].to_numpy(dtype=float)) - reference["mu_z"], 0.01)) for dataset in DATASETS)
        high_z = min(float(np.quantile(np.log2(1.0 + frames[dataset]["d_tick"].to_numpy(dtype=float)) - reference["mu_z"], 0.99)) for dataset in DATASETS)
        require(low_z < high_z, "empty_common_curve_support")
        curves = pd.concat([derive_curve(models[dataset], reference, low_z, high_z, dataset) for dataset in DATASETS], ignore_index=True)
        curve_path = OUT / "curve_source_data.csv"
        write_csv_new(curve_path, curves)
        full = {"stage": "d1bc_time_function_exploration_v1_full", "created_at_utc": now(), "input_lock_sha256": lock_hash, "state_hashes": state_hashes, "reference": reference, "common_curve_support": {"low_z": low_z, "high_z": high_z, "low_seconds": float((2.0 ** (low_z + reference["mu_z"]) - 1.0) / 64.0), "high_seconds": float((2.0 ** (high_z + reference["mu_z"]) - 1.0) / 64.0), "quantiles": "within_each_partition_P1_to_P99_then_intersected"}, "models": models, "segment_support": supports, "curve_source_data_sha256": digest(curve_path), "inference": "unweighted_OLS_two_way_CRV1_match_and_player_normal_approximation; exploratory_3df_Wald_for_added_state_slope_hinges"}
        write_json_new(full_path, full)
    f3 = all(model["g3_pass"] for model in full["models"].values())
    f4 = all(item["F4_segment_pass"] for items in full["segment_support"].values() for item in items)
    gate_path = OUT / "checkpoint_04_gate_audit.json"
    if not gate_path.exists():
        write_json_new(gate_path, {"stage": "d1bc_time_function_exploration_v1_gate_audit", "created_at_utc": now(), "input_lock_sha256": lock_hash, "F0": {"status": "pass", "reason": "input_hash_chain_locked"}, "F1": {"status": "pass", "reason": "state_identity_and_d1a_knots_validated"}, "F2": {"status": "pass", "reason": "smoke_estimable_without_effect_output"}, "F3": {"status": "pass" if f3 else "fail", "reason": "full_models_estimable_with_two_way_CRV1"}, "F4": {"status": "pass" if f4 else "pending", "reason": "each_predeclared_time_segment_has_at_least_30_match_clusters", "segments": full["segment_support"]}, "F5": {"status": "assessed", "reason": "D1B_and_D1C_are_historically_outcome_exposed_exploratory_data_no_confirmatory_or_model_promotion_claim"}})
    summary_path = OUT / "checkpoint_05_summary.json"
    if not summary_path.exists():
        compact = {dataset: {"rows": model["n"], "match_clusters": model["match_clusters"], "player_clusters": model["player_clusters"], "max_leverage": model["max_leverage"], "state_slope_hinge_wald": model["state_slope_hinge_wald"]} for dataset, model in full["models"].items()}
        write_json_new(summary_path, {"stage": "d1bc_time_function_exploration_v1_summary", "created_at_utc": now(), "input_lock_sha256": lock_hash, "reference": full["reference"], "common_curve_support": full["common_curve_support"], "models": compact, "interpretation": "Exploratory D1-B discovery and same-specification D1-C cross-partition description only. The state-slope hinge Wald quantities and pointwise intervals do not change the frozen primary model, its Holm family, or any confirmatory qualification."})
    hashes_path = OUT / "file_sha256.json"
    hashes = output_hashes()
    if hashes_path.exists():
        require(read_json(hashes_path) == hashes, "output_hash_mismatch")
    else:
        write_json_new(hashes_path, hashes)
    manifest_path = OUT / "manifest.json"
    if not manifest_path.exists():
        write_json_new(manifest_path, {"stage": "d1bc_time_function_exploration_v1", "status": "passed", "created_at_utc": now(), "classification": "user_authorized_exploratory_function_form_analysis_on_historically_outcome_exposed_D1B_D1C", "input_lock_sha256": lock_hash, "gate_statuses": {"F0": "pass", "F1": "pass", "F2": "pass", "F3": "pass" if f3 else "fail", "F4": "pass" if f4 else "pending", "F5": "assessed"}, "interpretation": "Observed engagement-sequence association only; no causal, opportunity-process, latent-state, mechanism, new-confirmation, or model-promotion claim.", "file_sha256_sha256": digest(hashes_path)})
    print(json.dumps({"status": "passed", "manifest": str(manifest_path)}, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    try:
        run(args.phase)
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True)
        failed = OUT / f"failed_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        failed.mkdir(exist_ok=False)
        write_json_new(failed / "failure_manifest.json", {"stage": "d1bc_time_function_exploration_v1", "status": "failed", "created_at_utc": now(), "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
