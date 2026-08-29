"""Run the frozen D1-only continuous-time residual association analysis."""

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
SOURCE = ROOT / "results" / "first_order_residual_round_boundary_interaction_d1_v1"
OUT = ROOT / "results" / "first_order_residual_continuous_time_d1_v1"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "24_D1连续时间调节一阶历史残差关联执行协议_v1_D1_only_2026-08-26.md"
DRAFT = ROOT / "docs" / "修订与补丁" / "23_D1连续时间调节一阶历史残差关联分析协议_草案_v1_D1_only_2026-08-26.md"
DATASETS = ("D1-A", "D1-B", "D1-C")
REQUIRED_COLUMNS = {
    "e_t", "H_t", "match_id", "focal_player_id", "event_tick",
    "history_source_event_tick", "time_since_history_tick",
}
EXPECTED = {
    "D1-A": {"rows": 55141, "match_clusters": 200, "player_clusters": 273,
             "quantiles": (123.0, 6321.0, 15612.0), "min": 1.0, "max": 231735.0,
             "mu": 11.531208255768277},
    "D1-B": {"rows": 31614, "match_clusters": 100, "player_clusters": 213,
             "quantiles": (122.0, 5868.0, 15037.7), "min": 1.0, "max": 118706.0,
             "mu": 11.389632448018476},
    "D1-C": {"rows": 24139, "match_clusters": 100, "player_clusters": 166,
             "quantiles": (117.0, 5810.0, 13833.0), "min": 1.0, "max": 78782.0,
             "mu": 11.356435324127022},
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def source_paths() -> list[Path]:
    paths = [
        PROTOCOL, DRAFT, Path(__file__).resolve(), SOURCE / "manifest.json",
        SOURCE / "file_sha256.json", SOURCE / "checkpoint_04_gate_audit.json",
    ]
    for dataset in DATASETS:
        paths.extend([
            SOURCE / "state" / dataset / "analysis_matrix.parquet",
            SOURCE / "state" / dataset / f"checkpoint_03_analysis_matrix_{dataset}.json",
        ])
    return paths


def verify_source_hashes() -> dict[str, Any]:
    expected = read_json(SOURCE / "file_sha256.json")
    mismatches = [
        relative for relative, expected_hash in expected.items()
        if not (SOURCE / relative).is_file() or digest(SOURCE / relative) != expected_hash
    ]
    manifest = read_json(SOURCE / "manifest.json")
    return {
        "checked_files": len(expected),
        "mismatches": mismatches,
        "manifest_file_sha256_matches": manifest.get("file_sha256_sha256") == digest(SOURCE / "file_sha256.json"),
    }


def input_lock() -> tuple[dict[str, Any], str]:
    path = OUT / "checkpoint_01_input_lock.json"
    inputs = {str(item): digest(item) for item in source_paths()}
    source_audit = verify_source_hashes()
    require(not source_audit["mismatches"], "source_file_sha256_mismatch")
    require(source_audit["manifest_file_sha256_matches"], "source_manifest_file_sha256_mismatch")
    if path.exists():
        saved = read_json(path)
        require(saved.get("stage") == "continuous_time_d1_v1_input_lock", "invalid_input_lock_stage")
        require(saved.get("inputs") == inputs, "input_lock_source_hash_changed")
        return saved, digest(path)
    value = {
        "stage": "continuous_time_d1_v1_input_lock",
        "created_at_utc": now(),
        "classification": "exploratory_nonconfirmatory_d1_only",
        "inputs": inputs,
        "source_hash_audit": source_audit,
        "e1_path_access_count": 0,
        "r_t_model_access_count": 0,
    }
    write_json_new(path, value)
    return value, digest(path)


def summary(frame: pd.DataFrame, dataset: str) -> tuple[dict[str, Any], bool]:
    times = frame["time_since_history_tick"].to_numpy(dtype=float)
    logged = np.log2(1.0 + times)
    expected = EXPECTED[dataset]
    observed = {
        "rows": int(len(frame)),
        "match_clusters": int(frame["match_id"].astype(str).nunique()),
        "player_clusters": int(frame["focal_player_id"].astype(str).nunique()),
        "T_p10_p50_p90": [float(value) for value in np.quantile(times, [0.10, 0.50, 0.90])],
        "T_min": float(np.min(times)),
        "T_max": float(np.max(times)),
        "mu_d": float(np.mean(logged)),
    }
    matches = (
        observed["rows"] == expected["rows"]
        and observed["match_clusters"] == expected["match_clusters"]
        and observed["player_clusters"] == expected["player_clusters"]
        and np.allclose(observed["T_p10_p50_p90"], expected["quantiles"], rtol=0.0, atol=1e-9)
        and math.isclose(observed["T_min"], expected["min"], rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(observed["T_max"], expected["max"], rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(observed["mu_d"], expected["mu"], rel_tol=0.0, abs_tol=1e-12)
    )
    return observed, matches


def preflight(dataset: str, lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_path = SOURCE / "state" / dataset / "analysis_matrix.parquet"
    source_checkpoint = SOURCE / "state" / dataset / f"checkpoint_03_analysis_matrix_{dataset}.json"
    checkpoint = read_json(source_checkpoint)
    require(checkpoint.get("matrix_sha256") == digest(source_path), f"source_matrix_hash_mismatch:{dataset}")
    require(int(checkpoint.get("rows", -1)) == EXPECTED[dataset]["rows"], f"source_matrix_row_lock_mismatch:{dataset}")
    frame = pd.read_parquet(source_path, columns=sorted(REQUIRED_COLUMNS))
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    require(not missing, f"missing_required_columns:{dataset}:{','.join(missing)}")

    finite_columns = ["e_t", "H_t", "event_tick", "history_source_event_tick", "time_since_history_tick"]
    nonfinite = int((~np.isfinite(frame[finite_columns].to_numpy(dtype=float))).sum())
    invalid_time = int((frame["time_since_history_tick"].to_numpy(dtype=float) < 1).sum())
    strict_time_violations = int((
        frame["history_source_event_tick"].to_numpy(dtype=float)
        >= frame["event_tick"].to_numpy(dtype=float)
    ).sum())
    time_summary, summary_matches = summary(frame, dataset)
    g0_audit = verify_source_hashes()
    g0_pass = not g0_audit["mismatches"] and g0_audit["manifest_file_sha256_matches"]
    g1_pass = bool(nonfinite == 0 and invalid_time == 0 and strict_time_violations == 0 and summary_matches)
    g2_pass = bool(
        time_summary["match_clusters"] >= 30 and time_summary["player_clusters"] >= 30
    )
    value = {
        "stage": "continuous_time_d1_v1_preflight",
        "dataset": dataset,
        "created_at_utc": now(),
        "input_lock_sha256": lock_hash,
        "source_matrix_sha256": digest(source_path),
        "metrics": {
            "missing_required_columns": missing,
            "nonfinite_required_values": nonfinite,
            "T_less_than_one": invalid_time,
            "strict_time_violations": strict_time_violations,
            "time_summary": time_summary,
            "expected_time_summary": EXPECTED[dataset],
            "summary_matches_frozen_values": summary_matches,
            "r_t_model_access_count": 0,
        },
        "G0": {"status": "pass" if g0_pass else "fail", "reason": "source_hash_chain_verified" if g0_pass else "source_hash_chain_mismatch"},
        "G1": {"status": "pass" if g1_pass else "fail", "reason": "full_matrix_time_transform_audit"},
        "G2": {"status": "pass" if g2_pass else "fail", "reason": "match_and_player_cluster_support"},
    }
    path = OUT / "state" / dataset / f"checkpoint_02_preflight_{dataset}.json"
    if path.exists():
        saved = read_json(path)
        require(saved.get("input_lock_sha256") == lock_hash, f"preflight_input_lock_mismatch:{dataset}")
        require(saved.get("source_matrix_sha256") == digest(source_path), f"preflight_matrix_mismatch:{dataset}")
        return frame, saved
    write_json_new(path, value)
    return frame, value


def normal_p_value(z_value: float) -> float:
    return float(math.erfc(abs(z_value) / math.sqrt(2)))


def contrast(coefficient: np.ndarray, covariance: np.ndarray, vector: np.ndarray) -> dict[str, float]:
    estimate = float(vector @ coefficient)
    variance = float(vector @ covariance @ vector)
    require(math.isfinite(variance) and variance > 0, "invalid_contrast_variance")
    standard_error = math.sqrt(variance)
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "ci95_low": float(estimate - 1.95996 * standard_error),
        "ci95_high": float(estimate + 1.95996 * standard_error),
    }


def fit_model(frame: pd.DataFrame, dataset: str) -> dict[str, Any]:
    n = int(len(frame))
    h = frame["H_t"].to_numpy(dtype=float)
    y = frame["e_t"].to_numpy(dtype=float)
    times = frame["time_since_history_tick"].to_numpy(dtype=float)
    z = np.log2(1.0 + times) - EXPECTED[dataset]["mu"]
    design = np.column_stack([np.ones(n), z, h, h * z])
    names = ("alpha", "a_time_main_effect", "beta_average_lag1_association", "kappa_time_modification")
    k = int(design.shape[1])
    require(n > k and np.linalg.matrix_rank(design) == k, "rank_deficient_continuous_time_design")
    inverse = np.linalg.inv(design.T @ design)
    coefficient = inverse @ design.T @ y
    residual = y - design @ coefficient
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)

    def meat(groups: pd.Series) -> tuple[np.ndarray, int]:
        _, codes = np.unique(groups.astype(str).to_numpy(), return_inverse=True)
        scores = design * residual[:, None]
        summed = np.zeros((int(codes.max()) + 1, k))
        np.add.at(summed, codes, scores)
        return summed.T @ summed, int(summed.shape[0])

    match_meat, match_clusters = meat(frame["match_id"])
    player_meat, player_clusters = meat(frame["focal_player_id"])
    intersection = frame["match_id"].astype(str) + "\x1f" + frame["focal_player_id"].astype(str)
    intersection_meat, intersection_clusters = meat(intersection)
    correction = lambda groups: (groups / (groups - 1)) * ((n - 1) / (n - k))
    covariance = inverse @ (
        correction(match_clusters) * match_meat + correction(player_clusters) * player_meat
        - correction(intersection_clusters) * intersection_meat
    ) @ inverse
    covariance = (covariance + covariance.T) / 2
    beta_vector = np.array([0.0, 0.0, 1.0, 0.0])
    kappa_vector = np.array([0.0, 0.0, 0.0, 1.0])
    g3_base = bool(
        np.isfinite(covariance).all() and match_clusters >= 30 and player_clusters >= 30
        and float(leverage.max()) <= 0.20
    )
    result: dict[str, Any] = {
        "n": n,
        "coefficient_names": list(names),
        "coefficients": {name: float(value) for name, value in zip(names, coefficient)},
        "match_clusters": match_clusters,
        "player_clusters": player_clusters,
        "intersection_clusters": intersection_clusters,
        "max_leverage": float(leverage.max()),
        "covariance": covariance.tolist(),
        "g3_pass": False,
    }
    if not g3_base:
        return result

    try:
        beta = contrast(coefficient, covariance, beta_vector)
        kappa = contrast(coefficient, covariance, kappa_vector)
        time_slopes: dict[str, dict[str, float]] = {}
        for label, tick in zip(("P10", "P50", "P90"), EXPECTED[dataset]["quantiles"]):
            vector = np.array([0.0, 0.0, 1.0, math.log2(1.0 + tick) - EXPECTED[dataset]["mu"]])
            time_slopes[label] = {"T": tick, **contrast(coefficient, covariance, vector)}
        beta_kappa_covariance = covariance[np.ix_([2, 3], [2, 3])]
        determinant = float(np.linalg.det(beta_kappa_covariance))
        wald: dict[str, Any] = {"eligible": bool(determinant > 0)}
        if determinant > 0:
            pair = coefficient[[2, 3]]
            statistic = float(pair @ np.linalg.inv(beta_kappa_covariance) @ pair)
            wald.update({"statistic": statistic, "df": 2})
        normal = NormalDist()
        critical = normal.inv_cdf(1 - 0.05 / (2 * 3))
        beta_se = beta["standard_error"]
        scenario_power = {}
        for effect in (0.05, 0.10, 0.20):
            signal = effect / beta_se
            scenario_power[f"abs_beta_{effect:.2f}"] = float(
                1 - normal.cdf(critical - signal) + normal.cdf(-critical - signal)
            )
        result.update({
            "g3_pass": True,
            "beta": {**beta, "z_value": beta["estimate"] / beta_se, "p_value": normal_p_value(beta["estimate"] / beta_se)},
            "kappa": kappa,
            "time_point_slopes": time_slopes,
            "joint_wald_beta_kappa": wald,
            "ci_half_width_95": float(1.95996 * beta_se),
            "bonferroni_familywise_critical_z": critical,
            "mde_80_fwer": float((critical + normal.inv_cdf(0.80)) * beta_se),
            "mde_90_fwer": float((critical + normal.inv_cdf(0.90)) * beta_se),
            "scenario_power_bonferroni_fwer": scenario_power,
        })
    except RuntimeError:
        return result
    return result


def model_or_checkpoint(dataset: str, frame: pd.DataFrame, preflight: dict[str, Any], lock_hash: str) -> dict[str, Any]:
    path = OUT / "state" / dataset / f"checkpoint_03_model_{dataset}.json"
    source_hash = preflight["source_matrix_sha256"]
    if path.exists():
        saved = read_json(path)
        require(saved.get("input_lock_sha256") == lock_hash, f"model_input_lock_mismatch:{dataset}")
        require(saved.get("source_matrix_sha256") == source_hash, f"model_matrix_mismatch:{dataset}")
        return saved["model"]
    model = fit_model(frame, dataset)
    write_json_new(path, {
        "stage": "continuous_time_d1_v1_model",
        "dataset": dataset,
        "created_at_utc": now(),
        "input_lock_sha256": lock_hash,
        "source_matrix_sha256": source_hash,
        "model": model,
    })
    return model


def holm(models: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    if not all(model["g3_pass"] for model in models.values()):
        return {dataset: None for dataset in models}
    ordered = sorted((float(model["beta"]["p_value"]), dataset) for dataset, model in models.items())
    adjusted: dict[str, float | None] = {}
    running = 0.0
    for position, (p_value, dataset) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - position) * p_value))
        adjusted[dataset] = running
    return adjusted


def gates_for(preflight_value: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    model_release = "pass" if (
        all(preflight_value[name]["status"] == "pass" for name in ("G0", "G1", "G2"))
        and model["g3_pass"]
    ) else "fail"
    return {
        "G0": {**preflight_value["G0"], "metrics": {"e1_path_access_count": 0, "r_t_model_access_count": 0, "source_hash_audit": verify_source_hashes()}},
        "G1": {**preflight_value["G1"], "metrics": preflight_value["metrics"]},
        "G2": {**preflight_value["G2"], "metrics": {"match_clusters": preflight_value["metrics"]["time_summary"]["match_clusters"], "player_clusters": preflight_value["metrics"]["time_summary"]["player_clusters"]}},
        "G3": {"status": "pass" if model["g3_pass"] else "fail", "reason": "unweighted_two_way_crv1_continuous_time_and_precision_thresholds", "metrics": model},
        "G4": {"status": "pending", "reason": "no_independent_locked_calibration_sample"},
        "G5": {"status": "pending", "reason": "no_complete_opportunity_model", "residual_centering": "none"},
        "overall": {
            "model_release": model_release,
            "interpretation_release": "observed_opportunity_association_only" if model_release == "pass" else "blocked",
            "forbidden_interpretations": ["round_boundary_effect", "state_effect", "causal_effect"],
        },
    }


def write_or_verify(path: Path, value: dict[str, Any], lock_hash: str) -> None:
    if path.exists():
        require(read_json(path).get("input_lock_sha256") == lock_hash, f"checkpoint_input_lock_mismatch:{path}")
        return
    write_json_new(path, value)


def output_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(OUT)): digest(path)
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"}
        and not any(part.startswith("failed_") for part in path.relative_to(OUT).parts)
    }


def run(selected: tuple[str, ...]) -> None:
    _, lock_hash = input_lock()
    preflights: dict[str, dict[str, Any]] = {}
    models: dict[str, dict[str, Any]] = {}
    for dataset in selected:
        frame, preflight_value = preflight(dataset, lock_hash)
        require(all(preflight_value[name]["status"] == "pass" for name in ("G0", "G1", "G2")), f"preflight_gate_failed:{dataset}")
        preflights[dataset] = preflight_value
        models[dataset] = model_or_checkpoint(dataset, frame, preflight_value, lock_hash)
    if set(selected) != set(DATASETS):
        print(json.dumps({"status": "partial_checkpoint_complete", "datasets": selected, "output_root": str(OUT)}, ensure_ascii=True))
        return

    require(all(model["g3_pass"] for model in models.values()), "model_gate_failed")
    adjusted = holm(models)
    gates: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        models[dataset]["holm_adjusted_beta_p_value_three_partition_family"] = adjusted[dataset]
        gates[dataset] = gates_for(preflights[dataset], models[dataset])
    require(all(value["overall"]["model_release"] == "pass" for value in gates.values()), "model_release_failed")
    write_or_verify(OUT / "checkpoint_04_gate_audit.json", {
        "stage": "continuous_time_d1_v1_gate_audit", "created_at_utc": now(),
        "input_lock_sha256": lock_hash, "gates": gates,
    }, lock_hash)
    write_or_verify(OUT / "checkpoint_05_models.json", {
        "stage": "continuous_time_d1_v1_models", "created_at_utc": now(),
        "input_lock_sha256": lock_hash, "models": models,
        "inference": "unweighted_two_way_crv1_match_and_focal_player_normal_approximation; holm_three_partition_beta_family",
    }, lock_hash)
    hashes_path = OUT / "file_sha256.json"
    hashes = output_hashes()
    if hashes_path.exists():
        require(read_json(hashes_path) == hashes, "output_file_sha256_mismatch")
    else:
        write_json_new(hashes_path, hashes)
    manifest_path = OUT / "manifest.json"
    manifest = {
        "stage": "continuous_time_d1_v1", "status": "passed", "created_at_utc": now(),
        "classification": "exploratory_nonconfirmatory_d1_only", "input_lock_sha256": lock_hash,
        "gate_statuses": {dataset: {name: value["status"] for name, value in gates[dataset].items() if name.startswith("G")} for dataset in DATASETS},
        "interpretation": "D1-only continuous-time observed-opportunity association; not a round-boundary, state, causal, calibration, or predictive-gain conclusion.",
        "file_sha256_sha256": digest(hashes_path),
    }
    if manifest_path.exists():
        saved = read_json(manifest_path)
        require(saved.get("input_lock_sha256") == lock_hash, "manifest_input_lock_mismatch")
        require(saved.get("file_sha256_sha256") == digest(hashes_path), "manifest_file_sha256_mismatch")
    else:
        write_json_new(manifest_path, manifest)
    print(json.dumps({"status": "passed", "manifest": str(manifest_path)}, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    try:
        run(tuple(args.datasets))
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True)
        failed = OUT / f"failed_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        failed.mkdir(exist_ok=False)
        write_json_new(failed / "failure_manifest.json", {
            "stage": "continuous_time_d1_v1", "status": "failed", "created_at_utc": now(),
            "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
