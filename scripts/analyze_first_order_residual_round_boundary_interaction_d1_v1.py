"""Run the frozen D1-only round-boundary interaction analysis."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "first_order_residual_round_boundary_interaction_d1_v1"
V4 = ROOT / "results" / "first_order_residual_persistence_d1_v4"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "18_D1一阶历史残差回合边界交互执行协议_v1_D1_only_2026-08-26.md"
DRAFT = ROOT / "docs" / "修订与补丁" / "17_D1一阶历史残差回合边界交互分析协议_草案_v1_2026-08-26.md"
ERRATUM = ROOT / "docs" / "报告" / "补丁与修订" / "08_D1一阶历史残差持续性执行报告_G5术语勘误_v1_2026-08-26.md"
DATASETS = ("D1-A", "D1-B", "D1-C")
REQUIRED_COLUMNS = {
    "analysis_eligible", "e_t", "H_t", "history_crosses_round", "match_id",
    "focal_player_id", "map_id", "round_id", "event_tick", "history_source_event_tick",
    "time_since_history_tick",
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


def json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def verify_or_write_json(path: Path, value: dict[str, Any], lock_hash: str) -> None:
    if not path.exists():
        write_json_new(path, value)
        return
    saved = json_file(path)
    require(saved.get("input_lock_sha256") == lock_hash, f"checkpoint_input_lock_mismatch:{path}")


def source_paths() -> list[Path]:
    paths = [
        PROTOCOL, DRAFT, ERRATUM, Path(__file__).resolve(),
        V4 / "manifest.json", V4 / "file_sha256.json", V4 / "checkpoint_03_gate_audit.json",
    ]
    for dataset in DATASETS:
        paths.extend([
            V4 / "state" / dataset / "history_state.parquet",
            V4 / "state" / dataset / f"checkpoint_02_history_{dataset}.json",
        ])
    return paths


def verify_v4_hashes() -> dict[str, Any]:
    expected = json_file(V4 / "file_sha256.json")
    mismatches: list[str] = []
    for relative, expected_hash in expected.items():
        path = V4 / relative
        if not path.is_file() or digest(path) != expected_hash:
            mismatches.append(relative)
    manifest = json_file(V4 / "manifest.json")
    file_sha_hash = digest(V4 / "file_sha256.json")
    return {
        "checked_files": len(expected),
        "mismatches": mismatches,
        "manifest_file_sha256_matches": manifest.get("file_sha256_sha256") == file_sha_hash,
    }


def input_lock() -> tuple[dict[str, Any], str]:
    path = OUT / "checkpoint_01_input_lock.json"
    current = {str(item): digest(item) for item in source_paths()}
    if path.exists():
        saved = json_file(path)
        require(saved.get("stage") == "round_boundary_interaction_d1_v1_input_lock", "invalid_input_lock_stage")
        require(saved.get("inputs") == current, "input_lock_source_hash_changed")
        return saved, digest(path)

    v4_hashes = verify_v4_hashes()
    require(not v4_hashes["mismatches"], "v4_file_sha256_mismatch")
    require(v4_hashes["manifest_file_sha256_matches"], "v4_manifest_file_sha256_mismatch")
    value = {
        "stage": "round_boundary_interaction_d1_v1_input_lock",
        "created_at_utc": now(),
        "classification": "exploratory_nonconfirmatory_d1_only",
        "inputs": current,
        "v4_hash_audit": v4_hashes,
        "e1_path_access_count": 0,
    }
    write_json_new(path, value)
    return value, digest(path)


def quantile_summary(series: pd.Series) -> dict[str, float]:
    values = series.to_numpy(dtype=float)
    return {
        "min": float(np.min(values)), "q1": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)), "q3": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def preflight(dataset: str, lock_hash: str, v4_gates: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    state_path = V4 / "state" / dataset / "history_state.parquet"
    checkpoint_path = V4 / "state" / dataset / f"checkpoint_02_history_{dataset}.json"
    source = pd.read_parquet(state_path)
    missing = sorted(REQUIRED_COLUMNS - set(source.columns))
    require(not missing, f"missing_required_columns:{dataset}:{','.join(missing)}")
    checkpoint = json_file(checkpoint_path)
    require(checkpoint.get("state_sha256") == digest(state_path), f"v4_state_hash_mismatch:{dataset}")
    expected_rows = int(checkpoint["history_audit"]["analysis_eligible_rows"])
    frame = source.loc[source["analysis_eligible"].astype(bool)].copy()
    require(len(frame) == expected_rows, f"analysis_eligible_count_mismatch:{dataset}")

    finite_columns = ["e_t", "H_t", "event_tick", "history_source_event_tick", "time_since_history_tick"]
    nonfinite = int((~np.isfinite(frame[finite_columns].to_numpy(dtype=float))).sum())
    strict_time_violations = int((frame["history_source_event_tick"].to_numpy(dtype=float) >= frame["event_tick"].to_numpy(dtype=float)).sum())
    nonpositive_interval = int((frame["time_since_history_tick"].to_numpy(dtype=float) <= 0).sum())
    round_boundary_missing = int(frame["history_crosses_round"].isna().sum())
    require(round_boundary_missing == 0, f"missing_history_crosses_round:{dataset}")
    frame["R_t"] = frame["history_crosses_round"].astype(bool).astype(int)
    r_values = sorted(int(value) for value in frame["R_t"].unique())
    same = frame.loc[frame["R_t"] == 0].copy()
    cross = frame.loc[frame["R_t"] == 1].copy()
    require(len(same) > 0 and len(cross) > 0, f"empty_round_boundary_group:{dataset}")

    same_lag, cross_lag = quantile_summary(same["time_since_history_tick"]), quantile_summary(cross["time_since_history_tick"])
    overlap_low = max(same_lag["q1"], cross_lag["q1"])
    overlap_high = min(same_lag["q3"], cross_lag["q3"])
    support_overlap = overlap_low <= overlap_high
    v4_g2 = v4_gates[dataset]["G2"]["metrics"]
    inherited_cross_map_violations = int(v4_g2["cross_map_history_rows"])

    matrix = frame[[
        "e_t", "H_t", "R_t", "match_id", "focal_player_id", "map_id", "round_id",
        "event_tick", "history_source_event_tick", "time_since_history_tick",
    ]].copy()
    matrix_path = OUT / "state" / dataset / "analysis_matrix.parquet"
    matrix_checkpoint = OUT / "state" / dataset / f"checkpoint_03_analysis_matrix_{dataset}.json"
    if not matrix_path.exists():
        write_parquet_new(matrix_path, matrix)
        write_json_new(matrix_checkpoint, {
            "stage": "round_boundary_interaction_d1_v1_analysis_matrix", "dataset": dataset,
            "created_at_utc": now(), "input_lock_sha256": lock_hash,
            "matrix_sha256": digest(matrix_path), "rows": int(len(matrix)),
        })
    else:
        require(matrix_checkpoint.is_file(), f"incomplete_analysis_matrix_checkpoint:{dataset}")
        saved_matrix = json_file(matrix_checkpoint)
        require(saved_matrix.get("input_lock_sha256") == lock_hash, f"analysis_matrix_lock_mismatch:{dataset}")
        require(saved_matrix.get("matrix_sha256") == digest(matrix_path), f"analysis_matrix_hash_mismatch:{dataset}")
        require(int(saved_matrix.get("rows", -1)) == len(matrix), f"analysis_matrix_row_mismatch:{dataset}")
        matrix = pd.read_parquet(matrix_path)

    g0_pass = not verify_v4_hashes()["mismatches"] and verify_v4_hashes()["manifest_file_sha256_matches"]
    g1_pass = nonfinite == 0 and strict_time_violations == 0 and inherited_cross_map_violations == 0 and r_values == [0, 1]
    g2_fail = nonpositive_interval > 0
    g2_status = "fail" if g2_fail else ("pass" if support_overlap else "pending")
    metrics = {
        "v4_analysis_eligible_rows": expected_rows, "analysis_matrix_rows": int(len(matrix)),
        "missing_required_columns": missing, "nonfinite_required_values": nonfinite,
        "strict_time_violations": strict_time_violations,
        "inherited_cross_map_violations": inherited_cross_map_violations,
        "missing_history_crosses_round": round_boundary_missing,
        "R_t_value_counts": {str(key): int(value) for key, value in frame["R_t"].value_counts().sort_index().items()},
        "n_same": int(len(same)), "n_cross": int(len(cross)),
        "same_match_clusters": int(same["match_id"].astype(str).nunique()),
        "cross_match_clusters": int(cross["match_id"].astype(str).nunique()),
        "same_player_clusters": int(same["focal_player_id"].astype(str).nunique()),
        "cross_player_clusters": int(cross["focal_player_id"].astype(str).nunique()),
        "same_time_since_history_tick": same_lag, "cross_time_since_history_tick": cross_lag,
        "iqr_overlap_interval": [overlap_low, overlap_high], "support_overlap": support_overlap,
        "nonpositive_or_nonfinite_interval": nonpositive_interval,
    }
    value = {
        "stage": "round_boundary_interaction_d1_v1_preflight", "dataset": dataset,
        "created_at_utc": now(), "input_lock_sha256": lock_hash,
        "matrix_sha256": digest(matrix_path), "metrics": metrics,
        "G0": {"status": "pass" if g0_pass else "fail", "reason": "v4_hash_chain_verified" if g0_pass else "v4_hash_chain_mismatch"},
        "G1": {"status": "pass" if g1_pass else "fail", "reason": "state_and_time_audit"},
        "G2": {"status": g2_status, "reason": "time_support_overlap" if support_overlap else ("time_support_no_iqr_overlap" if not g2_fail else "invalid_time_interval")},
    }
    checkpoint_out = OUT / "state" / dataset / f"checkpoint_02_preflight_{dataset}.json"
    verify_or_write_json(checkpoint_out, value, lock_hash)
    return matrix, value


def normal_p_value(z_value: float) -> float:
    return float(math.erfc(abs(z_value) / math.sqrt(2)))


def fit_two_way_interaction(frame: pd.DataFrame) -> dict[str, Any]:
    n = int(len(frame))
    h = frame["H_t"].to_numpy(dtype=float)
    r = frame["R_t"].to_numpy(dtype=float)
    y = frame["e_t"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(n), h, r, h * r])
    k = int(design.shape[1])
    require(n > k, "insufficient_rows_for_interaction")
    require(np.linalg.matrix_rank(design) == k, "rank_deficient_interaction_design")
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
    delta_variance = float(covariance[3, 3])
    g3_pass = bool(
        match_clusters >= 30 and player_clusters >= 30 and float(leverage.max()) <= 0.20
        and np.isfinite(covariance).all() and delta_variance > 0
    )
    result: dict[str, Any] = {
        "n": n, "alpha": float(coefficient[0]), "beta_same": float(coefficient[1]),
        "gamma_cross_round_intercept_difference": float(coefficient[2]), "delta_slope_difference_cross_minus_same": float(coefficient[3]),
        "beta_cross": float(coefficient[1] + coefficient[3]), "match_clusters": match_clusters,
        "player_clusters": player_clusters, "intersection_clusters": intersection_clusters,
        "max_leverage": float(leverage.max()), "covariance": covariance.tolist(),
        "delta_variance": delta_variance, "g3_pass": g3_pass,
    }
    if g3_pass:
        delta_se = math.sqrt(delta_variance)
        cross_variance = float(covariance[1, 1] + covariance[3, 3] + 2 * covariance[1, 3])
        result.update({
            "delta_se": delta_se, "delta_z_value": float(coefficient[3] / delta_se),
            "delta_p_value": normal_p_value(float(coefficient[3] / delta_se)),
            "delta_ci95_low": float(coefficient[3] - 1.95996 * delta_se),
            "delta_ci95_high": float(coefficient[3] + 1.95996 * delta_se),
            "beta_same_se": math.sqrt(float(covariance[1, 1])),
            "beta_cross_se": math.sqrt(cross_variance),
            "beta_same_ci95_low": float(coefficient[1] - 1.95996 * math.sqrt(float(covariance[1, 1]))),
            "beta_same_ci95_high": float(coefficient[1] + 1.95996 * math.sqrt(float(covariance[1, 1]))),
            "beta_cross_ci95_low": float(coefficient[1] + coefficient[3] - 1.95996 * math.sqrt(cross_variance)),
            "beta_cross_ci95_high": float(coefficient[1] + coefficient[3] + 1.95996 * math.sqrt(cross_variance)),
        })
    return result


def holm(models: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    if not all(value.get("g3_pass") for value in models.values()):
        return {dataset: None for dataset in models}
    ordered = sorted((float(value["delta_p_value"]), dataset) for dataset, value in models.items())
    adjusted: dict[str, float | None] = {}
    running = 0.0
    count = len(ordered)
    for position, (p_value, dataset) in enumerate(ordered):
        running = max(running, min(1.0, (count - position) * p_value))
        adjusted[dataset] = running
    return adjusted


def run() -> None:
    _, lock_hash = input_lock()
    v4_gate_audit = json_file(V4 / "checkpoint_03_gate_audit.json")["gates"]
    matrices: dict[str, pd.DataFrame] = {}
    preflights: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        matrices[dataset], preflights[dataset] = preflight(dataset, lock_hash, v4_gate_audit)

    models = {dataset: fit_two_way_interaction(frame) for dataset, frame in matrices.items()}
    adjusted = holm(models)
    gates: dict[str, Any] = {}
    for dataset in DATASETS:
        preflight_value, model = preflights[dataset], models[dataset]
        g4_source = v4_gate_audit[dataset]["G4"]
        g5_source = v4_gate_audit[dataset]["G5"]
        g3_status = "pass" if model["g3_pass"] else "fail"
        model_release = "pass" if all(preflight_value[name]["status"] == "pass" for name in ("G0", "G1")) and preflight_value["G2"]["status"] != "fail" and g3_status == "pass" else "fail"
        interpretation = "observed_opportunity_association_only" if model_release == "pass" else "blocked"
        if model["g3_pass"]:
            model["holm_adjusted_delta_p_value_three_partition_family"] = adjusted[dataset]
        gates[dataset] = {
            "G0": {**preflight_value["G0"], "metrics": {"e1_path_access_count": 0, "v4_hash_audit": verify_v4_hashes()}},
            "G1": {**preflight_value["G1"], "metrics": preflight_value["metrics"]},
            "G2": {**preflight_value["G2"], "metrics": preflight_value["metrics"]},
            "G3": {"status": g3_status, "reason": "two_way_crv1_interaction_thresholds", "metrics": model},
            "G4": {"status": "pending", "reason": "no_independent_locked_calibration_sample", "inherited_v4": g4_source},
            "G5": {"status": "pending", "reason": "no_complete_opportunity_model", "residual_centering": "none", "inherited_v4": g5_source},
            "overall": {
                "model_release": model_release, "support_release": preflight_value["G2"]["status"],
                "interpretation_release": interpretation,
                "forbidden_interpretations": ["unexplained_persistence", "state_effect", "causal_effect"],
            },
        }
    require(all(value["overall"]["model_release"] == "pass" for value in gates.values()), "model_release_failed")

    gate_path = OUT / "checkpoint_04_gate_audit.json"
    model_path = OUT / "checkpoint_05_models.json"
    verify_or_write_json(gate_path, {
        "stage": "round_boundary_interaction_d1_v1_gate_audit", "created_at_utc": now(),
        "input_lock_sha256": lock_hash, "gates": gates,
    }, lock_hash)
    verify_or_write_json(model_path, {
        "stage": "round_boundary_interaction_d1_v1_models", "created_at_utc": now(),
        "input_lock_sha256": lock_hash, "models": models,
        "inference": "two_way_crv1_match_and_focal_player_normal_approximation; holm_three_partition_delta_family",
    }, lock_hash)

    hashes = OUT / "file_sha256.json"
    if not hashes.exists():
        write_json_new(hashes, {
            str(path.relative_to(OUT)): digest(path)
            for path in sorted(OUT.rglob("*"))
            if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"}
        })
    else:
        saved_hashes = json_file(hashes)
        for relative, expected_hash in saved_hashes.items():
            require((OUT / relative).is_file() and digest(OUT / relative) == expected_hash, f"output_hash_mismatch:{relative}")
    manifest = OUT / "manifest.json"
    if not manifest.exists():
        write_json_new(manifest, {
            "stage": "round_boundary_interaction_d1_v1", "status": "passed", "created_at_utc": now(),
            "classification": "exploratory_nonconfirmatory_d1_only", "input_lock_sha256": lock_hash,
            "gate_statuses": {dataset: {name: value["status"] for name, value in gates[dataset].items() if name.startswith("G")} for dataset in DATASETS},
            "interpretation": "Conditional-on-observed-opportunity D1-only round-boundary slope-difference association; not a hot-hand, state, causal, calibration, or predictive-gain conclusion.",
            "file_sha256_sha256": digest(hashes),
        })
    else:
        saved_manifest = json_file(manifest)
        require(saved_manifest.get("input_lock_sha256") == lock_hash, "manifest_input_lock_mismatch")
        require(saved_manifest.get("file_sha256_sha256") == digest(hashes), "manifest_file_sha256_mismatch")
    print(json.dumps({"status": "passed", "manifest": str(manifest)}, ensure_ascii=True))


def main() -> None:
    try:
        run()
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True)
        failed = OUT / f"failed_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        failed.mkdir(exist_ok=False)
        write_json_new(failed / "failure_manifest.json", {
            "stage": "round_boundary_interaction_d1_v1", "status": "failed", "created_at_utc": now(),
            "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
