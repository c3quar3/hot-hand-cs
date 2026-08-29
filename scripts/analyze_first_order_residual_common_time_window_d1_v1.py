"""Run the frozen D1-only common-time-window interaction analysis."""

from __future__ import annotations

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
SOURCE = ROOT / "results" / "first_order_residual_round_boundary_interaction_d1_v1"
OUT = ROOT / "results" / "first_order_residual_common_time_window_d1_v1"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "20_D1共同时间支持域回合边界关联执行协议_v1_D1_only_2026-08-26.md"
DRAFT = ROOT / "docs" / "修订与补丁" / "19_D1共同时间支持域回合边界关联分析协议_草案_v1_2026-08-26.md"
DATASETS = ("D1-A", "D1-B", "D1-C")
WINDOW_LOW = 3072
WINDOW_HIGH = 3584
EXPECTED = {
    "D1-A": {0: (361, 164, 137), 1: (909, 186, 191)},
    "D1-B": {0: (227, 81, 113), 1: (563, 97, 169)},
    "D1-C": {0: (156, 76, 83), 1: (410, 85, 126)},
}
REQUIRED_COLUMNS = {
    "e_t", "H_t", "R_t", "match_id", "focal_player_id", "map_id", "round_id",
    "event_tick", "history_source_event_tick", "time_since_history_tick",
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
    expected = json_file(SOURCE / "file_sha256.json")
    mismatches = [
        relative for relative, expected_hash in expected.items()
        if not (SOURCE / relative).is_file() or digest(SOURCE / relative) != expected_hash
    ]
    manifest = json_file(SOURCE / "manifest.json")
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
        saved = json_file(path)
        require(saved.get("stage") == "common_time_window_d1_v1_input_lock", "invalid_input_lock_stage")
        require(saved.get("inputs") == inputs, "input_lock_source_hash_changed")
        return saved, digest(path)
    value = {
        "stage": "common_time_window_d1_v1_input_lock",
        "created_at_utc": now(),
        "classification": "exploratory_nonconfirmatory_d1_only",
        "inputs": inputs,
        "source_hash_audit": source_audit,
        "e1_path_access_count": 0,
    }
    write_json_new(path, value)
    return value, digest(path)


def quantiles(series: pd.Series) -> dict[str, float]:
    values = series.to_numpy(dtype=float)
    return {
        "min": float(np.min(values)), "q1": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)), "q3": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def preflight(dataset: str, lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_path = SOURCE / "state" / dataset / "analysis_matrix.parquet"
    source_checkpoint = json_file(SOURCE / "state" / dataset / f"checkpoint_03_analysis_matrix_{dataset}.json")
    require(source_checkpoint.get("matrix_sha256") == digest(source_path), f"source_matrix_hash_mismatch:{dataset}")
    source = pd.read_parquet(source_path)
    missing = sorted(REQUIRED_COLUMNS - set(source.columns))
    require(not missing, f"missing_required_columns:{dataset}:{','.join(missing)}")
    finite = ["e_t", "H_t", "R_t", "event_tick", "history_source_event_tick", "time_since_history_tick"]
    nonfinite = int((~np.isfinite(source[finite].to_numpy(dtype=float))).sum())
    strict_time_violations = int((source["history_source_event_tick"] >= source["event_tick"]).sum())
    r_values = sorted(int(value) for value in source["R_t"].unique())
    window_mask = (source["time_since_history_tick"] >= WINDOW_LOW) & (source["time_since_history_tick"] < WINDOW_HIGH)
    frame = source.loc[window_mask].copy()
    outside_rows = int((~window_mask).sum())
    group_metrics: dict[str, Any] = {}
    counts_match = True
    for r_value, name in ((0, "same"), (1, "cross")):
        group = frame.loc[frame["R_t"] == r_value]
        observed = (
            int(len(group)), int(group["match_id"].astype(str).nunique()),
            int(group["focal_player_id"].astype(str).nunique()),
        )
        expected = EXPECTED[dataset][r_value]
        counts_match = counts_match and observed == expected
        group_metrics[name] = {
            "rows": observed[0], "match_clusters": observed[1], "player_clusters": observed[2],
            "expected_rows": expected[0], "expected_match_clusters": expected[1],
            "expected_player_clusters": expected[2], "time_since_history_tick": quantiles(group["time_since_history_tick"]),
        }
    require(counts_match, f"window_support_count_mismatch:{dataset}")
    require((frame["time_since_history_tick"] >= WINDOW_LOW).all() and (frame["time_since_history_tick"] < WINDOW_HIGH).all(), f"window_boundary_violation:{dataset}")
    output_path = OUT / "state" / dataset / "analysis_matrix.parquet"
    checkpoint_path = OUT / "state" / dataset / f"checkpoint_02_window_preflight_{dataset}.json"
    if not output_path.exists():
        write_parquet_new(output_path, frame)
        write_json_new(checkpoint_path, {
            "stage": "common_time_window_d1_v1_window_preflight", "dataset": dataset,
            "created_at_utc": now(), "input_lock_sha256": lock_hash,
            "matrix_sha256": digest(output_path), "rows": int(len(frame)),
        })
    else:
        saved = json_file(checkpoint_path)
        require(saved.get("input_lock_sha256") == lock_hash, f"checkpoint_input_lock_mismatch:{dataset}")
        require(saved.get("matrix_sha256") == digest(output_path), f"output_matrix_hash_mismatch:{dataset}")
        require(saved.get("rows") == len(frame), f"output_matrix_row_mismatch:{dataset}")
        frame = pd.read_parquet(output_path)
    g0_pass = not verify_source_hashes()["mismatches"] and verify_source_hashes()["manifest_file_sha256_matches"]
    g1_pass = nonfinite == 0 and strict_time_violations == 0 and r_values == [0, 1] and counts_match
    g2_pass = all(item["match_clusters"] >= 30 and item["player_clusters"] >= 30 for item in group_metrics.values())
    return frame, {
        "stage": "common_time_window_d1_v1_window_preflight", "dataset": dataset,
        "created_at_utc": now(), "input_lock_sha256": lock_hash,
        "matrix_sha256": digest(output_path),
        "metrics": {
            "source_rows": int(len(source)), "window_rows": int(len(frame)), "outside_window_rows": outside_rows,
            "window_low_inclusive": WINDOW_LOW, "window_high_exclusive": WINDOW_HIGH,
            "missing_required_columns": missing, "nonfinite_required_values": nonfinite,
            "strict_time_violations": strict_time_violations, "R_t_values": r_values,
            "groups": group_metrics,
        },
        "G0": {"status": "pass" if g0_pass else "fail", "reason": "source_hash_chain_verified" if g0_pass else "source_hash_chain_mismatch"},
        "G1": {"status": "pass" if g1_pass else "fail", "reason": "fixed_window_and_row_audit"},
        "G2": {"status": "pass" if g2_pass else "fail", "reason": "common_window_cluster_support"},
    }


def normal_p_value(z_value: float) -> float:
    return float(math.erfc(abs(z_value) / math.sqrt(2)))


def fit_model(frame: pd.DataFrame) -> dict[str, Any]:
    n = int(len(frame))
    h = frame["H_t"].to_numpy(dtype=float)
    r = frame["R_t"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(n), h, r, h * r])
    y = frame["e_t"].to_numpy(dtype=float)
    k = int(design.shape[1])
    require(n > k and np.linalg.matrix_rank(design) == k, "rank_deficient_interaction_design")
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
    g3_pass = bool(match_clusters >= 30 and player_clusters >= 30 and float(leverage.max()) <= 0.20 and np.isfinite(covariance).all() and delta_variance > 0)
    model: dict[str, Any] = {
        "n": n, "alpha": float(coefficient[0]), "beta_same": float(coefficient[1]),
        "gamma_cross_round_intercept_difference": float(coefficient[2]),
        "delta_slope_difference_cross_minus_same": float(coefficient[3]),
        "beta_cross": float(coefficient[1] + coefficient[3]), "match_clusters": match_clusters,
        "player_clusters": player_clusters, "intersection_clusters": intersection_clusters,
        "max_leverage": float(leverage.max()), "covariance": covariance.tolist(),
        "delta_variance": delta_variance, "g3_pass": g3_pass,
    }
    if g3_pass:
        delta_se = math.sqrt(delta_variance)
        model.update({
            "delta_se": delta_se, "delta_z_value": float(coefficient[3] / delta_se),
            "delta_p_value": normal_p_value(float(coefficient[3] / delta_se)),
            "delta_ci95_low": float(coefficient[3] - 1.95996 * delta_se),
            "delta_ci95_high": float(coefficient[3] + 1.95996 * delta_se),
        })
    return model


def holm(models: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    if not all(model["g3_pass"] for model in models.values()):
        return {dataset: None for dataset in models}
    ordered = sorted((float(model["delta_p_value"]), dataset) for dataset, model in models.items())
    adjusted: dict[str, float | None] = {}
    running = 0.0
    for position, (p_value, dataset) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - position) * p_value))
        adjusted[dataset] = running
    return adjusted


def verify_or_write(path: Path, value: dict[str, Any], lock_hash: str) -> None:
    if not path.exists():
        write_json_new(path, value)
        return
    saved = json_file(path)
    require(saved.get("input_lock_sha256") == lock_hash, f"checkpoint_input_lock_mismatch:{path}")


def run() -> None:
    _, lock_hash = input_lock()
    matrices: dict[str, pd.DataFrame] = {}
    preflights: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        matrices[dataset], preflights[dataset] = preflight(dataset, lock_hash)
    models = {dataset: fit_model(matrix) for dataset, matrix in matrices.items()}
    adjusted = holm(models)
    gates: dict[str, Any] = {}
    for dataset in DATASETS:
        preflight_value, model = preflights[dataset], models[dataset]
        g3_status = "pass" if model["g3_pass"] else "fail"
        model_release = "pass" if all(preflight_value[name]["status"] == "pass" for name in ("G0", "G1", "G2")) and g3_status == "pass" else "fail"
        if model["g3_pass"]:
            model["holm_adjusted_delta_p_value_three_partition_family"] = adjusted[dataset]
        gates[dataset] = {
            "G0": {**preflight_value["G0"], "metrics": {"e1_path_access_count": 0, "source_hash_audit": verify_source_hashes()}},
            "G1": {**preflight_value["G1"], "metrics": preflight_value["metrics"]},
            "G2": {**preflight_value["G2"], "metrics": preflight_value["metrics"]},
            "G3": {"status": g3_status, "reason": "two_way_crv1_interaction_thresholds", "metrics": model},
            "G4": {"status": "pending", "reason": "no_independent_locked_calibration_sample"},
            "G5": {"status": "pending", "reason": "no_complete_opportunity_model", "residual_centering": "none"},
            "overall": {
                "model_release": model_release, "support_release": "pass" if preflight_value["G2"]["status"] == "pass" else "blocked",
                "interpretation_release": "observed_opportunity_association_only" if model_release == "pass" else "blocked",
                "forbidden_interpretations": ["round_boundary_effect", "state_effect", "causal_effect"],
            },
        }
    require(all(value["overall"]["model_release"] == "pass" for value in gates.values()), "model_release_failed")
    verify_or_write(OUT / "checkpoint_03_gate_audit.json", {
        "stage": "common_time_window_d1_v1_gate_audit", "created_at_utc": now(), "input_lock_sha256": lock_hash, "gates": gates,
    }, lock_hash)
    verify_or_write(OUT / "checkpoint_04_models.json", {
        "stage": "common_time_window_d1_v1_models", "created_at_utc": now(), "input_lock_sha256": lock_hash,
        "models": models, "inference": "two_way_crv1_match_and_focal_player_normal_approximation; holm_three_partition_delta_family",
    }, lock_hash)
    hashes = OUT / "file_sha256.json"
    if not hashes.exists():
        write_json_new(hashes, {str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"}})
    else:
        for relative, expected_hash in json_file(hashes).items():
            require((OUT / relative).is_file() and digest(OUT / relative) == expected_hash, f"output_hash_mismatch:{relative}")
    manifest = OUT / "manifest.json"
    if not manifest.exists():
        write_json_new(manifest, {
            "stage": "common_time_window_d1_v1", "status": "passed", "created_at_utc": now(),
            "classification": "exploratory_nonconfirmatory_d1_only", "input_lock_sha256": lock_hash,
            "gate_statuses": {dataset: {name: value["status"] for name, value in gates[dataset].items() if name.startswith("G")} for dataset in DATASETS},
            "interpretation": "D1-only common-time-window observed-opportunity association; not a round-boundary, state, causal, calibration, or predictive-gain conclusion.",
            "file_sha256_sha256": digest(hashes),
        })
    else:
        saved = json_file(manifest)
        require(saved.get("input_lock_sha256") == lock_hash, "manifest_input_lock_mismatch")
        require(saved.get("file_sha256_sha256") == digest(hashes), "manifest_file_sha256_mismatch")
    print(json.dumps({"status": "passed", "manifest": str(manifest)}, ensure_ascii=True))


def main() -> None:
    try:
        run()
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True)
        failed = OUT / f"failed_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        failed.mkdir(exist_ok=False)
        write_json_new(failed / "failure_manifest.json", {
            "stage": "common_time_window_d1_v1", "status": "failed", "created_at_utc": now(),
            "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
