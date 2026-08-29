"""Run the frozen D1-only time-stratified common-support interaction analysis."""

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
OUT = ROOT / "results" / "first_order_residual_time_stratified_common_support_d1_v1"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "22_D1时间分层共同支持域回合边界关联执行协议_v1_D1_only_2026-08-26.md"
DRAFT = ROOT / "docs" / "修订与补丁" / "21_D1时间分层共同支持域回合边界关联执行协议_草案_v1_D1_only_2026-08-26.md"
DATASETS = ("D1-A", "D1-B", "D1-C")
BINS = (2048, 2560, 3072, 3584, 4096, 4608)
WINDOW_LOW = BINS[0]
WINDOW_HIGH = 5120
SUPPORT_COLUMNS = {"R_t", "time_since_history_tick", "match_id", "focal_player_id"}
REQUIRED_COLUMNS = {
    "e_t", "H_t", "R_t", "match_id", "focal_player_id", "event_tick",
    "history_source_event_tick", "time_since_history_tick",
}
EXPECTED: dict[str, dict[int, dict[int, tuple[int, int, int]]]] = {
    "D1-A": {
        2048: {0: (713, 187, 176), 1: (151, 94, 83)},
        2560: {0: (556, 185, 159), 1: (613, 181, 172)},
        3072: {0: (361, 164, 137), 1: (909, 186, 191)},
        3584: {0: (274, 136, 121), 1: (1102, 199, 199)},
        4096: {0: (187, 115, 97), 1: (1334, 196, 212)},
        4608: {0: (137, 103, 83), 1: (1512, 200, 225)},
    },
    "D1-B": {
        2048: {0: (450, 97, 153), 1: (99, 51, 67)},
        2560: {0: (327, 91, 128), 1: (367, 92, 149)},
        3072: {0: (227, 81, 113), 1: (563, 97, 169)},
        3584: {0: (156, 76, 93), 1: (646, 97, 176)},
        4096: {0: (99, 59, 72), 1: (756, 99, 172)},
        4608: {0: (76, 52, 61), 1: (839, 98, 177)},
    },
    "D1-C": {
        2048: {0: (358, 92, 109), 1: (110, 59, 77)},
        2560: {0: (205, 80, 98), 1: (323, 91, 116)},
        3072: {0: (156, 76, 83), 1: (410, 85, 126)},
        3584: {0: (99, 54, 66), 1: (511, 93, 120)},
        4096: {0: (80, 54, 57), 1: (626, 100, 135)},
        4608: {0: (52, 36, 43), 1: (664, 98, 140)},
    },
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
        require(saved.get("stage") == "time_stratified_common_support_d1_v1_input_lock", "invalid_input_lock_stage")
        require(saved.get("inputs") == inputs, "input_lock_source_hash_changed")
        return saved, digest(path)
    value = {
        "stage": "time_stratified_common_support_d1_v1_input_lock",
        "created_at_utc": now(),
        "classification": "exploratory_nonconfirmatory_d1_only",
        "inputs": inputs,
        "source_hash_audit": source_audit,
        "e1_path_access_count": 0,
    }
    write_json_new(path, value)
    return value, digest(path)


def bin_low(values: pd.Series) -> np.ndarray:
    return (np.floor(values.to_numpy(dtype=float) / 512) * 512).astype(int)


def cell_metrics(frame: pd.DataFrame, dataset: str) -> tuple[list[dict[str, Any]], bool]:
    values: list[dict[str, Any]] = []
    matches = True
    for tick in BINS:
        for group_value, group_name in ((0, "same"), (1, "cross")):
            part = frame.loc[(frame["time_bin_low"] == tick) & (frame["R_t"] == group_value)]
            observed = (
                int(len(part)), int(part["match_id"].astype(str).nunique()),
                int(part["focal_player_id"].astype(str).nunique()),
            )
            expected = EXPECTED[dataset][tick][group_value]
            matches = matches and observed == expected
            values.append({
                "time_bin_low": tick,
                "time_bin_high_exclusive": tick + 512,
                "group": group_name,
                "R_t": group_value,
                "rows": observed[0],
                "match_clusters": observed[1],
                "player_clusters": observed[2],
                "expected_rows": expected[0],
                "expected_match_clusters": expected[1],
                "expected_player_clusters": expected[2],
            })
    return values, matches


def support_preflight(dataset: str, lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_path = SOURCE / "state" / dataset / "analysis_matrix.parquet"
    source_checkpoint = read_json(SOURCE / "state" / dataset / f"checkpoint_03_analysis_matrix_{dataset}.json")
    require(source_checkpoint.get("matrix_sha256") == digest(source_path), f"source_matrix_hash_mismatch:{dataset}")
    support = pd.read_parquet(source_path, columns=sorted(SUPPORT_COLUMNS))
    require(SUPPORT_COLUMNS.issubset(support.columns), f"support_columns_missing:{dataset}")
    support["time_bin_low"] = bin_low(support["time_since_history_tick"])
    mask = (support["time_since_history_tick"] >= WINDOW_LOW) & (support["time_since_history_tick"] < WINDOW_HIGH)
    eligible = support.loc[mask].copy()
    cells, counts_match = cell_metrics(eligible, dataset)
    r_values = sorted(int(value) for value in support["R_t"].unique())
    finite_support = int((~np.isfinite(support[["R_t", "time_since_history_tick"]].to_numpy(dtype=float))).sum())
    require(counts_match, f"support_count_mismatch:{dataset}")
    require(r_values == [0, 1], f"invalid_R_t_values:{dataset}")
    require(finite_support == 0, f"nonfinite_support_values:{dataset}")
    require(set(eligible["time_bin_low"].unique()) == set(BINS), f"missing_or_extra_time_bins:{dataset}")

    output_path = OUT / "state" / dataset / "analysis_matrix.parquet"
    checkpoint_path = OUT / "state" / dataset / f"checkpoint_02_support_preflight_{dataset}.json"
    if output_path.exists():
        saved = read_json(checkpoint_path)
        require(saved.get("input_lock_sha256") == lock_hash, f"checkpoint_input_lock_mismatch:{dataset}")
        require(saved.get("matrix_sha256") == digest(output_path), f"output_matrix_hash_mismatch:{dataset}")
        frame = pd.read_parquet(output_path)
    else:
        source = pd.read_parquet(source_path)
        missing = sorted(REQUIRED_COLUMNS - set(source.columns))
        require(not missing, f"missing_required_columns:{dataset}:{','.join(missing)}")
        frame = source.loc[(source["time_since_history_tick"] >= WINDOW_LOW) & (source["time_since_history_tick"] < WINDOW_HIGH)].copy()
        frame["time_bin_low"] = bin_low(frame["time_since_history_tick"])
        write_parquet_new(output_path, frame)
        write_json_new(checkpoint_path, {
            "stage": "time_stratified_common_support_d1_v1_support_preflight",
            "dataset": dataset,
            "created_at_utc": now(),
            "input_lock_sha256": lock_hash,
            "matrix_sha256": digest(output_path),
            "rows": int(len(frame)),
        })
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    finite_columns = ["e_t", "H_t", "R_t", "event_tick", "history_source_event_tick", "time_since_history_tick"]
    nonfinite = int((~np.isfinite(frame[finite_columns].to_numpy(dtype=float))).sum()) if not missing else -1
    strict_time_violations = int((frame["history_source_event_tick"] >= frame["event_tick"]).sum()) if not missing else -1
    frame_cells, output_counts_match = cell_metrics(frame, dataset)
    within_bounds = bool(((frame["time_since_history_tick"] >= WINDOW_LOW) & (frame["time_since_history_tick"] < WINDOW_HIGH)).all())
    g0_audit = verify_source_hashes()
    g0_pass = not g0_audit["mismatches"] and g0_audit["manifest_file_sha256_matches"]
    g1_pass = bool(not missing and nonfinite == 0 and strict_time_violations == 0 and within_bounds and output_counts_match)
    g2_pass = all(item["match_clusters"] >= 30 and item["player_clusters"] >= 30 for item in frame_cells)
    return frame, {
        "stage": "time_stratified_common_support_d1_v1_support_preflight",
        "dataset": dataset,
        "input_lock_sha256": lock_hash,
        "matrix_sha256": digest(output_path),
        "metrics": {
            "source_rows": int(len(support)),
            "window_rows": int(len(frame)),
            "outside_window_rows": int(len(support) - len(frame)),
            "window_low_inclusive": WINDOW_LOW,
            "window_high_exclusive": WINDOW_HIGH,
            "R_t_values": r_values,
            "support_nonfinite_values": finite_support,
            "missing_required_columns": missing,
            "nonfinite_required_values": nonfinite,
            "strict_time_violations": strict_time_violations,
            "cells": frame_cells,
        },
        "G0": {"status": "pass" if g0_pass else "fail", "reason": "source_hash_chain_verified" if g0_pass else "source_hash_chain_mismatch"},
        "G1": {"status": "pass" if g1_pass else "fail", "reason": "fixed_six_bin_eligibility_audit"},
        "G2": {"status": "pass" if g2_pass else "fail", "reason": "every_bin_group_cluster_support"},
    }


def effective_count(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.square(weights).sum())


def weight_audit(frame: pd.DataFrame, dataset: str, lock_hash: str, matrix_hash: str) -> tuple[np.ndarray, dict[str, Any], str]:
    counts = frame.groupby(["time_bin_low", "R_t"]).size()
    weights = np.array([1.0 / counts.loc[(tick, r)] for tick, r in zip(frame["time_bin_low"], frame["R_t"])], dtype=float)
    weights *= len(weights) / weights.sum()
    cells: list[dict[str, Any]] = []
    totals: list[float] = []
    for tick in BINS:
        for group_value in (0, 1):
            mask = (frame["time_bin_low"].to_numpy() == tick) & (frame["R_t"].to_numpy(dtype=int) == group_value)
            total = float(weights[mask].sum())
            totals.append(total)
            cells.append({
                "time_bin_low": tick,
                "R_t": group_value,
                "rows": int(mask.sum()),
                "raw_weight": float(1.0 / counts.loc[(tick, group_value)]),
                "scaled_weight": float(weights[mask][0]),
                "weighted_total": total,
            })
    def cluster_effective(column: str) -> float:
        totals_by_cluster = pd.DataFrame({"cluster": frame[column].astype(str), "weight": weights}).groupby("cluster")["weight"].sum().to_numpy(dtype=float)
        return effective_count(totals_by_cluster)
    value = {
        "stage": "time_stratified_common_support_d1_v1_weight_audit",
        "dataset": dataset,
        "created_at_utc": now(),
        "input_lock_sha256": lock_hash,
        "matrix_sha256": matrix_hash,
        "cells": cells,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_ratio_max_to_min": float(weights.max() / weights.min()),
        "weighted_cell_total_min": float(min(totals)),
        "weighted_cell_total_max": float(max(totals)),
        "weighted_cell_totals_equal": bool(np.allclose(totals, totals[0], rtol=1e-12, atol=1e-12)),
        "kish_effective_n": effective_count(weights),
        "effective_match_clusters": cluster_effective("match_id"),
        "effective_player_clusters": cluster_effective("focal_player_id"),
    }
    path = OUT / "state" / dataset / f"checkpoint_03_weight_audit_{dataset}.json"
    if path.exists():
        saved = read_json(path)
        require(saved.get("input_lock_sha256") == lock_hash, f"weight_audit_input_lock_mismatch:{dataset}")
        require(saved.get("matrix_sha256") == matrix_hash, f"weight_audit_matrix_mismatch:{dataset}")
        return weights, saved, digest(path)
    write_json_new(path, value)
    return weights, value, digest(path)


def normal_p_value(z_value: float) -> float:
    return float(math.erfc(abs(z_value) / math.sqrt(2)))


def fit_model(frame: pd.DataFrame, weights: np.ndarray, audit: dict[str, Any]) -> dict[str, Any]:
    n = int(len(frame))
    h = frame["H_t"].to_numpy(dtype=float)
    r = frame["R_t"].to_numpy(dtype=float)
    y = frame["e_t"].to_numpy(dtype=float)
    columns: list[np.ndarray] = []
    names: list[str] = []
    bins = frame["time_bin_low"].to_numpy(dtype=int)
    for tick in BINS:
        indicator = (bins == tick).astype(float)
        columns.extend([indicator, indicator * r, indicator * h])
        names.extend([f"alpha_{tick}", f"gamma_cross_{tick}", f"beta_common_{tick}"])
    columns.append(h * r)
    names.append("delta_slope_difference_cross_minus_same")
    design = np.column_stack(columns)
    k = int(design.shape[1])
    require(n > k and np.linalg.matrix_rank(design) == k, "rank_deficient_weighted_interaction_design")
    sqrt_weights = np.sqrt(weights)
    weighted_design = design * sqrt_weights[:, None]
    inverse = np.linalg.inv(weighted_design.T @ weighted_design)
    coefficient = inverse @ weighted_design.T @ (y * sqrt_weights)
    residual = y - design @ coefficient
    leverage = np.einsum("ij,jk,ik->i", weighted_design, inverse, weighted_design)

    def meat(groups: pd.Series) -> tuple[np.ndarray, int]:
        _, codes = np.unique(groups.astype(str).to_numpy(), return_inverse=True)
        scores = design * (weights * residual)[:, None]
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
    delta_index = k - 1
    delta = float(coefficient[delta_index])
    delta_variance = float(covariance[delta_index, delta_index])
    g3_pass = bool(
        np.isfinite(covariance).all() and delta_variance > 0 and match_clusters >= 30 and player_clusters >= 30
        and audit["effective_match_clusters"] >= 30 and audit["effective_player_clusters"] >= 30
        and audit["weight_ratio_max_to_min"] <= 15 and audit["weighted_cell_totals_equal"]
        and float(leverage.max()) <= 0.20
    )
    model: dict[str, Any] = {
        "n": n,
        "coefficient_names": names,
        "coefficients": {name: float(value) for name, value in zip(names, coefficient)},
        "delta_slope_difference_cross_minus_same": delta,
        "match_clusters": match_clusters,
        "player_clusters": player_clusters,
        "intersection_clusters": intersection_clusters,
        "max_weighted_leverage": float(leverage.max()),
        "covariance": covariance.tolist(),
        "delta_variance": delta_variance,
        "g3_pass": g3_pass,
        "weight_audit": {
            "kish_effective_n": audit["kish_effective_n"],
            "effective_match_clusters": audit["effective_match_clusters"],
            "effective_player_clusters": audit["effective_player_clusters"],
            "weight_ratio_max_to_min": audit["weight_ratio_max_to_min"],
        },
    }
    if g3_pass:
        normal = NormalDist()
        critical = normal.inv_cdf(1 - 0.05 / (2 * 3))
        se = math.sqrt(delta_variance)
        scenario_power = {}
        for effect in (0.05, 0.10, 0.20):
            signal = effect / se
            scenario_power[f"abs_delta_{effect:.2f}"] = float(1 - normal.cdf(critical - signal) + normal.cdf(-critical - signal))
        model.update({
            "delta_se": se,
            "delta_z_value": float(delta / se),
            "delta_p_value": normal_p_value(delta / se),
            "delta_ci95_low": float(delta - 1.95996 * se),
            "delta_ci95_high": float(delta + 1.95996 * se),
            "ci_half_width_95": float(1.95996 * se),
            "bonferroni_familywise_critical_z": critical,
            "mde_80_fwer": float((critical + normal.inv_cdf(0.80)) * se),
            "mde_90_fwer": float((critical + normal.inv_cdf(0.90)) * se),
            "scenario_power_bonferroni_fwer": scenario_power,
        })
    return model


def model_or_checkpoint(dataset: str, frame: pd.DataFrame, weights: np.ndarray, audit: dict[str, Any], lock_hash: str, matrix_hash: str, audit_hash: str) -> tuple[dict[str, Any], str]:
    path = OUT / "state" / dataset / f"checkpoint_04_model_{dataset}.json"
    if path.exists():
        saved = read_json(path)
        require(saved.get("input_lock_sha256") == lock_hash, f"model_input_lock_mismatch:{dataset}")
        require(saved.get("matrix_sha256") == matrix_hash, f"model_matrix_mismatch:{dataset}")
        require(saved.get("weight_audit_sha256") == audit_hash, f"model_weight_audit_mismatch:{dataset}")
        return saved["model"], digest(path)
    model = fit_model(frame, weights, audit)
    value = {
        "stage": "time_stratified_common_support_d1_v1_model",
        "dataset": dataset,
        "created_at_utc": now(),
        "input_lock_sha256": lock_hash,
        "matrix_sha256": matrix_hash,
        "weight_audit_sha256": audit_hash,
        "model": model,
    }
    write_json_new(path, value)
    return model, digest(path)


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


def write_or_verify(path: Path, value: dict[str, Any], lock_hash: str) -> dict[str, Any]:
    if path.exists():
        saved = read_json(path)
        require(saved.get("input_lock_sha256") == lock_hash, f"checkpoint_input_lock_mismatch:{path}")
        return saved
    write_json_new(path, value)
    return value


def gates_for(dataset: str, preflight: dict[str, Any], audit: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    g3_status = "pass" if model["g3_pass"] else "fail"
    model_release = "pass" if all(preflight[name]["status"] == "pass" for name in ("G0", "G1", "G2")) and model["g3_pass"] else "fail"
    return {
        "G0": {**preflight["G0"], "metrics": {"e1_path_access_count": 0, "source_hash_audit": verify_source_hashes()}},
        "G1": {**preflight["G1"], "metrics": preflight["metrics"]},
        "G2": {**preflight["G2"], "metrics": {"cells": preflight["metrics"]["cells"]}},
        "G3": {
            "status": g3_status,
            "reason": "weighted_two_way_crv1_interaction_and_precision_thresholds",
            "metrics": model,
        },
        "G4": {"status": "pending", "reason": "no_independent_locked_calibration_sample"},
        "G5": {"status": "pending", "reason": "no_complete_opportunity_model", "residual_centering": "none"},
        "overall": {
            "model_release": model_release,
            "support_release": "pass" if preflight["G2"]["status"] == "pass" else "blocked",
            "interpretation_release": "observed_opportunity_association_only" if model_release == "pass" else "blocked",
            "forbidden_interpretations": ["round_boundary_effect", "state_effect", "causal_effect"],
        },
    }


def output_hashes() -> dict[str, str]:
    paths = [
        path for path in sorted(OUT.rglob("*"))
        if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"} and not any(part.startswith("failed_") for part in path.relative_to(OUT).parts)
    ]
    return {str(path.relative_to(OUT)): digest(path) for path in paths}


def run(selected: tuple[str, ...]) -> None:
    _, lock_hash = input_lock()
    matrices: dict[str, pd.DataFrame] = {}
    preflights: dict[str, dict[str, Any]] = {}
    audits: dict[str, dict[str, Any]] = {}
    models: dict[str, dict[str, Any]] = {}
    for dataset in selected:
        frame, preflight = support_preflight(dataset, lock_hash)
        require(all(preflight[name]["status"] == "pass" for name in ("G0", "G1", "G2")), f"preflight_gate_failed:{dataset}")
        weights, audit, audit_hash = weight_audit(frame, dataset, lock_hash, preflight["matrix_sha256"])
        model, _ = model_or_checkpoint(dataset, frame, weights, audit, lock_hash, preflight["matrix_sha256"], audit_hash)
        matrices[dataset] = frame
        preflights[dataset] = preflight
        audits[dataset] = audit
        models[dataset] = model
    if set(selected) != set(DATASETS):
        print(json.dumps({"status": "partial_checkpoint_complete", "datasets": selected, "output_root": str(OUT)}, ensure_ascii=True))
        return

    require(all(model["g3_pass"] for model in models.values()), "model_gate_failed")
    adjusted = holm(models)
    for dataset, model in models.items():
        model["holm_adjusted_delta_p_value_three_partition_family"] = adjusted[dataset]
    gates = {dataset: gates_for(dataset, preflights[dataset], audits[dataset], models[dataset]) for dataset in DATASETS}
    require(all(value["overall"]["model_release"] == "pass" for value in gates.values()), "model_release_failed")
    write_or_verify(OUT / "checkpoint_05_gate_audit.json", {
        "stage": "time_stratified_common_support_d1_v1_gate_audit",
        "created_at_utc": now(),
        "input_lock_sha256": lock_hash,
        "gates": gates,
    }, lock_hash)
    write_or_verify(OUT / "checkpoint_06_models.json", {
        "stage": "time_stratified_common_support_d1_v1_models",
        "created_at_utc": now(),
        "input_lock_sha256": lock_hash,
        "models": models,
        "inference": "weighted_two_way_crv1_match_and_focal_player_normal_approximation; holm_three_partition_delta_family",
    }, lock_hash)
    hashes_path = OUT / "file_sha256.json"
    hashes = output_hashes()
    if hashes_path.exists():
        require(read_json(hashes_path) == hashes, "output_file_sha256_mismatch")
    else:
        write_json_new(hashes_path, hashes)
    manifest_path = OUT / "manifest.json"
    manifest = {
        "stage": "time_stratified_common_support_d1_v1",
        "status": "passed",
        "created_at_utc": now(),
        "classification": "exploratory_nonconfirmatory_d1_only",
        "input_lock_sha256": lock_hash,
        "gate_statuses": {dataset: {name: value["status"] for name, value in gates[dataset].items() if name.startswith("G")} for dataset in DATASETS},
        "interpretation": "D1-only time-stratified common-support observed-opportunity association; not a round-boundary, state, causal, calibration, or predictive-gain conclusion.",
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
    selected = tuple(args.datasets)
    try:
        run(selected)
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True)
        failed = OUT / f"failed_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        failed.mkdir(exist_ok=False)
        write_json_new(failed / "failure_manifest.json", {
            "stage": "time_stratified_common_support_d1_v1",
            "status": "failed",
            "created_at_utc": now(),
            "error_type": type(error).__name__,
            "reason": str(error),
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
