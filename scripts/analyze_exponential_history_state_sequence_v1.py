"""Run the frozen 60-second exponential-history state sequence."""

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
OUT = ROOT / "results" / "exponential_history_state_sequence_v1"
ANALYSIS = ROOT / "docs" / "修订与补丁" / "49_60秒指数衰减历史状态顺序检验分析协议_v1_2026-08-26.md"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "50_60秒指数衰减历史状态顺序检验执行协议_v1_2026-08-26.md"
HALF_LIFE_SECONDS = 60.0
TICK_RATE = 64.0
DATASETS = ("D1-A", "D1-B", "D1-C", "E1-v2")
SOURCES = {
    "D1-A": ROOT / "results" / "conditional_residual_asset_d1_v2" / "D1-A" / "residual_asset.parquet",
    "D1-B": ROOT / "results" / "conditional_residual_asset_d1_v2" / "D1-B" / "residual_asset.parquet",
    "D1-C": ROOT / "results" / "conditional_residual_asset_d1_v2" / "D1-C" / "residual_asset.parquet",
    "E1-v2": ROOT / "results" / "conditional_residual_asset_e1_v1" / "E1-v2" / "residual_asset.parquet",
}
REQUIRED = {
    "physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id", "opponent_player_id",
    "e_t", "residual_valid", "primary_analysis_eligible", "p_support_0p01",
}
NORMAL = NormalDist()
Z95 = 1.959963984540054


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


def write_parquet_new(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    frame.to_parquet(path, index=False)


def source_paths() -> list[Path]:
    paths = [ANALYSIS, PROTOCOL, Path(__file__).resolve()]
    for root in (ROOT / "results" / "conditional_residual_asset_d1_v2", ROOT / "results" / "conditional_residual_asset_e1_v1"):
        paths.extend([root / "manifest.json", root / "file_sha256.json", root / "input_lock.json", root / "field_dictionary.json"])
    paths.extend(SOURCES.values())
    paths.extend([
        ROOT / "results" / "conditional_residual_asset_d1_v2" / "label_access_receipts" / "D1-B.json",
        ROOT / "results" / "conditional_residual_asset_d1_v2" / "label_access_receipts" / "D1-C.json",
        ROOT / "results" / "conditional_residual_asset_e1_v1" / "label_access_receipt.json",
    ])
    return paths


def ensure_input_lock() -> tuple[dict[str, Any], str]:
    path = OUT / "checkpoint_01_input_lock.json"
    inputs = {str(item.relative_to(ROOT)): digest(item) for item in source_paths()}
    value = {
        "stage": "exponential_history_state_sequence_v1_input_lock",
        "created_at_utc": now(),
        "classification": "authorized_exploratory_sequence_with_e1_specification_replication",
        "datasets": list(DATASETS),
        "roles": {"D1-A": "development_reference", "D1-B": "ordered_exploratory_test_1", "D1-C": "ordered_exploratory_test_2", "E1-v2": "confirmation_stage_specification_replication_not_newly_independent"},
        "outcome_access": "all_input_residual_assets_previously_materialized; no_raw_input_or_rescoring_access",
        "half_life_seconds": HALF_LIFE_SECONDS,
        "half_life_rationale": "one_half_of_the_approximately_120_second_regular_round_duration; fixed_without_using_existing_effect_or_loss_values",
        "tick_rate": TICK_RATE,
        "reference_dataset": "D1-A",
        "inputs": inputs,
    }
    if path.exists():
        saved = read_json(path)
        require(saved.get("inputs") == inputs, "input_lock_hash_mismatch")
        require(saved.get("half_life_seconds") == HALF_LIFE_SECONDS, "input_lock_half_life_mismatch")
        return saved, digest(path)
    write_json_new(path, value)
    return value, digest(path)


def validate_asset(asset: pd.DataFrame, dataset: str) -> dict[str, Any]:
    missing = sorted(REQUIRED - set(asset.columns))
    require(not missing, f"missing_columns:{dataset}:{','.join(missing)}")
    require(asset["physical_event_id"].is_unique, f"nonunique_physical_event_id:{dataset}")
    require(asset["residual_valid"].all(), f"invalid_residual:{dataset}")
    require(asset["primary_analysis_eligible"].all(), f"ineligible_primary_row:{dataset}")
    require(asset["p_support_0p01"].all(), f"outside_p_support:{dataset}")
    require((asset["focal_player_id"] != asset["opponent_player_id"]).all(), f"self_duel:{dataset}")
    values = asset[["e_t", "event_tick"]].to_numpy(dtype=float)
    require(np.isfinite(values).all(), f"nonfinite_residual_or_tick:{dataset}")
    return {"rows": int(len(asset)), "physical_event_id_unique": True, "all_primary_eligible": True}


def participant_ledger(asset: pd.DataFrame) -> pd.DataFrame:
    base = asset[["physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id", "opponent_player_id", "e_t"]].copy()
    focal = base[["physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id", "e_t"]].rename(columns={"focal_player_id": "player_id", "e_t": "signed_e"})
    opponent = base[["physical_event_id", "match_id", "map_id", "event_tick", "opponent_player_id", "e_t"]].rename(columns={"opponent_player_id": "player_id", "e_t": "signed_e"})
    opponent["signed_e"] = -opponent["signed_e"]
    ledger = pd.concat([focal, opponent], ignore_index=True)
    ledger["player_id"] = ledger["player_id"].astype(str)
    require(not ledger.duplicated(["physical_event_id", "player_id"]).any(), "duplicate_participant_physical_event")
    return ledger.sort_values(["match_id", "map_id", "player_id", "event_tick", "physical_event_id"], kind="stable").reset_index(drop=True)


def build_state(asset: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    ledger = participant_ledger(asset)
    rows: list[dict[str, Any]] = []
    maximum_direct_error = 0.0
    concurrent_batches = 0
    for (match_id, map_id, player_id), group in ledger.groupby(["match_id", "map_id", "player_id"], sort=False):
        running_state = 0.0
        prior_tick: float | None = None
        history_ticks: list[float] = []
        history_e: list[float] = []
        for tick, batch in group.groupby("event_tick", sort=False):
            tick_value = float(tick)
            if prior_tick is None:
                pre_state = 0.0
            else:
                q = 2.0 ** (-((tick_value - prior_tick) / TICK_RATE) / HALF_LIFE_SECONDS)
                pre_state = q * running_state
            if len(batch) > 1:
                concurrent_batches += 1
            if history_e:
                direct = float(np.sum(np.asarray(history_e) * np.power(2.0, -(tick_value - np.asarray(history_ticks)) / TICK_RATE / HALF_LIFE_SECONDS)))
                maximum_direct_error = max(maximum_direct_error, abs(direct - pre_state))
            for row in batch.itertuples(index=False):
                rows.append({
                    "physical_event_id": str(row.physical_event_id), "match_id": str(match_id), "map_id": str(map_id),
                    "player_id": str(player_id), "event_tick": tick_value, "e_t": float(row.signed_e),
                    "S_h60": pre_state, "strict_prior_event_count": len(history_e),
                    "prior_tick": prior_tick,
                })
            running_state = pre_state + float(batch["signed_e"].sum())
            history_ticks.extend([tick_value] * len(batch))
            history_e.extend(batch["signed_e"].astype(float).tolist())
            prior_tick = tick_value
    state = pd.DataFrame(rows)
    state["analysis_eligible"] = state["strict_prior_event_count"].ge(1)
    eligible = state.loc[state["analysis_eligible"]].copy()
    eligible["d_tick"] = eligible["event_tick"] - eligible["prior_tick"]
    require((eligible["d_tick"] >= 1).all(), f"nonpositive_history_interval:{dataset}")
    require(np.isfinite(eligible[["e_t", "S_h60", "d_tick"]].to_numpy(dtype=float)).all(), f"nonfinite_state:{dataset}")
    require(maximum_direct_error <= 1e-10, f"direct_sum_recursion_mismatch:{dataset}")
    audit = {
        "source_rows": int(len(asset)), "participant_rows": int(len(state)), "analysis_rows": int(len(eligible)),
        "concurrent_tick_batches": int(concurrent_batches), "maximum_direct_sum_error": float(maximum_direct_error),
        "match_clusters": int(eligible["match_id"].nunique()), "player_clusters": int(eligible["player_id"].nunique()),
        "excluded_no_strict_prior_history": int((~state["analysis_eligible"]).sum()),
    }
    return eligible.sort_values(["match_id", "map_id", "player_id", "event_tick", "physical_event_id"], kind="stable").reset_index(drop=True), audit


def load_or_build(dataset: str, lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    state_path = OUT / "state" / dataset / "state_matrix.parquet"
    checkpoint_path = OUT / "state" / dataset / "checkpoint_02_preflight.json"
    if state_path.exists() or checkpoint_path.exists():
        require(state_path.exists() and checkpoint_path.exists(), f"incomplete_state_checkpoint:{dataset}")
        checkpoint = read_json(checkpoint_path)
        require(checkpoint.get("input_lock_sha256") == lock_hash, f"state_lock_mismatch:{dataset}")
        require(checkpoint.get("state_matrix_sha256") == digest(state_path), f"state_hash_mismatch:{dataset}")
        return pd.read_parquet(state_path), checkpoint
    asset = pd.read_parquet(SOURCES[dataset])
    audit = validate_asset(asset, dataset)
    state, state_audit = build_state(asset, dataset)
    write_parquet_new(state_path, state)
    g0 = bool(audit["physical_event_id_unique"] and audit["all_primary_eligible"])
    g1 = bool(state_audit["maximum_direct_sum_error"] <= 1e-10 and state_audit["analysis_rows"] > 0)
    g2 = bool(state_audit["match_clusters"] >= 30 and state_audit["player_clusters"] >= 30)
    checkpoint = {
        "stage": "exponential_history_state_sequence_v1_preflight", "created_at_utc": now(), "dataset": dataset,
        "input_lock_sha256": lock_hash, "source_asset_sha256": digest(SOURCES[dataset]), "state_matrix_sha256": digest(state_path),
        "G0": {"status": "pass" if g0 else "fail", "reason": "eligible_unique_physical_event_asset", "metrics": audit},
        "G1": {"status": "pass" if g1 else "fail", "reason": "signed_participant_ledger_and_strict_batch_state", "metrics": state_audit},
        "G2": {"status": "pass" if g2 else "fail", "reason": "cluster_support", "metrics": state_audit},
    }
    write_json_new(checkpoint_path, checkpoint)
    return state, checkpoint


def reference_stats(frame: pd.DataFrame, lock_hash: str) -> tuple[dict[str, float], str]:
    path = OUT / "reference" / "D1-A_reference.json"
    z_raw = np.log2(1.0 + frame["d_tick"].to_numpy(dtype=float))
    state = frame["S_h60"].to_numpy(dtype=float)
    value = {
        "stage": "exponential_history_state_sequence_v1_reference", "created_at_utc": now(), "dataset": "D1-A",
        "input_lock_sha256": lock_hash, "rows": int(len(frame)),
        "mu_z": float(z_raw.mean()), "mu_S": float(state.mean()), "sigma_S_ddof1": float(state.std(ddof=1)),
        "S_star_p10": float(np.quantile((state - state.mean()) / state.std(ddof=1), 0.10)),
        "S_star_p90": float(np.quantile((state - state.mean()) / state.std(ddof=1), 0.90)),
        "p_hat_reference": 0.5,
    }
    require(math.isfinite(value["sigma_S_ddof1"]) and value["sigma_S_ddof1"] > 0, "invalid_reference_state_scale")
    if path.exists():
        saved = read_json(path)
        require(saved.get("input_lock_sha256") == lock_hash, "reference_lock_mismatch")
        for key in ("mu_z", "mu_S", "sigma_S_ddof1", "S_star_p10", "S_star_p90"):
            require(math.isclose(float(saved[key]), float(value[key]), rel_tol=0.0, abs_tol=1e-12), f"reference_value_mismatch:{key}")
        return {key: float(saved[key]) for key in ("mu_z", "mu_S", "sigma_S_ddof1", "S_star_p10", "S_star_p90")}, digest(path)
    write_json_new(path, value)
    return {key: float(value[key]) for key in ("mu_z", "mu_S", "sigma_S_ddof1", "S_star_p10", "S_star_p90")}, digest(path)


def fit(frame: pd.DataFrame, reference: dict[str, float], expose: bool) -> dict[str, Any]:
    n = len(frame)
    z = np.log2(1.0 + frame["d_tick"].to_numpy(dtype=float)) - reference["mu_z"]
    s = (frame["S_h60"].to_numpy(dtype=float) - reference["mu_S"]) / reference["sigma_S_ddof1"]
    design = np.column_stack([np.ones(n), z, s, s * z])
    names = ["alpha", "a_time", "beta_state", "kappa_state_time"]
    k = design.shape[1]
    require(n > k and np.linalg.matrix_rank(design) == k, "rank_deficient_design")
    inverse = np.linalg.inv(design.T @ design)
    response = frame["e_t"].to_numpy(dtype=float)
    coefficient = inverse @ design.T @ response
    residual = response - design @ coefficient
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)

    def meat(groups: pd.Series) -> tuple[np.ndarray, int]:
        _, codes = np.unique(groups.astype(str).to_numpy(), return_inverse=True)
        summed = np.zeros((int(codes.max()) + 1, k))
        np.add.at(summed, codes, design * residual[:, None])
        return summed.T @ summed, len(summed)

    match_meat, matches = meat(frame["match_id"])
    player_meat, players = meat(frame["player_id"])
    intersection_meat, intersections = meat(frame["match_id"].astype(str) + "\x1f" + frame["player_id"].astype(str))
    correction = lambda groups: (groups / (groups - 1)) * ((n - 1) / (n - k))
    covariance = inverse @ (correction(matches) * match_meat + correction(players) * player_meat - correction(intersections) * intersection_meat) @ inverse
    covariance = (covariance + covariance.T) / 2
    g3 = bool(np.isfinite(covariance).all() and covariance[2, 2] > 0 and matches >= 30 and players >= 30 and intersections >= 30 and leverage.max() <= 0.20)
    result: dict[str, Any] = {"n": int(n), "match_clusters": int(matches), "player_clusters": int(players), "intersection_clusters": int(intersections), "max_leverage": float(leverage.max()), "g3_pass": g3}
    if not expose or not g3:
        return result
    estimate = float(coefficient[2])
    se = math.sqrt(float(covariance[2, 2]))
    z_value = estimate / se
    result.update({
        "coefficient_names": names, "coefficients": {name: float(value) for name, value in zip(names, coefficient)},
        "beta": {"estimate": estimate, "standard_error": se, "z_value": z_value, "p_value": float(math.erfc(abs(z_value) / math.sqrt(2)),), "ci95_low": estimate - Z95 * se, "ci95_high": estimate + Z95 * se},
        "ci_half_width_95": Z95 * se, "mde_80_single": (Z95 + NORMAL.inv_cdf(0.80)) * se,
    })
    return result


def smoke(states: dict[str, pd.DataFrame], checkpoints: dict[str, dict[str, Any]], reference: dict[str, float], lock_hash: str, reference_hash: str) -> None:
    path = OUT / "checkpoint_03_smoke.json"
    state_hashes = {dataset: digest(OUT / "state" / dataset / "state_matrix.parquet") for dataset in DATASETS}
    if path.exists():
        saved = read_json(path)
        require(saved.get("input_lock_sha256") == lock_hash and saved.get("state_hashes") == state_hashes and saved.get("reference_sha256") == reference_hash, "smoke_checkpoint_mismatch")
        return
    metrics: dict[str, Any] = {}
    for dataset, state in states.items():
        matches = sorted(state["match_id"].unique())[:30]
        subset = state.loc[state["match_id"].isin(matches)].copy()
        metrics[dataset] = fit(subset, reference, expose=False)
    passed = all(value["g3_pass"] for value in metrics.values())
    value = {"stage": "exponential_history_state_sequence_v1_smoke", "created_at_utc": now(), "input_lock_sha256": lock_hash, "reference_sha256": reference_hash, "state_hashes": state_hashes, "subset_rule": "30_lexicographically_smallest_match_id_clusters_per_dataset", "G0_G2": {dataset: {gate: checkpoints[dataset][gate] for gate in ("G0", "G1", "G2")} for dataset in DATASETS}, "G3": {"status": "pass" if passed else "fail", "reason": "smoke_models_estimable_without_effect_output", "metrics": metrics}}
    write_json_new(path, value)
    require(passed, "smoke_g3_failed")


def holm_two(models: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(("D1-B", "D1-C"), key=lambda name: models[name]["beta"]["p_value"])
    running = 0.0
    for position, name in enumerate(ordered):
        adjusted = min(1.0, (2 - position) * models[name]["beta"]["p_value"])
        running = max(running, adjusted)
        models[name]["beta"]["holm_p_value_within_d1_tests"] = running


def output_hashes() -> dict[str, str]:
    return {str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"} and not any(part.startswith("failed_") for part in path.relative_to(OUT).parts)}


def run(phase: str) -> None:
    _, lock_hash = ensure_input_lock()
    states: dict[str, pd.DataFrame] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        states[dataset], checkpoints[dataset] = load_or_build(dataset, lock_hash)
        require(all(checkpoints[dataset][gate]["status"] == "pass" for gate in ("G0", "G1", "G2")), f"preflight_failed:{dataset}")
    reference, reference_hash = reference_stats(states["D1-A"], lock_hash)
    smoke(states, checkpoints, reference, lock_hash, reference_hash)
    if phase == "smoke":
        print(json.dumps({"status": "smoke_passed", "output_root": str(OUT)}, ensure_ascii=True))
        return
    models_path = OUT / "checkpoint_04_models.json"
    state_hashes = {dataset: digest(OUT / "state" / dataset / "state_matrix.parquet") for dataset in DATASETS}
    if models_path.exists():
        saved = read_json(models_path)
        require(saved.get("input_lock_sha256") == lock_hash and saved.get("reference_sha256") == reference_hash and saved.get("state_hashes") == state_hashes, "model_checkpoint_mismatch")
        models = saved["models"]
    else:
        models = {dataset: fit(states[dataset], reference, expose=True) for dataset in DATASETS}
        require(all(model["g3_pass"] for model in models.values()), "full_g3_failed")
        holm_two(models)
        write_json_new(models_path, {"stage": "exponential_history_state_sequence_v1_models", "created_at_utc": now(), "input_lock_sha256": lock_hash, "reference_sha256": reference_hash, "state_hashes": state_hashes, "models": models})
    d4 = {dataset: {"status": "pass" if model["ci_half_width_95"] <= 0.020 and model["mde_80_single"] <= 0.030 else "pending", "ci_half_width_95": model["ci_half_width_95"], "mde_80_single": model["mde_80_single"]} for dataset, model in models.items()}
    stage = {}
    for dataset, model in models.items():
        p_value = model["beta"].get("holm_p_value_within_d1_tests", model["beta"]["p_value"])
        stage[dataset] = "positive_association" if model["beta"]["estimate"] > 0 and p_value < 0.05 else "indeterminate"
    gates = {"D3": {"status": "pass", "reason": "all_models_estimable", "metrics": {dataset: {key: model[key] for key in ("n", "match_clusters", "player_clusters", "intersection_clusters", "max_leverage")} for dataset, model in models.items()}}, "D4": {"status": "pass" if all(value["status"] == "pass" for value in d4.values()) else "pending", "items": d4}, "D5": {"status": "pending", "reason": "all_test_and_replication_assets_had_prior_historical_outcome_access; no_new_independent_confirmation_claim"}, "stage_decisions": stage}
    gate_path = OUT / "checkpoint_05_gate_audit.json"
    if not gate_path.exists():
        write_json_new(gate_path, {"stage": "exponential_history_state_sequence_v1_gate_audit", "created_at_utc": now(), "input_lock_sha256": lock_hash, "gates": gates})
    summary_path = OUT / "checkpoint_06_summary.json"
    if not summary_path.exists():
        write_json_new(summary_path, {"stage": "exponential_history_state_sequence_v1_summary", "created_at_utc": now(), "input_lock_sha256": lock_hash, "reference": reference, "models": models, "inference": "unweighted_ols_two_way_crv1_match_and_player_normal_approximation; Holm_only_for_D1-B_and_D1-C"})
    hashes_path = OUT / "file_sha256.json"
    hashes = output_hashes()
    if hashes_path.exists():
        require(read_json(hashes_path) == hashes, "output_hash_mismatch")
    else:
        write_json_new(hashes_path, hashes)
    manifest_path = OUT / "manifest.json"
    manifest = {"stage": "exponential_history_state_sequence_v1", "status": "passed", "created_at_utc": now(), "classification": "authorized_exploratory_sequence_with_e1_specification_replication", "input_lock_sha256": lock_hash, "gate_statuses": {name: value["status"] for name, value in gates.items() if name.startswith("D")}, "interpretation": "Observed engagement-sequence association only. D1-B/C are historical-result-exposed exploratory tests; E1-v2 is a specified replication stage, not new independent confirmatory evidence. No causal, opportunity-process, state-mechanism, or p_hat predictive-gain claim.", "file_sha256_sha256": digest(hashes_path)}
    if not manifest_path.exists():
        write_json_new(manifest_path, manifest)
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
        write_json_new(failed / "failure_manifest.json", {"stage": "exponential_history_state_sequence_v1", "status": "failed", "created_at_utc": now(), "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
