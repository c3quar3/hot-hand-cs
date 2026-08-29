"""Execute the frozen E1-v2 exponential multi-event residual analysis."""

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
DATASET = "E1-v2"
ASSET_ROOT = ROOT / "results" / "conditional_residual_asset_e1_v1"
SOURCE = ASSET_ROOT / DATASET / "residual_asset.parquet"
BASE_ROOT = ROOT / "results" / "first_order_residual_continuous_time_e1_v1"
BASE_MATRIX = BASE_ROOT / "state" / DATASET / "analysis_matrix.parquet"
OUT = ROOT / "results" / "exponential_multi_event_residual_e1_v1"
DRAFT = ROOT / "docs" / "修订与补丁" / "33_E1v2指数衰减多事件历史残差分析协议_草案_v1_2026-08-26.md"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "34_E1v2指数衰减多事件历史残差执行协议_v1_2026-08-26.md"
HALF_LIVES_SECONDS = (2.0, 10.0, 60.0)
TICK_RATE = 64.0
BASE_MATRIX_SHA256 = "7bd6851bcbe8e601cc1244cdc24c2b3d9757b6c4158bd84aabc10bf3548cdb92"
BASE_COLUMNS = ["e_t", "H_t", "match_id", "focal_player_id", "event_tick", "history_source_event_tick", "time_since_history_tick"]


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


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


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def locked_inputs() -> list[Path]:
    return [
        DRAFT,
        PROTOCOL,
        Path(__file__).resolve(),
        SOURCE,
        ASSET_ROOT / "manifest.json",
        ASSET_ROOT / "file_sha256.json",
        ASSET_ROOT / "input_lock.json",
        ASSET_ROOT / "label_access_receipt.json",
        ASSET_ROOT / "field_dictionary.json",
        BASE_MATRIX,
        BASE_ROOT / "checkpoint_01_input_lock.json",
        BASE_ROOT / "checkpoint_02_preflight_E1-v2.json",
        BASE_ROOT / "manifest.json",
    ]


def ensure_input_lock() -> tuple[dict[str, Any], str]:
    path = OUT / "checkpoint_01_input_lock.json"
    inputs = {str(item.relative_to(ROOT)): digest(item) for item in locked_inputs()}
    value = {
        "stage": "exponential_multi_event_e1_v1_input_lock",
        "created_at_utc": now(),
        "dataset": DATASET,
        "classification": "exploratory_nonconfirmatory",
        "outcome_access": "existing_authorized_e1_residual_asset_only_no_raw_or_rescoring_access",
        "inputs": inputs,
        "half_lives_seconds": list(HALF_LIVES_SECONDS),
        "tick_rate": TICK_RATE,
        "base_matrix_sha256_expected": BASE_MATRIX_SHA256,
    }
    if path.exists():
        saved = read_json(path)
        require(saved.get("inputs") == inputs, "input_lock_hash_mismatch")
        require(saved.get("half_lives_seconds") == list(HALF_LIVES_SECONDS), "input_lock_half_lives_mismatch")
        return saved, digest(path)
    write_json_new(path, value)
    return value, digest(path)


def source_audit() -> dict[str, Any]:
    manifest = read_json(ASSET_ROOT / "manifest.json")
    hashes = read_json(ASSET_ROOT / "file_sha256.json")
    require(manifest.get("status") == "passed", "residual_asset_not_passed")
    require(manifest.get("partition", {}).get("rows") == 77169, "residual_asset_row_count_mismatch")
    require(digest(SOURCE) == "e5c75c07cf8bc385eec0af848110e8c9d8ce3dcdf15077784915301e2a243832", "residual_asset_hash_mismatch")
    require(digest(BASE_MATRIX) == BASE_MATRIX_SHA256, "base_matrix_hash_mismatch")
    return {
        "asset_manifest_status": manifest.get("status"),
        "asset_rows": manifest.get("partition", {}).get("rows"),
        "asset_file_sha_entries": len(hashes),
        "source_residual_asset_sha256": digest(SOURCE),
        "base_matrix_sha256": digest(BASE_MATRIX),
    }


def validate_asset(asset: pd.DataFrame) -> dict[str, Any]:
    required = {
        "physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id", "opponent_player_id",
        "e_t", "residual_valid", "primary_analysis_eligible", "p_support_0p01",
    }
    require(required.issubset(asset.columns), "asset_required_fields_missing")
    require(len(asset) == 77169, "asset_row_count_mismatch")
    require(asset["physical_event_id"].is_unique, "asset_physical_event_not_unique")
    require(asset["residual_valid"].all() and asset["primary_analysis_eligible"].all() and asset["p_support_0p01"].all(), "asset_eligibility_mismatch")
    values = asset[["e_t", "event_tick"]].to_numpy(dtype=float)
    require(np.isfinite(values).all(), "asset_nonfinite_values")
    return {"rows": int(len(asset)), "physical_event_unique": True, "all_primary_eligible": True}


def rebuild_frozen_current_rows(asset: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    base = asset.loc[:, ["physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id", "opponent_player_id", "e_t"]].copy()
    focal = base[["physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id", "e_t"]].rename(columns={"focal_player_id": "player_id", "e_t": "signed_e"})
    opponent = base[["physical_event_id", "match_id", "map_id", "event_tick", "opponent_player_id", "e_t"]].rename(columns={"opponent_player_id": "player_id", "e_t": "signed_e"})
    opponent["signed_e"] = -opponent["signed_e"]
    events = pd.concat([focal, opponent], ignore_index=True)
    events["player_id"] = events["player_id"].astype(str)
    events = events.sort_values(["match_id", "map_id", "player_id", "event_tick", "physical_event_id"], kind="stable")
    previous: dict[tuple[str, str], dict[str, Any] | None] = {}
    ambiguous: set[tuple[str, str]] = set()
    for _, group in events.groupby(["match_id", "map_id", "player_id"], sort=False, dropna=False):
        last: dict[str, Any] | None = None
        blocked = False
        for _, tick_group in group.groupby("event_tick", sort=False, dropna=False):
            for row in tick_group.itertuples(index=False):
                key = (str(row.physical_event_id), str(row.player_id))
                previous[key] = last
                if last is None and blocked:
                    ambiguous.add(key)
            if len(tick_group) == 1:
                row = tick_group.iloc[0]
                last = {"physical_event_id": str(row["physical_event_id"]), "event_tick": float(row["event_tick"]), "signed_e": float(row["signed_e"])}
                blocked = False
            else:
                last = None
                blocked = True
    state = base.copy()
    state["history_source_event_tick"] = np.nan
    state["history_source_physical_event_id"] = None
    state["H_t"] = np.nan
    state["history_missing_code"] = None
    for index, row in state.iterrows():
        prior = previous.get((str(row["physical_event_id"]), str(row["focal_player_id"])))
        if prior is None:
            key = (str(row["physical_event_id"]), str(row["focal_player_id"]))
            state.at[index, "history_missing_code"] = "ambiguous_concurrent_prior_tick" if key in ambiguous else "no_prior_same_map_eligible_event"
            continue
        require(prior["event_tick"] < float(row["event_tick"]), "non_strict_frozen_history")
        state.at[index, "history_source_event_tick"] = prior["event_tick"]
        state.at[index, "history_source_physical_event_id"] = prior["physical_event_id"]
        state.at[index, "H_t"] = prior["signed_e"]
    state["time_since_history_tick"] = state["event_tick"] - state["history_source_event_tick"]
    eligible = state.loc[state["H_t"].notna()].copy()
    frozen = eligible.rename(columns={"focal_player_id": "focal_player_id"})
    rebuilt = frozen[["e_t", "H_t", "match_id", "focal_player_id", "event_tick", "history_source_event_tick", "time_since_history_tick"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt, reference.reset_index(drop=True), check_dtype=True, check_exact=True)
    require(len(eligible) == 74321, "frozen_current_row_count_mismatch")
    return eligible


def build_decay_state(asset: pd.DataFrame, current: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = asset[["physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id", "opponent_player_id", "e_t"]].copy()
    focal = base[["physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id", "e_t"]].rename(columns={"focal_player_id": "player_id", "e_t": "signed_e"})
    opponent = base[["physical_event_id", "match_id", "map_id", "event_tick", "opponent_player_id", "e_t"]].rename(columns={"opponent_player_id": "player_id", "e_t": "signed_e"})
    opponent["signed_e"] = -opponent["signed_e"]
    events = pd.concat([focal, opponent], ignore_index=True)
    events["player_id"] = events["player_id"].astype(str)
    events = events.sort_values(["match_id", "map_id", "player_id", "event_tick", "physical_event_id"], kind="stable")
    target_keys = {(str(row.physical_event_id), str(row.focal_player_id)) for row in current.itertuples(index=False)}
    states: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, int]] = {}
    maximum_direct_error = np.zeros(len(HALF_LIVES_SECONDS), dtype=float)
    concurrent_batches = 0
    for _, group in events.groupby(["match_id", "map_id", "player_id"], sort=False, dropna=False):
        running_s = np.zeros(len(HALF_LIVES_SECONDS), dtype=float)
        running_v = np.zeros(len(HALF_LIVES_SECONDS), dtype=float)
        prior_tick: int | None = None
        history_ticks: list[float] = []
        history_e: list[float] = []
        for tick, tick_group in group.groupby("event_tick", sort=False, dropna=False):
            tick_value = float(tick)
            if prior_tick is None:
                pre_s = np.zeros(len(HALF_LIVES_SECONDS), dtype=float)
                pre_v = np.zeros(len(HALF_LIVES_SECONDS), dtype=float)
            else:
                delta_seconds = (tick_value - prior_tick) / TICK_RATE
                q = np.power(2.0, -delta_seconds / np.asarray(HALF_LIVES_SECONDS))
                pre_s = q * running_s
                pre_v = q * q * running_v
            if len(tick_group) > 1:
                concurrent_batches += 1
            for row in tick_group.itertuples(index=False):
                key = (str(row.physical_event_id), str(row.player_id))
                states[key] = (pre_s.copy(), pre_v.copy(), len(history_e))
                if key in target_keys:
                    if history_e:
                        old_ticks = np.asarray(history_ticks, dtype=float)
                        old_e = np.asarray(history_e, dtype=float)
                        direct = np.array([
                            float(np.sum(old_e * np.power(2.0, -(tick_value - old_ticks) / TICK_RATE / half_life)))
                            for half_life in HALF_LIVES_SECONDS
                        ])
                    else:
                        direct = np.zeros(len(HALF_LIVES_SECONDS), dtype=float)
                    maximum_direct_error = np.maximum(maximum_direct_error, np.abs(direct - pre_s))
            running_s = pre_s + float(tick_group["signed_e"].sum())
            running_v = pre_v + float(len(tick_group))
            history_ticks.extend([tick_value] * len(tick_group))
            history_e.extend(tick_group["signed_e"].astype(float).tolist())
            prior_tick = int(tick)
    state = current.copy()
    for position, half_life in enumerate(HALF_LIVES_SECONDS):
        values = [states[(str(row.physical_event_id), str(row.focal_player_id))] for row in state.itertuples(index=False)]
        state[f"S_h{int(half_life)}"] = [float(value[0][position]) for value in values]
        state[f"V_h{int(half_life)}"] = [float(value[1][position]) for value in values]
    state["strict_prior_event_count"] = [int(states[(str(row.physical_event_id), str(row.focal_player_id))][2]) for row in state.itertuples(index=False)]
    for half_life in HALF_LIVES_SECONDS:
        suffix = f"h{int(half_life)}"
        q = np.power(2.0, -state["time_since_history_tick"].to_numpy(dtype=float) / TICK_RATE / half_life)
        state[f"L_{suffix}"] = q * state["H_t"].to_numpy(dtype=float)
        state[f"A_{suffix}"] = state[f"S_{suffix}"].to_numpy(dtype=float) - state[f"L_{suffix}"].to_numpy(dtype=float)
        state[f"tail_eligible_{suffix}"] = state["strict_prior_event_count"].ge(2)
    require(np.isfinite(state.filter(regex=r"^(S|V|L|A)_").to_numpy(dtype=float)).all(), "nonfinite_decay_state")
    require((state.filter(regex=r"^V_") >= 0).to_numpy().all(), "negative_squared_weight_sum")
    audit = {
        "state_rows": int(len(state)),
        "concurrent_tick_batches": int(concurrent_batches),
        "maximum_direct_sum_error_by_half_life": {str(int(h)): float(error) for h, error in zip(HALF_LIVES_SECONDS, maximum_direct_error)},
        "tail_eligible_rows_by_half_life": {str(int(h)): int(state[f"tail_eligible_h{int(h)}"].sum()) for h in HALF_LIVES_SECONDS},
        "tail_ineligible_reason": "fewer_than_two_strict_prior_effective_events",
    }
    require(maximum_direct_error.max() <= 1e-10, "direct_sum_recursion_mismatch")
    return state, audit


def load_or_build(lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    state_path = OUT / "state" / DATASET / "exponential_state_matrix.parquet"
    checkpoint = OUT / "checkpoint_02_preflight_E1-v2.json"
    if state_path.exists() or checkpoint.exists():
        require(state_path.exists() and checkpoint.exists(), "incomplete_preflight_checkpoint")
        saved = read_json(checkpoint)
        require(saved.get("input_lock_sha256") == lock_hash, "preflight_lock_mismatch")
        require(saved.get("state_matrix_sha256") == digest(state_path), "preflight_state_hash_mismatch")
        return pd.read_parquet(state_path), saved
    audit = source_audit()
    asset = pd.read_parquet(SOURCE)
    field_audit = validate_asset(asset)
    reference = pd.read_parquet(BASE_MATRIX)
    require(list(reference.columns) == BASE_COLUMNS and len(reference) == 74321, "base_matrix_structure_mismatch")
    current = rebuild_frozen_current_rows(asset, reference)
    state, state_audit = build_decay_state(asset, current)
    tail_counts = state_audit["tail_eligible_rows_by_half_life"]
    support = {key: value / len(state) for key, value in tail_counts.items()}
    write_parquet_new(state_path, state)
    g2_pass = all(value >= 0.80 for value in support.values()) and state["match_id"].nunique() >= 30 and state["focal_player_id"].nunique() >= 30
    value = {
        "stage": "exponential_multi_event_e1_v1_preflight",
        "created_at_utc": now(),
        "dataset": DATASET,
        "input_lock_sha256": lock_hash,
        "state_matrix_sha256": digest(state_path),
        "G0": {"status": "pass", "reason": "authorized_e1_asset_and_base_matrix_hash_chain", "metrics": audit},
        "G1": {"status": "pass", "reason": "frozen_current_rows_rebuilt_and_strict_batch_state_verified", "metrics": {"field_audit": field_audit, "state_audit": state_audit}},
        "G2": {"status": "pass" if g2_pass else "fail", "reason": "fixed_current_rows_tail_support_and_cluster_support", "metrics": {"tail_support_fraction": support, "match_clusters": int(state["match_id"].nunique()), "player_clusters": int(state["focal_player_id"].nunique())}},
    }
    write_json_new(checkpoint, value)
    return state, value


def contrast(coefficient: np.ndarray, covariance: np.ndarray, vector: np.ndarray) -> dict[str, float]:
    variance = float(vector @ covariance @ vector)
    require(math.isfinite(variance) and variance > 0, "invalid_contrast_variance")
    se = math.sqrt(variance)
    estimate = float(vector @ coefficient)
    return {"estimate": estimate, "standard_error": se, "ci95_low": estimate - 1.959963984540054 * se, "ci95_high": estimate + 1.959963984540054 * se}


def fit(frame: pd.DataFrame, half_life: float) -> dict[str, Any]:
    suffix = f"h{int(half_life)}"
    sample = frame.loc[frame[f"tail_eligible_{suffix}"].copy()].copy()
    n = len(sample)
    l_scale = float(sample[f"L_{suffix}"].std(ddof=1))
    a_scale = float(sample[f"A_{suffix}"].std(ddof=1))
    require(math.isfinite(l_scale) and l_scale > 0 and math.isfinite(a_scale) and a_scale > 0, "nonpositive_state_scale")
    z = np.log2(1 + sample["time_since_history_tick"].to_numpy(dtype=float))
    mu = float(z.mean())
    z = z - mu
    l = sample[f"L_{suffix}"].to_numpy(dtype=float) / l_scale
    a = sample[f"A_{suffix}"].to_numpy(dtype=float) / a_scale
    design = np.column_stack([np.ones(n), z, l, l * z, a, a * z])
    names = ["alpha", "a_time_main_effect", "b_L_decayed_lag1", "k_L_time_modification", "b_A_earlier_tail", "k_A_time_modification"]
    k = design.shape[1]
    require(n > k and np.linalg.matrix_rank(design) == k, "rank_deficient_decay_design")
    inverse = np.linalg.inv(design.T @ design)
    response = sample["e_t"].to_numpy(dtype=float)
    coefficient = inverse @ design.T @ response
    residual = response - design @ coefficient
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)

    def meat(groups: pd.Series) -> tuple[np.ndarray, int]:
        _, codes = np.unique(groups.astype(str).to_numpy(), return_inverse=True)
        summed = np.zeros((int(codes.max()) + 1, k))
        np.add.at(summed, codes, design * residual[:, None])
        return summed.T @ summed, len(summed)

    match_meat, match_clusters = meat(sample["match_id"])
    player_meat, player_clusters = meat(sample["focal_player_id"])
    intersection_meat, intersection_clusters = meat(sample["match_id"].astype(str) + "\x1f" + sample["focal_player_id"].astype(str))
    correction = lambda groups: (groups / (groups - 1)) * ((n - 1) / (n - k))
    covariance = inverse @ (correction(match_clusters) * match_meat + correction(player_clusters) * player_meat - correction(intersection_clusters) * intersection_meat) @ inverse
    covariance = (covariance + covariance.T) / 2
    g3_pass = bool(np.isfinite(covariance).all() and covariance[4, 4] > 0 and match_clusters >= 30 and player_clusters >= 30 and leverage.max() <= 0.20)
    result: dict[str, Any] = {
        "half_life_seconds": half_life,
        "n": int(n),
        "tail_support_fraction": float(n / len(frame)),
        "state_scales": {"L_standard_deviation": l_scale, "A_standard_deviation": a_scale},
        "time_mu": mu,
        "coefficient_names": names,
        "coefficients": {name: float(value) for name, value in zip(names, coefficient)},
        "match_clusters": int(match_clusters),
        "player_clusters": int(player_clusters),
        "intersection_clusters": int(intersection_clusters),
        "max_leverage": float(leverage.max()),
        "covariance": covariance.tolist(),
        "g3_pass": g3_pass,
    }
    if not g3_pass:
        return result
    b_a = contrast(coefficient, covariance, np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    z_value = b_a["estimate"] / b_a["standard_error"]
    critical = NormalDist().inv_cdf(1 - 0.05 / 2)
    result.update({
        "b_A": {**b_a, "z_value": z_value, "p_value": float(math.erfc(abs(z_value) / math.sqrt(2)))},
        "ci_half_width_95": 1.959963984540054 * b_a["standard_error"],
        "mde_80_single": (critical + NormalDist().inv_cdf(0.80)) * b_a["standard_error"],
    })
    return result


def smoke(lock_hash: str, state: pd.DataFrame, preflight: dict[str, Any]) -> dict[str, Any]:
    path = OUT / "checkpoint_03_smoke_E1-v2.json"
    state_hash = digest(OUT / "state" / DATASET / "exponential_state_matrix.parquet")
    if path.exists():
        saved = read_json(path)
        require(saved.get("input_lock_sha256") == lock_hash and saved.get("state_matrix_sha256") == state_hash, "smoke_checkpoint_mismatch")
        return saved
    matches = sorted(state["match_id"].astype(str).unique())[:30]
    subset = state.loc[state["match_id"].astype(str).isin(matches)].copy()
    models = {str(int(h)): fit(subset, h) for h in HALF_LIVES_SECONDS}
    g3_pass = all(model["g3_pass"] for model in models.values())
    metrics = {label: {key: model[key] for key in ("n", "tail_support_fraction", "match_clusters", "player_clusters", "intersection_clusters", "max_leverage", "g3_pass")} for label, model in models.items()}
    value = {
        "stage": "exponential_multi_event_e1_v1_smoke",
        "created_at_utc": now(),
        "dataset": DATASET,
        "input_lock_sha256": lock_hash,
        "state_matrix_sha256": state_hash,
        "subset_rule": "30_lexicographically_smallest_match_id_clusters",
        "subset_rows": int(len(subset)),
        "G0": preflight["G0"],
        "G1": preflight["G1"],
        "G2": {"status": "pass" if subset["match_id"].nunique() >= 30 and subset["focal_player_id"].nunique() >= 30 else "fail", "reason": "smoke_cluster_support"},
        "G3": {"status": "pass" if g3_pass else "fail", "reason": "smoke_models_estimable", "metrics": metrics},
    }
    write_json_new(path, value)
    return value


def holm_adjust(models: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(models, key=lambda key: models[key]["b_A"]["p_value"])
    running = 0.0
    total = len(ordered)
    for position, key in enumerate(ordered):
        adjusted = min(1.0, (total - position) * models[key]["b_A"]["p_value"])
        running = max(running, adjusted)
        models[key]["b_A"]["holm_p_value"] = running


def output_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(OUT)): digest(path)
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"} and not any(part.startswith("failed_") for part in path.relative_to(OUT).parts)
    }


def run(phase: str) -> None:
    _, lock_hash = ensure_input_lock()
    state, preflight = load_or_build(lock_hash)
    require(all(preflight[name]["status"] == "pass" for name in ("G0", "G1", "G2")), "preflight_gate_failed")
    smoke_value = smoke(lock_hash, state, preflight)
    require(all(smoke_value[name]["status"] == "pass" for name in ("G0", "G1", "G2", "G3")), "smoke_gate_failed")
    if phase == "smoke":
        print(json.dumps({"status": "smoke_passed", "output_root": str(OUT)}, ensure_ascii=True))
        return
    model_path = OUT / "checkpoint_04_models_E1-v2.json"
    state_hash = digest(OUT / "state" / DATASET / "exponential_state_matrix.parquet")
    if model_path.exists():
        saved = read_json(model_path)
        require(saved.get("input_lock_sha256") == lock_hash and saved.get("state_matrix_sha256") == state_hash, "model_checkpoint_mismatch")
        models = saved["models"]
    else:
        models = {str(int(h)): fit(state, h) for h in HALF_LIVES_SECONDS}
        require(all(model["g3_pass"] for model in models.values()), "model_gate_failed")
        holm_adjust(models)
        write_json_new(model_path, {"stage": "exponential_multi_event_e1_v1_models", "created_at_utc": now(), "dataset": DATASET, "input_lock_sha256": lock_hash, "state_matrix_sha256": state_hash, "models": models})
    require(all(model["g3_pass"] for model in models.values()), "model_gate_failed")
    g4_items = {
        label: {
            "status": "pass" if model["ci_half_width_95"] <= 0.02 and model["mde_80_single"] <= 0.03 else "pending",
            "ci_half_width_95": model["ci_half_width_95"],
            "mde_80_single": model["mde_80_single"],
        }
        for label, model in models.items()
    }
    gates = {
        "G0": preflight["G0"],
        "G1": preflight["G1"],
        "G2": preflight["G2"],
        "G3": {"status": "pass", "reason": "all_half_life_models_estimable_with_two_way_crv1", "metrics": {label: {key: model[key] for key in ("n", "match_clusters", "player_clusters", "max_leverage")} for label, model in models.items()}},
        "G4": {"status": "pass" if all(item["status"] == "pass" for item in g4_items.values()) else "pending", "reason": "per_half_life_precision_thresholds", "items": g4_items},
        "G5": {"status": "pending", "reason": "independent_confirmation_requires_post_report_user_authorization"},
        "overall": {"model_release": "pass", "interpretation_release": "exploratory_observed_sequence_incremental_association_only", "forbidden_interpretations": ["confirmation_upgrade", "state_effect", "causal_effect", "opportunity_process_claim", "predictive_gain"]},
    }
    gate_path = OUT / "checkpoint_05_gate_audit.json"
    if gate_path.exists():
        require(read_json(gate_path).get("input_lock_sha256") == lock_hash, "gate_checkpoint_lock_mismatch")
    else:
        write_json_new(gate_path, {"stage": "exponential_multi_event_e1_v1_gate_audit", "created_at_utc": now(), "input_lock_sha256": lock_hash, "gates": gates})
    summary_path = OUT / "checkpoint_06_models.json"
    if summary_path.exists():
        require(read_json(summary_path).get("input_lock_sha256") == lock_hash, "summary_checkpoint_lock_mismatch")
    else:
        write_json_new(summary_path, {"stage": "exponential_multi_event_e1_v1_summary", "created_at_utc": now(), "input_lock_sha256": lock_hash, "models": models, "inference": "unweighted_ols_two_way_crv1_match_and_focal_player_normal_approximation; three_half_life_holm_family_for_b_A_only"})
    hashes_path = OUT / "file_sha256.json"
    hashes = output_hashes()
    if hashes_path.exists():
        require(read_json(hashes_path) == hashes, "output_file_sha256_mismatch")
    else:
        write_json_new(hashes_path, hashes)
    manifest_path = OUT / "manifest.json"
    manifest = {"stage": "exponential_multi_event_e1_v1", "status": "passed", "created_at_utc": now(), "dataset": DATASET, "classification": "exploratory_nonconfirmatory", "input_lock_sha256": lock_hash, "gate_statuses": {name: value["status"] for name, value in gates.items() if name.startswith("G")}, "interpretation": "E1-v2 observed continuous engagement sequence only; exploratory incremental association of earlier exponentially decayed residual history conditional on decayed lag-1; not confirmation, state, causal, opportunity-process, or predictive-gain evidence.", "file_sha256_sha256": digest(hashes_path)}
    if manifest_path.exists():
        saved = read_json(manifest_path)
        require(saved.get("input_lock_sha256") == lock_hash and saved.get("file_sha256_sha256") == digest(hashes_path), "manifest_resume_mismatch")
    else:
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
        write_json_new(failed / "failure_manifest.json", {"stage": "exponential_multi_event_e1_v1", "status": "failed", "created_at_utc": now(), "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
