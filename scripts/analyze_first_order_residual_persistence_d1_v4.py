"""Execute the frozen D1-only first-order residual persistence protocol v4."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MAIN = Path(__file__).resolve().parents[1]
PROTOCOL = MAIN / "docs" / "修订与补丁" / "16_一阶历史残差持续性执行协议_v4_D1_only_2026-08-26.md"
OUT = MAIN / "results" / "first_order_residual_persistence_d1_v4"
INPUT_ROOT = MAIN / "results" / "conditional_residual_asset_d1_v2"
INPUTS = {dataset: INPUT_ROOT / dataset / "residual_asset.parquet" for dataset in ("D1-A", "D1-B", "D1-C")}
INPUT_METADATA = [
    INPUT_ROOT / name for name in ("manifest.json", "file_sha256.json", "input_lock.json", "field_dictionary.json")
] + [
    INPUT_ROOT / dataset / name for dataset in INPUTS for name in ("checkpoint.json", "structural_temporal_audit.json")
]
REQUIRED_COLUMNS = {
    "dataset", "physical_event_id", "match_id", "map_id", "round_id", "event_tick", "anchor_tick",
    "focal_player_id", "opponent_player_id", "Y_t", "p_hat", "r_t", "e_t", "p_support_0p01",
    "residual_valid", "primary_analysis_eligible",
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required_input_missing:{path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_parquet_new(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def input_records() -> dict[str, dict[str, Any]]:
    paths = [PROTOCOL, Path(__file__).resolve(), *INPUTS.values(), *INPUT_METADATA]
    return {str(path): file_record(path) for path in paths}


def ensure_input_lock() -> str:
    expected = input_records()
    lock = OUT / "checkpoint_01_input_lock.json"
    if not OUT.exists():
        OUT.mkdir(parents=True)
    if not lock.exists():
        write_json_new(lock, {
            "stage": "first_order_residual_persistence_d1_v4_input_lock",
            "created_at_utc": now(),
            "classification": "exploratory_nonconfirmatory_d1_only",
            "protocol": str(PROTOCOL), "inputs": expected,
            "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__},
        })
    saved = json.loads(lock.read_text(encoding="utf-8"))
    require(saved.get("inputs") == expected, "resume_input_lock_hash_mismatch")
    require(not any("E1" in path or "e1" in path for path in saved["inputs"]), "e1_path_present_in_d1_input_lock")
    return digest(lock)


def write_failure(error: BaseException) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    folder = OUT / f"failed_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    folder.mkdir(exist_ok=False)
    write_json_new(folder / "failure_manifest.json", {
        "stage": "first_order_residual_persistence_d1_v4", "status": "failed", "created_at_utc": now(),
        "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc(),
    })


def validate_asset(dataset: str, frame: pd.DataFrame) -> dict[str, Any]:
    require(not (REQUIRED_COLUMNS - set(frame.columns)), f"missing_required_columns:{dataset}")
    require(frame["dataset"].astype(str).eq(dataset).all(), f"dataset_column_mismatch:{dataset}")
    require(frame["physical_event_id"].notna().all() and not frame["physical_event_id"].duplicated().any(), f"physical_event_id_not_unique:{dataset}")
    require(frame["focal_player_id"].notna().all() and frame["opponent_player_id"].notna().all(), f"missing_player_id:{dataset}")
    require(frame["focal_player_id"].astype(str).ne(frame["opponent_player_id"].astype(str)).all(), f"self_duel:{dataset}")
    require(frame["anchor_tick"].lt(frame["event_tick"]).all(), f"anchor_time_gate_failed:{dataset}")
    require(np.isfinite(frame[["event_tick", "anchor_tick", "Y_t", "p_hat", "r_t", "e_t"]].to_numpy(dtype=float)).all(), f"nonfinite_required_values:{dataset}")
    require(frame["Y_t"].isin([0, 1]).all(), f"nonbinary_outcome:{dataset}")
    require(frame["p_hat"].between(0, 1, inclusive="neither").all(), f"invalid_probability:{dataset}")
    eligible = frame["primary_analysis_eligible"].astype(bool) & frame["residual_valid"].astype(bool)
    require(frame.loc[eligible, "p_hat"].between(0.01, 0.99, inclusive="both").all(), f"eligible_support_gate_failed:{dataset}")
    return {"rows": int(len(frame)), "primary_eligible_rows": int(eligible.sum()), "physical_event_id_unique": True, "source_canonical_direction_inherited": True, "anchor_before_event": True}


def build_history(dataset: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    state = frame.copy()
    state["primary_eligible"] = state["primary_analysis_eligible"].astype(bool) & state["residual_valid"].astype(bool)
    state["primary_eligible"] &= state["p_hat"].between(0.01, 0.99, inclusive="both")
    source = state.loc[state["primary_eligible"]].copy()
    focal = source[["physical_event_id", "match_id", "map_id", "round_id", "event_tick", "focal_player_id", "e_t"]].rename(columns={"focal_player_id": "player_id", "e_t": "signed_e"})
    opponent = source[["physical_event_id", "match_id", "map_id", "round_id", "event_tick", "opponent_player_id", "e_t"]].rename(columns={"opponent_player_id": "player_id", "e_t": "signed_e"})
    opponent["signed_e"] = -opponent["signed_e"]
    player_events = pd.concat([focal, opponent], ignore_index=True)
    player_events["player_id"] = player_events["player_id"].astype(str)
    player_events = player_events.sort_values(["match_id", "map_id", "player_id", "event_tick", "physical_event_id"], kind="stable")

    previous: dict[tuple[str, str], dict[str, Any] | None] = {}
    ambiguous: set[tuple[str, str]] = set()
    for _, group in player_events.groupby(["match_id", "map_id", "player_id"], sort=False, dropna=False):
        last: dict[str, Any] | None = None
        blocked_by_concurrent_prior = False
        for _, same_tick in group.groupby("event_tick", sort=False, dropna=False):
            for row in same_tick.itertuples(index=False):
                key = (str(row.physical_event_id), str(row.player_id))
                previous[key] = last
                if last is None and blocked_by_concurrent_prior:
                    ambiguous.add(key)
            if len(same_tick) == 1:
                row = same_tick.iloc[0]
                last = {"physical_event_id": str(row["physical_event_id"]), "event_tick": float(row["event_tick"]), "round_id": str(row["round_id"]), "signed_e": float(row["signed_e"])}
                blocked_by_concurrent_prior = False
            else:
                last = None
                blocked_by_concurrent_prior = True

    state["history_source_physical_event_id"] = None
    state["history_source_event_tick"] = np.nan
    state["H_t"] = np.nan
    state["history_crosses_round"] = pd.NA
    state["history_missing_code"] = "current_not_primary_eligible"
    for idx, row in state.loc[state["primary_eligible"]].iterrows():
        key = (str(row["physical_event_id"]), str(row["focal_player_id"]))
        prior = previous.get(key)
        if prior is None:
            state.at[idx, "history_missing_code"] = "ambiguous_concurrent_prior_tick" if key in ambiguous else "no_prior_same_map_eligible_event"
            continue
        require(prior["event_tick"] < float(row["event_tick"]), f"non_strict_history:{dataset}")
        state.at[idx, "history_source_physical_event_id"] = prior["physical_event_id"]
        state.at[idx, "history_source_event_tick"] = prior["event_tick"]
        state.at[idx, "H_t"] = prior["signed_e"]
        state.at[idx, "history_crosses_round"] = prior["round_id"] != str(row["round_id"])
        state.at[idx, "history_missing_code"] = None
    state["analysis_eligible"] = state["primary_eligible"] & state["H_t"].notna()
    state["time_since_history_tick"] = state["event_tick"] - state["history_source_event_tick"]
    valid = state.loc[state["analysis_eligible"]]
    require(valid["history_source_event_tick"].lt(valid["event_tick"]).all(), f"history_not_strictly_earlier:{dataset}")
    audit = {
        "primary_eligible_rows": int(state["primary_eligible"].sum()), "analysis_eligible_rows": int(state["analysis_eligible"].sum()),
        "history_missing_counts": {str(k): int(v) for k, v in state.loc[~state["analysis_eligible"], "history_missing_code"].fillna("none").value_counts().items()},
        "same_round_history_rows": int((valid["history_crosses_round"] == False).sum()), "cross_round_history_rows": int((valid["history_crosses_round"] == True).sum()),
        "ambiguous_concurrent_prior_tick_rows": int(state["history_missing_code"].eq("ambiguous_concurrent_prior_tick").sum()), "cross_map_history_rows": 0, "same_tick_history_rows": 0,
    }
    return state, audit


def fit_two_way(frame: pd.DataFrame) -> dict[str, Any]:
    n = int(len(frame))
    if n <= 2 or float(frame["H_t"].max() - frame["H_t"].min()) == 0:
        return {"n": n, "alpha": None, "beta": None, "g3_pass": False, "reason": "insufficient_rows_or_predictor_variation"}
    x = np.column_stack([np.ones(n), frame["H_t"].to_numpy(dtype=float)])
    y = frame["e_t"].to_numpy(dtype=float)
    try:
        inverse = np.linalg.inv(x.T @ x)
    except np.linalg.LinAlgError:
        return {"n": n, "alpha": None, "beta": None, "g3_pass": False, "reason": "singular_design"}
    coefficient = inverse @ (x.T @ y)
    residual = y - x @ coefficient
    leverage = np.einsum("ij,jk,ik->i", x, inverse, x)

    def meat(values: pd.Series) -> tuple[np.ndarray, int]:
        codes, groups = pd.factorize(values, sort=False)
        scores = x * residual[:, None]
        summed = np.zeros((len(groups), 2))
        np.add.at(summed, codes, scores)
        return summed.T @ summed, len(groups)

    match_meat, match_clusters = meat(frame["match_id"].astype(str))
    player_meat, player_clusters = meat(frame["focal_player_id"].astype(str))
    intersection = frame["match_id"].astype(str) + "\x1f" + frame["focal_player_id"].astype(str)
    intersection_meat, intersection_clusters = meat(intersection)
    if min(match_clusters, player_clusters, intersection_clusters) <= 1:
        return {"n": n, "alpha": float(coefficient[0]), "beta": float(coefficient[1]), "g3_pass": False, "reason": "insufficient_cluster_groups", "match_clusters": match_clusters, "player_clusters": player_clusters, "max_leverage": float(leverage.max())}
    correction = lambda groups: (groups / (groups - 1)) * ((n - 1) / (n - 2))
    covariance = inverse @ (correction(match_clusters) * match_meat + correction(player_clusters) * player_meat - correction(intersection_clusters) * intersection_meat) @ inverse
    covariance = (covariance + covariance.T) / 2
    beta_variance = float(covariance[1, 1])
    g3_pass = bool(match_clusters >= 30 and player_clusters >= 30 and float(leverage.max()) <= 0.20 and np.isfinite(covariance).all() and beta_variance > 0)
    result: dict[str, Any] = {"n": n, "alpha": float(coefficient[0]), "beta": float(coefficient[1]), "match_clusters": int(match_clusters), "player_clusters": int(player_clusters), "intersection_clusters": int(intersection_clusters), "max_leverage": float(leverage.max()), "covariance": covariance.tolist(), "g3_pass": g3_pass}
    if g3_pass:
        se = math.sqrt(beta_variance)
        z = float(coefficient[1] / se)
        result.update({"beta_se": se, "z_value": z, "p_value": math.erfc(abs(z) / math.sqrt(2)), "ci95_low": float(coefficient[1] - 1.959963984540054 * se), "ci95_high": float(coefficient[1] + 1.959963984540054 * se)})
    else:
        result.update({"beta_se": None, "z_value": None, "p_value": None, "ci95_low": None, "ci95_high": None})
    return result


def holm(models: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    adjusted: dict[str, float | None] = {dataset: None for dataset in models}
    ordered = sorted((model["p_value"], dataset) for dataset, model in models.items() if model.get("p_value") is not None)
    running = 0.0
    for rank, (p_value, dataset) in enumerate(ordered):
        running = max(running, min(1.0, (len(models) - rank) * float(p_value)))
        adjusted[dataset] = running
    return adjusted


def calibration(frame: pd.DataFrame) -> dict[str, Any]:
    x = frame.loc[frame["primary_eligible"]].copy()
    p, y = x["p_hat"].to_numpy(dtype=float), x["Y_t"].to_numpy(dtype=float)
    z = np.log(p / (1 - p))
    design, coefficient = np.column_stack([np.ones(len(x)), z]), np.array([0.0, 1.0])
    for _ in range(100):
        mu = 1 / (1 + np.exp(-np.clip(design @ coefficient, -30, 30)))
        weight = np.maximum(mu * (1 - mu), 1e-10)
        step = np.linalg.solve(design.T @ (weight[:, None] * design), design.T @ (y - mu))
        coefficient += step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    else:
        raise RuntimeError("logistic_calibration_nonconvergence")
    binned = x.assign(_bin=pd.qcut(x["p_hat"], q=10, duplicates="drop")).groupby("_bin", observed=True).agg(n=("p_hat", "size"), mean_p_hat=("p_hat", "mean"), mean_r_t=("r_t", "mean")).reset_index()
    return {"n": int(len(x)), "mean_r_t": float(x["r_t"].mean()), "logistic_intercept": float(coefficient[0]), "logistic_slope": float(coefficient[1]), "bins": [{"bin": str(row["_bin"]), "n": int(row["n"]), "mean_p_hat": float(row["mean_p_hat"]), "mean_r_t": float(row["mean_r_t"])} for _, row in binned.iterrows()]}


def opportunity(frame: pd.DataFrame) -> dict[str, Any]:
    x = frame.loc[frame["primary_eligible"]].copy()
    focal = x[["physical_event_id", "match_id", "map_id", "round_id", "event_tick", "focal_player_id", "e_t"]].rename(columns={"focal_player_id": "player_id", "e_t": "signed_e"})
    opponent = x[["physical_event_id", "match_id", "map_id", "round_id", "event_tick", "opponent_player_id", "e_t"]].rename(columns={"opponent_player_id": "player_id", "e_t": "signed_e"})
    opponent["signed_e"] = -opponent["signed_e"]
    player_events = pd.concat([focal, opponent], ignore_index=True)
    player_events["player_id"] = player_events["player_id"].astype(str)
    player_events["same_round_reengage"] = False
    player_events["cross_round_reengage"] = False
    for _, group in player_events.groupby(["match_id", "map_id", "player_id"], sort=False, dropna=False):
        ordered = group.sort_values(["event_tick", "physical_event_id"], kind="stable")
        rows = list(ordered.itertuples())
        for position, row in enumerate(rows):
            later = [other for other in rows[position + 1:] if float(other.event_tick) > float(row.event_tick)]
            player_events.at[row.Index, "same_round_reengage"] = any(str(other.round_id) == str(row.round_id) for other in later)
            player_events.at[row.Index, "cross_round_reengage"] = any(str(other.round_id) != str(row.round_id) for other in later)
    def corr(column: str) -> float | None:
        a, b = player_events["signed_e"].to_numpy(dtype=float), player_events[column].astype(float).to_numpy()
        return None if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])
    return {"player_event_rows": int(len(player_events)), "same_round_reengagement_rate": float(player_events["same_round_reengage"].mean()), "cross_round_reengagement_rate": float(player_events["cross_round_reengage"].mean()), "signed_residual_same_round_reengagement_correlation": corr("same_round_reengage"), "signed_residual_cross_round_reengagement_correlation": corr("cross_round_reengage")}


def valid_history_checkpoint(path: Path, state_path: Path, lock_hash: str) -> bool:
    if not path.is_file() or not state_path.is_file():
        return False
    saved = json.loads(path.read_text(encoding="utf-8"))
    return saved.get("input_lock_sha256") == lock_hash and saved.get("state_sha256") == digest(state_path)


def write_or_verify_json(path: Path, value: dict[str, Any], lock_hash: str) -> None:
    if not path.exists():
        write_json_new(path, value)
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    require(saved.get("input_lock_sha256") == lock_hash, f"resume_checkpoint_lock_mismatch:{path.name}")


def main() -> None:
    try:
        lock_hash = ensure_input_lock()
        frames = {dataset: pd.read_parquet(path) for dataset, path in INPUTS.items()}
        g1 = {dataset: validate_asset(dataset, frame) for dataset, frame in frames.items()}
        states, history_audits = {}, {}
        for dataset, frame in frames.items():
            states[dataset], history_audits[dataset] = build_history(dataset, frame)
        # All G0--G2 checks precede any event-level output.
        for dataset, state in states.items():
            state_path = OUT / "state" / dataset / "history_state.parquet"
            checkpoint = OUT / "state" / dataset / f"checkpoint_02_history_{dataset}.json"
            if not valid_history_checkpoint(checkpoint, state_path, lock_hash):
                require(not state_path.exists() and not checkpoint.exists(), f"incomplete_history_checkpoint:{dataset}")
                write_parquet_new(state_path, state)
                write_json_new(checkpoint, {"stage": "first_order_residual_persistence_d1_v4_history", "dataset": dataset, "completed_at_utc": now(), "input_lock_sha256": lock_hash, "state_sha256": digest(state_path), "history_audit": history_audits[dataset]})

        primary, sensitivity, gates = {}, {}, {}
        for dataset, state in states.items():
            model_frame = state.loc[state["analysis_eligible"]].copy()
            primary[dataset] = fit_two_way(model_frame)
            same_round = model_frame.loc[model_frame["history_crosses_round"] == False].copy()
            sensitivity[dataset] = fit_two_way(same_round)
            try:
                g4 = {"status": "indeterminate", "reason": "no_independent_locked_calibration_sample", "diagnostics": calibration(state)}
            except Exception as error:
                g4 = {"status": "fail", "reason": f"calibration_diagnostic_failed:{type(error).__name__}:{error}"}
            gates[dataset] = {
                "G0": {"status": "pass", "reason": "d1_only_input_lock_hashes_verified"},
                "G1": {"status": "pass", "reason": "structural_time_audit_passed", "metrics": g1[dataset]},
                "G2": {"status": "pass", "reason": "history_state_reconstructed_without_same_tick_or_cross_map_history", "metrics": history_audits[dataset]},
                "G3": {"status": "pass" if primary[dataset]["g3_pass"] else "fail", "reason": "two_way_crv1_thresholds", "metrics": primary[dataset]},
                "G4": g4,
                "G5": {"status": "indeterminate", "reason": "no_complete_event_opportunity_generation_model", "diagnostics": opportunity(state)},
            }
        for dataset, value in holm(primary).items():
            primary[dataset]["holm_adjusted_p_value_three_partition_family"] = value

        write_or_verify_json(OUT / "checkpoint_03_gate_audit.json", {"stage": "first_order_residual_persistence_d1_v4_gate_audit", "created_at_utc": now(), "input_lock_sha256": lock_hash, "gates": gates}, lock_hash)
        write_or_verify_json(OUT / "checkpoint_04_models.json", {"stage": "first_order_residual_persistence_d1_v4_models", "created_at_utc": now(), "input_lock_sha256": lock_hash, "primary_same_map_cross_round": primary, "sensitivity_same_map_same_round": sensitivity, "inference": "two_way_crv1_match_and_focal_player_normal_approximation; holm_three_partition_family"}, lock_hash)
        hashes = OUT / "file_sha256.json"
        if not hashes.exists():
            write_json_new(hashes, {str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"}})
        manifest = OUT / "manifest.json"
        if not manifest.exists():
            write_json_new(manifest, {"stage": "first_order_residual_persistence_d1_v4", "status": "passed", "created_at_utc": now(), "classification": "exploratory_nonconfirmatory_d1_only", "input_lock_sha256": lock_hash, "gate_statuses": {dataset: {name: value["status"] for name, value in gate_set.items()} for dataset, gate_set in gates.items()}, "interpretation": "D1-only, conditional-on-observed-opportunity and model-dependent first-order residual association only; not a hot-hand, state, causal, calibration, or predictive-gain conclusion.", "file_sha256_sha256": digest(hashes)})
        print(json.dumps({"status": "passed", "manifest": str(manifest), "gate_statuses": {dataset: {name: value["status"] for name, value in gate_set.items()} for dataset, gate_set in gates.items()}}, ensure_ascii=True))
    except Exception as error:
        write_failure(error)
        raise


if __name__ == "__main__":
    main()
