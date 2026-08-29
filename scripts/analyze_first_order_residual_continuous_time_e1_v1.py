"""Execute the frozen E1-v2 continuous-time residual association replication."""

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
ASSET = ROOT / "results" / "conditional_residual_asset_e1_v1"
SOURCE = ASSET / DATASET / "residual_asset.parquet"
OUT = ROOT / "results" / "first_order_residual_continuous_time_e1_v1"
DRAFT = ROOT / "docs" / "修订与补丁" / "28_E1v2连续时间调节一阶历史残差关联分析协议_草案_v1_2026-08-26.md"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "29_E1v2连续时间调节一阶历史残差关联执行协议_v1_2026-08-26.md"
INDEPENDENCE_RECORDS = (
    ROOT / "results" / "d1a_c_selection_training_v1" / "manifest.json",
    ROOT / "docs" / "修订与补丁" / "24_D1连续时间调节一阶历史残差关联执行协议_v1_D1_only_2026-08-26.md",
    ROOT / "scripts" / "analyze_first_order_residual_continuous_time_d1_v1.py",
    ROOT / "docs" / "报告" / "补丁与修订" / "12_D1连续时间调节一阶历史残差关联执行报告_v1_2026-08-26.md",
    ROOT / "results" / "external_generalization_stop_v1" / "D1-C" / "manifest.json",
)
ASSET_METADATA = (
    ASSET / "manifest.json", ASSET / "file_sha256.json", ASSET / "input_lock.json",
    ASSET / "label_access_receipt.json", ASSET / "field_dictionary.json",
    ASSET / DATASET / "structural_temporal_audit.json", ASSET / DATASET / "input_candidate_reference.json",
)
REQUIRED = {
    "dataset", "physical_event_id", "match_id", "map_id", "round_id", "event_tick", "anchor_tick",
    "focal_player_id", "opponent_player_id", "Y_t", "p_hat", "r_t", "e_t", "p_support_0p01",
    "residual_valid", "primary_analysis_eligible",
}
ANALYSIS_COLUMNS = [
    "e_t", "H_t", "match_id", "focal_player_id", "event_tick", "history_source_event_tick",
    "time_since_history_tick",
]


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_parquet_new(path: Path, frame: pd.DataFrame) -> None:
    require(not path.exists(), f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def source_audit() -> dict[str, Any]:
    expected = read_json(ASSET / "file_sha256.json")
    mismatches = [
        name for name, expected_hash in expected.items()
        if not (ASSET / name).is_file() or digest(ASSET / name) != expected_hash
    ]
    manifest = read_json(ASSET / "manifest.json")
    receipt = read_json(ASSET / "label_access_receipt.json")
    return {
        "checked_files": len(expected),
        "mismatches": mismatches,
        "manifest_file_sha256_matches": manifest.get("file_sha256_sha256") == digest(ASSET / "file_sha256.json"),
        "asset_status": manifest.get("status"),
        "asset_dataset": receipt.get("dataset"),
        "asset_probability_role": receipt.get("probability_role"),
    }


def input_hashes() -> dict[str, str]:
    paths = (DRAFT, PROTOCOL, Path(__file__).resolve(), SOURCE, *ASSET_METADATA, *INDEPENDENCE_RECORDS)
    return {str(path): digest(path) for path in paths}


def ensure_input_lock() -> tuple[dict[str, Any], str]:
    audit = source_audit()
    require(not audit["mismatches"], "asset_file_sha256_mismatch")
    require(audit["manifest_file_sha256_matches"], "asset_manifest_file_sha256_mismatch")
    require(audit["asset_status"] == "passed", "asset_not_passed")
    require(audit["asset_dataset"] == DATASET, "asset_dataset_receipt_mismatch")
    require(audit["asset_probability_role"] == "fixed_model_external_score_nonconfirmatory", "asset_probability_role_mismatch")
    inputs = input_hashes()
    path = OUT / "checkpoint_01_input_lock.json"
    if path.exists():
        saved = read_json(path)
        require(saved.get("stage") == "continuous_time_e1_v1_input_lock", "input_lock_stage_mismatch")
        require(saved.get("inputs") == inputs, "resume_input_lock_hash_mismatch")
        require(saved.get("source_hash_audit") == audit, "resume_source_audit_mismatch")
        return saved, digest(path)
    value = {
        "stage": "continuous_time_e1_v1_input_lock", "created_at_utc": now(),
        "classification": "exploratory_nonconfirmatory_independent_replication_candidate",
        "inputs": inputs, "source_hash_audit": audit,
        "outcome_access": "existing_authorized_asset_only_no_raw_or_rescoring_access",
        "independence_attestation": "D1 training and continuous-time records are hash-locked provenance only; E1-v2 did not participate in their recorded inputs or design.",
    }
    write_json_new(path, value)
    return value, digest(path)


def validate_asset(frame: pd.DataFrame) -> dict[str, Any]:
    missing = sorted(REQUIRED - set(frame.columns))
    require(not missing, f"missing_required_columns:{','.join(missing)}")
    require(frame["dataset"].astype(str).eq(DATASET).all(), "dataset_column_mismatch")
    require(frame["physical_event_id"].notna().all() and not frame["physical_event_id"].duplicated().any(), "physical_event_id_not_unique")
    require(frame["focal_player_id"].notna().all() and frame["opponent_player_id"].notna().all(), "missing_player_id")
    require(frame["focal_player_id"].astype(str).ne(frame["opponent_player_id"].astype(str)).all(), "self_duel")
    require(frame["anchor_tick"].lt(frame["event_tick"]).all(), "anchor_time_gate_failed")
    finite = np.isfinite(frame[["event_tick", "anchor_tick", "Y_t", "p_hat", "r_t", "e_t"]].to_numpy(dtype=float))
    require(finite.all(), "nonfinite_required_values")
    require(frame["Y_t"].isin([0, 1]).all(), "nonbinary_outcome")
    require(frame["p_hat"].between(0, 1, inclusive="neither").all(), "invalid_probability")
    eligible = frame["primary_analysis_eligible"].astype(bool) & frame["residual_valid"].astype(bool)
    require(frame.loc[eligible, "p_hat"].between(0.01, 0.99, inclusive="both").all(), "eligible_support_gate_failed")
    return {
        "rows": int(len(frame)), "primary_eligible_rows": int(eligible.sum()),
        "nonfinite_required_values": int((~finite).sum()), "physical_event_id_unique": True,
        "source_canonical_direction_inherited": True, "anchor_before_event": True,
    }


def build_analysis_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
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
        blocked = False
        for _, same_tick in group.groupby("event_tick", sort=False, dropna=False):
            for row in same_tick.itertuples(index=False):
                key = (str(row.physical_event_id), str(row.player_id))
                previous[key] = last
                if last is None and blocked:
                    ambiguous.add(key)
            if len(same_tick) == 1:
                row = same_tick.iloc[0]
                last = {"physical_event_id": str(row["physical_event_id"]), "event_tick": float(row["event_tick"]), "signed_e": float(row["signed_e"])}
                blocked = False
            else:
                last = None
                blocked = True
    state["history_source_event_tick"] = np.nan
    state["H_t"] = np.nan
    state["history_missing_code"] = "current_not_primary_eligible"
    for index, row in state.loc[state["primary_eligible"]].iterrows():
        prior = previous.get((str(row["physical_event_id"]), str(row["focal_player_id"])))
        if prior is None:
            key = (str(row["physical_event_id"]), str(row["focal_player_id"]))
            state.at[index, "history_missing_code"] = "ambiguous_concurrent_prior_tick" if key in ambiguous else "no_prior_same_map_eligible_event"
            continue
        require(prior["event_tick"] < float(row["event_tick"]), "non_strict_history")
        state.at[index, "history_source_event_tick"] = prior["event_tick"]
        state.at[index, "H_t"] = prior["signed_e"]
        state.at[index, "history_missing_code"] = None
    state["analysis_eligible"] = state["primary_eligible"] & state["H_t"].notna()
    state["time_since_history_tick"] = state["event_tick"] - state["history_source_event_tick"]
    matrix = state.loc[state["analysis_eligible"], ANALYSIS_COLUMNS].copy()
    require(matrix["history_source_event_tick"].lt(matrix["event_tick"]).all(), "history_not_strictly_earlier")
    require((matrix["time_since_history_tick"] >= 1).all(), "invalid_history_interval")
    audit = {
        "primary_eligible_rows": int(state["primary_eligible"].sum()), "analysis_eligible_rows": int(len(matrix)),
        "history_missing_counts": {str(k): int(v) for k, v in state.loc[~state["analysis_eligible"], "history_missing_code"].fillna("none").value_counts().items()},
        "ambiguous_concurrent_prior_tick_rows": int(state["history_missing_code"].eq("ambiguous_concurrent_prior_tick").sum()),
        "cross_map_history_rows": 0, "same_tick_history_rows": 0,
    }
    return matrix, audit


def time_summary(frame: pd.DataFrame) -> dict[str, Any]:
    times = frame["time_since_history_tick"].to_numpy(dtype=float)
    return {
        "rows": int(len(frame)), "match_clusters": int(frame["match_id"].astype(str).nunique()),
        "player_clusters": int(frame["focal_player_id"].astype(str).nunique()),
        "T_p10_p50_p90": [float(value) for value in np.quantile(times, [0.10, 0.50, 0.90])],
        "T_min": float(times.min()), "T_max": float(times.max()), "mu": float(np.log2(1 + times).mean()),
    }


def load_or_build(lock_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrix_path = OUT / "state" / DATASET / "analysis_matrix.parquet"
    checkpoint = OUT / "checkpoint_02_preflight_E1-v2.json"
    source_hash = digest(SOURCE)
    if checkpoint.exists() or matrix_path.exists():
        require(checkpoint.exists() and matrix_path.exists(), "incomplete_preflight_checkpoint")
        saved = read_json(checkpoint)
        require(saved.get("input_lock_sha256") == lock_hash, "preflight_input_lock_mismatch")
        require(saved.get("source_residual_asset_sha256") == source_hash, "preflight_source_hash_mismatch")
        require(saved.get("analysis_matrix_sha256") == digest(matrix_path), "preflight_matrix_hash_mismatch")
        matrix = pd.read_parquet(matrix_path)
        require(saved["metrics"]["time_summary"] == time_summary(matrix), "resume_time_summary_mismatch")
        return matrix, saved
    frame = pd.read_parquet(SOURCE)
    field_audit = validate_asset(frame)
    matrix, history_audit = build_analysis_matrix(frame)
    summary = time_summary(matrix)
    finite = int((~np.isfinite(matrix[["e_t", "H_t", "event_tick", "history_source_event_tick", "time_since_history_tick"]].to_numpy(dtype=float))).sum())
    write_parquet_new(matrix_path, matrix)
    value = {
        "stage": "continuous_time_e1_v1_preflight", "dataset": DATASET, "created_at_utc": now(),
        "input_lock_sha256": lock_hash, "source_residual_asset_sha256": source_hash,
        "analysis_matrix_sha256": digest(matrix_path),
        "metrics": {"field_audit": field_audit, "history_audit": history_audit, "time_summary": summary, "nonfinite_analysis_values": finite},
        "G0": {"status": "pass", "reason": "asset_hash_chain_and_authorized_boundary_verified"},
        "G1": {"status": "pass" if finite == 0 else "fail", "reason": "fields_direction_history_and_complete_time_summary"},
        "G2": {"status": "pass" if summary["match_clusters"] >= 30 and summary["player_clusters"] >= 30 else "fail", "reason": "match_and_player_cluster_support"},
    }
    write_json_new(checkpoint, value)
    return matrix, value


def contrast(coefficient: np.ndarray, covariance: np.ndarray, vector: np.ndarray) -> dict[str, float]:
    variance = float(vector @ covariance @ vector)
    require(math.isfinite(variance) and variance > 0, "invalid_contrast_variance")
    se = math.sqrt(variance)
    estimate = float(vector @ coefficient)
    return {"estimate": estimate, "standard_error": se, "ci95_low": estimate - 1.95996 * se, "ci95_high": estimate + 1.95996 * se}


def fit(frame: pd.DataFrame, summary: dict[str, Any]) -> dict[str, Any]:
    n = len(frame)
    z = np.log2(1 + frame["time_since_history_tick"].to_numpy(dtype=float)) - summary["mu"]
    design = np.column_stack([np.ones(n), z, frame["H_t"].to_numpy(dtype=float), frame["H_t"].to_numpy(dtype=float) * z])
    k = design.shape[1]
    require(n > k and np.linalg.matrix_rank(design) == k, "rank_deficient_continuous_time_design")
    inverse = np.linalg.inv(design.T @ design)
    coefficient = inverse @ design.T @ frame["e_t"].to_numpy(dtype=float)
    residual = frame["e_t"].to_numpy(dtype=float) - design @ coefficient
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
    def meat(groups: pd.Series) -> tuple[np.ndarray, int]:
        _, codes = np.unique(groups.astype(str).to_numpy(), return_inverse=True)
        summed = np.zeros((int(codes.max()) + 1, k))
        np.add.at(summed, codes, design * residual[:, None])
        return summed.T @ summed, len(summed)
    match_meat, match_clusters = meat(frame["match_id"])
    player_meat, player_clusters = meat(frame["focal_player_id"])
    intersection_meat, intersection_clusters = meat(frame["match_id"].astype(str) + "\x1f" + frame["focal_player_id"].astype(str))
    correction = lambda groups: (groups / (groups - 1)) * ((n - 1) / (n - k))
    covariance = inverse @ (correction(match_clusters) * match_meat + correction(player_clusters) * player_meat - correction(intersection_clusters) * intersection_meat) @ inverse
    covariance = (covariance + covariance.T) / 2
    g3 = bool(np.isfinite(covariance).all() and covariance[2, 2] > 0 and match_clusters >= 30 and player_clusters >= 30 and leverage.max() <= 0.20)
    result: dict[str, Any] = {"n": int(n), "coefficient_names": ["alpha", "a_time_main_effect", "beta_average_lag1_association", "kappa_time_modification"], "coefficients": {name: float(value) for name, value in zip(["alpha", "a_time_main_effect", "beta_average_lag1_association", "kappa_time_modification"], coefficient)}, "match_clusters": match_clusters, "player_clusters": player_clusters, "intersection_clusters": intersection_clusters, "max_leverage": float(leverage.max()), "covariance": covariance.tolist(), "g3_pass": g3}
    if not g3:
        return result
    beta = contrast(coefficient, covariance, np.array([0.0, 0.0, 1.0, 0.0]))
    kappa = contrast(coefficient, covariance, np.array([0.0, 0.0, 0.0, 1.0]))
    normal = NormalDist()
    critical = normal.inv_cdf(1 - 0.05 / 2)
    beta_se = beta["standard_error"]
    slopes = {label: {"T": tick, **contrast(coefficient, covariance, np.array([0.0, 0.0, 1.0, math.log2(1 + tick) - summary["mu"]]))} for label, tick in zip(("P10", "P50", "P90"), summary["T_p10_p50_p90"])}
    pair_cov = covariance[np.ix_([2, 3], [2, 3])]
    determinant = float(np.linalg.det(pair_cov))
    result.update({
        "beta": {**beta, "z_value": beta["estimate"] / beta_se, "p_value": float(math.erfc(abs(beta["estimate"] / beta_se) / math.sqrt(2)))},
        "kappa": kappa, "time_point_slopes": slopes,
        "joint_wald_beta_kappa": {"eligible": determinant > 0, "statistic": float(coefficient[[2, 3]] @ np.linalg.inv(pair_cov) @ coefficient[[2, 3]]) if determinant > 0 else None, "df": 2 if determinant > 0 else None},
        "ci_half_width_95": 1.95996 * beta_se, "familywise_critical_z_singleton": critical,
        "mde_80_fwer": (critical + normal.inv_cdf(0.80)) * beta_se, "mde_90_fwer": (critical + normal.inv_cdf(0.90)) * beta_se,
        "scenario_power_singleton_family": {f"abs_beta_{effect:.2f}": float(1 - normal.cdf(critical - effect / beta_se) + normal.cdf(-critical - effect / beta_se)) for effect in (0.05, 0.10, 0.20)},
    })
    return result


def smoke(lock_hash: str, matrix: pd.DataFrame, preflight: dict[str, Any]) -> dict[str, Any]:
    path = OUT / "checkpoint_03_smoke_E1-v2.json"
    matrix_hash = digest(OUT / "state" / DATASET / "analysis_matrix.parquet")
    if path.exists():
        saved = read_json(path)
        require(saved.get("input_lock_sha256") == lock_hash and saved.get("analysis_matrix_sha256") == matrix_hash, "smoke_checkpoint_mismatch")
        return saved
    matches = sorted(matrix["match_id"].astype(str).unique())[:30]
    subset = matrix.loc[matrix["match_id"].astype(str).isin(matches)].copy()
    result = fit(subset, time_summary(subset))
    value = {"stage": "continuous_time_e1_v1_smoke", "dataset": DATASET, "created_at_utc": now(), "input_lock_sha256": lock_hash, "analysis_matrix_sha256": matrix_hash, "subset_rule": "30_lexicographically_smallest_match_id_clusters", "subset_rows": int(len(subset)), "subset_match_clusters": int(subset["match_id"].astype(str).nunique()), "subset_player_clusters": int(subset["focal_player_id"].astype(str).nunique()), "G0": preflight["G0"], "G1": preflight["G1"], "G2": {"status": "pass" if result["match_clusters"] >= 30 and result["player_clusters"] >= 30 else "fail", "reason": "smoke_cluster_support"}, "G3": {"status": "pass" if result["g3_pass"] else "fail", "reason": "smoke_model_and_precision_gate", "metrics": {key: result[key] for key in ("n", "match_clusters", "player_clusters", "intersection_clusters", "max_leverage", "g3_pass")}}}
    write_json_new(path, value)
    return value


def output_hashes() -> dict[str, str]:
    return {str(path.relative_to(OUT)): digest(path) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"file_sha256.json", "manifest.json"} and not any(part.startswith("failed_") for part in path.relative_to(OUT).parts)}


def run(phase: str) -> None:
    _, lock_hash = ensure_input_lock()
    matrix, preflight = load_or_build(lock_hash)
    require(all(preflight[name]["status"] == "pass" for name in ("G0", "G1", "G2")), "preflight_gate_failed")
    smoke_value = smoke(lock_hash, matrix, preflight)
    require(all(smoke_value[name]["status"] == "pass" for name in ("G0", "G1", "G2", "G3")), "smoke_gate_failed")
    if phase == "smoke":
        print(json.dumps({"status": "smoke_passed", "output_root": str(OUT)}, ensure_ascii=True))
        return
    model_path = OUT / "checkpoint_04_model_E1-v2.json"
    matrix_hash = digest(OUT / "state" / DATASET / "analysis_matrix.parquet")
    if model_path.exists():
        model_checkpoint = read_json(model_path)
        require(model_checkpoint.get("input_lock_sha256") == lock_hash and model_checkpoint.get("analysis_matrix_sha256") == matrix_hash, "model_checkpoint_mismatch")
        model = model_checkpoint["model"]
    else:
        model = fit(matrix, preflight["metrics"]["time_summary"])
        write_json_new(model_path, {"stage": "continuous_time_e1_v1_model", "dataset": DATASET, "created_at_utc": now(), "input_lock_sha256": lock_hash, "analysis_matrix_sha256": matrix_hash, "model": model})
    require(model["g3_pass"], "model_gate_failed")
    gates = {"G0": {**preflight["G0"], "metrics": {"source_hash_audit": source_audit()}}, "G1": {**preflight["G1"], "metrics": preflight["metrics"]}, "G2": {**preflight["G2"], "metrics": {"match_clusters": model["match_clusters"], "player_clusters": model["player_clusters"]}}, "G3": {"status": "pass", "reason": "unweighted_two_way_crv1_continuous_time_and_precision_thresholds", "metrics": model}, "G4": {"status": "pending", "reason": "no_independent_locked_calibration_sample"}, "G5": {"status": "pending", "reason": "no_complete_opportunity_model", "residual_centering": "none"}, "overall": {"model_release": "pass", "interpretation_release": "observed_opportunity_association_only", "forbidden_interpretations": ["d1_confirmation", "state_effect", "causal_effect", "calibration_proof", "opportunity_process_claim"]}}
    for path, value in ((OUT / "checkpoint_05_gate_audit.json", {"stage": "continuous_time_e1_v1_gate_audit", "created_at_utc": now(), "input_lock_sha256": lock_hash, "gates": gates}), (OUT / "checkpoint_06_models.json", {"stage": "continuous_time_e1_v1_models", "created_at_utc": now(), "input_lock_sha256": lock_hash, "models": {DATASET: {**model, "holm_adjusted_beta_p_value_singleton_family": model["beta"]["p_value"]}}, "inference": "unweighted_two_way_crv1_match_and_focal_player_normal_approximation; singleton_prelocked_beta_family"})):
        if path.exists():
            require(read_json(path).get("input_lock_sha256") == lock_hash, f"resume_checkpoint_lock_mismatch:{path.name}")
        else:
            write_json_new(path, value)
    hashes_path = OUT / "file_sha256.json"
    hashes = output_hashes()
    if hashes_path.exists():
        require(read_json(hashes_path) == hashes, "output_file_sha256_mismatch")
    else:
        write_json_new(hashes_path, hashes)
    manifest_path = OUT / "manifest.json"
    manifest = {"stage": "continuous_time_e1_v1", "status": "passed", "created_at_utc": now(), "classification": "exploratory_nonconfirmatory_independent_replication_candidate", "input_lock_sha256": lock_hash, "gate_statuses": {name: value["status"] for name, value in gates.items() if name.startswith("G")}, "interpretation": "E1-v2 independent, conditional-on-observed-opportunity continuous-time residual association only; not confirmation, state, causal, calibration, or predictive-gain evidence.", "file_sha256_sha256": digest(hashes_path)}
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
        write_json_new(failed / "failure_manifest.json", {"stage": "continuous_time_e1_v1", "status": "failed", "created_at_utc": now(), "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
