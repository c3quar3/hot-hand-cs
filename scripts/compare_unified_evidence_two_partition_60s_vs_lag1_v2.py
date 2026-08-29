"""Compare frozen B1, M1, and M3 residual loss on the final E-A/E-B partitions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import compare_unified_evidence_two_partition_60s_vs_lag1_v1 as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "unified_evidence_two_partition_loss_comparison_v2"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "60_统一证据池两分区含时间基线残差损失比较执行锁_v2_2026-08-27.md"
B1_FIT = ROOT / "results" / "history_residual_baseline_ladder_oof_d1_v1" / "state" / "F1" / "training_fit.json"
B1_ROOT = ROOT / "results" / "history_residual_baseline_ladder_oof_d1_v1"
PAIRS = (("B1_vs_M1", "B1_time_only", "M1_lag1"), ("B1_vs_M3", "B1_time_only", "M3_decay_sum"), ("M1_vs_M3", "M1_lag1", "M3_decay_sum"))


def now() -> str:
    return datetime.now(UTC).isoformat()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, value: dict) -> None:
    require(not path.exists(), f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def write_parquet_new(path: Path, frame: pd.DataFrame) -> None:
    require(not path.exists(), f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def lock_inputs() -> tuple[dict, str]:
    path = OUT / "input_lock.json"
    paths = core.input_paths() + [PROTOCOL, Path(__file__).resolve(), B1_FIT, B1_ROOT / "manifest.json", B1_ROOT / "file_sha256.json", ROOT / "scripts" / "compare_unified_evidence_two_partition_60s_vs_lag1_v1.py"]
    inputs = {str(item.relative_to(ROOT)): core.digest(item) for item in paths}
    baseline_audit = core.verify_hash_tree(B1_ROOT)
    require(not baseline_audit["mismatches"] and baseline_audit["manifest_file_sha256_matches"], "b1_fit_hash_chain_mismatch")
    payload = {
        "stage": "unified_evidence_two_partition_loss_comparison_v2_input_lock",
        "created_at_utc": now(),
        "classification": "user_authorized_final_source_fixed_b1_m1_m3_residual_loss_comparison_with_prior_e1_outcome_access_disclosed",
        "inputs": inputs,
        "b1_hash_audit": baseline_audit,
        "e1_asset_hash_audit": core.verify_hash_tree(core.ASSET_ROOT),
        "confirmation_hash_audit": core.verify_hash_tree(core.CONFIRMATION_ROOT),
        "seed": core.SEED,
        "bootstrap_draws": core.BOOTSTRAPS,
        "pairs": [name for name, _, _ in PAIRS],
    }
    if path.exists():
        saved = read_json(path)
        require(saved["inputs"] == inputs and saved["pairs"] == payload["pairs"], "input_hash_changed_after_lock")
        return saved, core.digest(path)
    write_json_new(path, payload)
    return payload, core.digest(path)


def add_b1(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    spec = read_json(B1_FIT)["B1"]
    z = np.log2(1.0 + output["d_tick"].to_numpy(float)) - float(spec["mu_train"])
    output["B1_time_only"] = float(spec["coefficients"][0]) + float(spec["coefficients"][1]) * z
    require(np.isfinite(output[["e_t", "B1_time_only", "M1_lag1", "M3_decay_sum"]].to_numpy(float)).all(), "nonfinite_prediction")
    return output


def smoke(lock_hash: str, frames: dict[str, pd.DataFrame], audits: dict[str, dict]) -> dict:
    path = OUT / "smoke.json"
    items = {}
    for partition in core.PARTITIONS:
        matches = sorted(frames[partition]["match_id"].astype(str).unique())[:30]
        view = frames[partition].loc[frames[partition]["match_id"].astype(str).isin(matches)]
        items[partition] = {"matches": len(matches), "shared_rows": len(view), "finite_predictions": bool(np.isfinite(view[["B1_time_only", "M1_lag1", "M3_decay_sum"]].to_numpy(float)).all()), "source_rows": audits[partition]["source_rows"], "no_loss_or_effect_estimates_written": True}
        require(items[partition]["matches"] == 30 and items[partition]["shared_rows"] > 0 and items[partition]["finite_predictions"], f"smoke_failed:{partition}")
    payload = {"stage": "unified_evidence_two_partition_loss_comparison_v2_smoke", "created_at_utc": now(), "input_lock_sha256": lock_hash, "status": "pass", "partitions": items}
    if path.exists():
        saved = read_json(path)
        require(saved["input_lock_sha256"] == lock_hash and saved["status"] == "pass", "smoke_checkpoint_mismatch")
        return saved
    write_json_new(path, payload)
    return payload


def pair_gain(values: pd.DataFrame, baseline: str, candidate: str) -> float:
    mean = values[[baseline, candidate]].mean()
    return float((mean[baseline] - mean[candidate]) / mean[baseline])


def compare_partition(frame: pd.DataFrame, partition_index: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scored = frame[["match_id", "e_t", "B1_time_only", "M1_lag1", "M3_decay_sum"]].copy()
    for model in ("B1_time_only", "M1_lag1", "M3_decay_sum"):
        scored[model] = (scored["e_t"] - scored[model]) ** 2
    losses = scored.groupby("match_id", as_index=False)[["B1_time_only", "M1_lag1", "M3_decay_sum"]].mean()
    losses["partition"] = core.PARTITIONS[partition_index]
    require(len(losses) == 100 and (losses[["B1_time_only", "M1_lag1", "M3_decay_sum"]] > 0).all().all(), f"invalid_match_losses:{core.PARTITIONS[partition_index]}")
    rng = np.random.default_rng(core.SEED + partition_index)
    draw_values = np.empty((core.BOOTSTRAPS, len(PAIRS)), dtype=float)
    for draw in range(core.BOOTSTRAPS):
        sample = losses.iloc[rng.integers(len(losses), size=len(losses))]
        for index, (_, baseline, candidate) in enumerate(PAIRS):
            draw_values[draw, index] = pair_gain(sample, baseline, candidate)
    rows, summary = [], {}
    for index, (name, baseline, candidate) in enumerate(PAIRS):
        point = pair_gain(losses, baseline, candidate)
        values = draw_values[:, index]
        low, high = np.quantile(values, [0.025, 0.975])
        sd = float(values.std(ddof=1))
        raw_p = math.erfc(abs(point / sd) / math.sqrt(2.0)) if sd > 0 else (0.0 if point != 0 else 1.0)
        summary[name] = {"baseline": baseline, "candidate": candidate, "relative_gain": point, "ci95_low": float(low), "ci95_high": float(high), "ci95_half_width": float((high - low) / 2.0), "bootstrap_sd": sd, "p_value_two_sided_raw": float(raw_p)}
        rows.extend({"partition": core.PARTITIONS[partition_index], "draw": draw, "comparison": name, "relative_gain": value} for draw, value in enumerate(values))
    return losses, pd.DataFrame(rows), summary


def holm(items: list[tuple[str, str, float]]) -> dict[tuple[str, str], float]:
    ordered = sorted(items, key=lambda item: item[2])
    adjusted, running, total = {}, 0.0, len(ordered)
    for index, (partition, name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[(partition, name)] = running
    return adjusted


def output_hashes() -> dict[str, str]:
    names = ("input_lock.json", "smoke.json", "row_audit.json", "match_losses.parquet", "bootstrap_gains.parquet", "comparison.json", "gates.json")
    return {name: core.digest(OUT / name) for name in names}


def run(phase: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _, lock_hash = lock_inputs()
    ledger = core.reconstruction_ledger()
    frames, audits = {}, {}
    for partition in core.PARTITIONS:
        frame, audit = core.partition_frame(partition, ledger)
        frames[partition], audits[partition] = add_b1(frame), audit
    smoke_payload = smoke(lock_hash, frames, audits)
    if phase == "smoke":
        print(json.dumps({"status": "passed", "phase": "smoke", "smoke": str(OUT / "smoke.json")}, ensure_ascii=True))
        return
    require(smoke_payload["input_lock_sha256"] == lock_hash, "smoke_lock_mismatch")
    if not (OUT / "row_audit.json").exists():
        write_json_new(OUT / "row_audit.json", {"stage": "unified_evidence_two_partition_loss_comparison_v2_rows", "created_at_utc": now(), "input_lock_sha256": lock_hash, "audits": audits})
    all_losses, all_draws, comparisons = [], [], {}
    for index, partition in enumerate(core.PARTITIONS):
        losses, draws, summary = compare_partition(frames[partition], index)
        all_losses.append(losses)
        all_draws.append(draws)
        comparisons[partition] = summary
    if not (OUT / "match_losses.parquet").exists():
        write_parquet_new(OUT / "match_losses.parquet", pd.concat(all_losses, ignore_index=True))
    if not (OUT / "bootstrap_gains.parquet").exists():
        write_parquet_new(OUT / "bootstrap_gains.parquet", pd.concat(all_draws, ignore_index=True))
    p_items = [(partition, name, comparisons[partition][name]["p_value_two_sided_raw"]) for partition in core.PARTITIONS for name, _, _ in PAIRS]
    adjusted = holm(p_items)
    for partition, name, _ in p_items:
        comparisons[partition][name]["p_value_holm"] = adjusted[(partition, name)]
        comparisons[partition][name]["positive_superiority"] = bool(comparisons[partition][name]["relative_gain"] > 0 and comparisons[partition][name]["ci95_low"] > 0 and adjusted[(partition, name)] < 0.05)
    if not (OUT / "comparison.json").exists():
        write_json_new(OUT / "comparison.json", {"stage": "unified_evidence_two_partition_loss_comparison_v2_comparison", "created_at_utc": now(), "input_lock_sha256": lock_hash, "inference": "2000_match_cluster_bootstrap_percentile_ci_normal_p_holm_six_pair_family", "comparisons": comparisons})
    precision = {f"{partition}:{name}": comparisons[partition][name]["ci95_half_width"] <= core.PRECISION_HALF_WIDTH for partition in core.PARTITIONS for name, _, _ in PAIRS}
    superiority = {f"{partition}:{name}": comparisons[partition][name]["positive_superiority"] for partition in core.PARTITIONS for name, _, _ in PAIRS}
    gates = {"T1_inputs_and_smoke": {"status": "pass"}, "T2_shared_rows_and_finite_prediction": {"status": "pass", "metrics": audits}, "T3_precision": {"status": "pass" if all(precision.values()) else "fail", "threshold_ci_half_width": core.PRECISION_HALF_WIDTH, "items": precision}, "T4_positive_superiority": {"status": "pass" if all(superiority.values()) and all(precision.values()) else "mixed_or_fail", "items": superiority}, "T5_final_source_record": {"status": "user_adjudicated_final_data_source_with_prior_e1_outcome_access_disclosed"}}
    if not (OUT / "gates.json").exists():
        write_json_new(OUT / "gates.json", {"stage": "unified_evidence_two_partition_loss_comparison_v2_gates", "created_at_utc": now(), "input_lock_sha256": lock_hash, "gates": gates})
    if not (OUT / "file_sha256.json").exists():
        write_json_new(OUT / "file_sha256.json", output_hashes())
    else:
        for relative, expected in read_json(OUT / "file_sha256.json").items():
            require((OUT / relative).is_file() and core.digest(OUT / relative) == expected, f"output_hash_mismatch:{relative}")
    status = "passed" if gates["T4_positive_superiority"]["status"] == "pass" else "completed_with_mixed_or_null_comparisons"
    if not (OUT / "manifest.json").exists():
        write_json_new(OUT / "manifest.json", {"stage": "unified_evidence_two_partition_loss_comparison_v2", "created_at_utc": now(), "status": status, "classification": "user_authorized_final_source_fixed_b1_m1_m3_residual_loss_comparison_with_prior_e1_outcome_access_disclosed", "input_lock_sha256": lock_hash, "smoke_sha256": core.digest(OUT / "smoke.json"), "file_sha256_sha256": core.digest(OUT / "file_sha256.json"), "gate_statuses": {name: item["status"] for name, item in gates.items()}, "interpretation": "Fixed B1/M1/M3 match-equal Pearson-residual squared-loss comparison only; not p_hat performance, association replacement, causal, latent-state, or opportunity-process evidence."})
    print(json.dumps({"status": status, "manifest": str(OUT / "manifest.json")}, ensure_ascii=True))


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
        write_json_new(failed / "failure_manifest.json", {"stage": "unified_evidence_two_partition_loss_comparison_v2", "status": "failed", "created_at_utc": now(), "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
