"""Compare frozen D1-A/F1 M1 and M3 residual loss on the final E-A/E-B partitions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "results" / "conditional_residual_asset_e1_v1" / "E1-v2" / "residual_asset.parquet"
ASSET_ROOT = ROOT / "results" / "conditional_residual_asset_e1_v1"
ASSIGNMENTS = ROOT / "data" / "unified_evidence_two_partitions_v3" / "match_assignments.parquet"
ALLOCATION_ROOT = ROOT / "data" / "unified_evidence_two_partitions_v3"
CONFIRMATION_ROOT = ROOT / "results" / "unified_evidence_two_partition_confirmation_v1"
FIT = ROOT / "results" / "history_residual_state_oof_d1_v2" / "state" / "F1" / "training_fit.json"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "59_统一证据池两分区60秒相对lag-1损失比较执行锁_v1_2026-08-27.md"
OUT = ROOT / "results" / "unified_evidence_two_partition_loss_comparison_v1"
PARTITIONS = ("E-A", "E-B")
SEED = 20260827
BOOTSTRAPS = 2000
PRECISION_HALF_WIDTH = 0.002


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


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


def verify_hash_tree(root: Path) -> dict:
    expected = read_json(root / "file_sha256.json")
    mismatches = []
    for relative, expected_hash in expected.items():
        path = root / relative.replace("\\", os.sep)
        if not path.is_file() or digest(path) != expected_hash:
            mismatches.append(relative)
    manifest = read_json(root / "manifest.json")
    return {
        "checked_files": len(expected),
        "mismatches": mismatches,
        "manifest_file_sha256_matches": manifest.get("file_sha256_sha256") == digest(root / "file_sha256.json"),
    }


def input_paths() -> list[Path]:
    paths = [
        PROTOCOL,
        Path(__file__).resolve(),
        ASSIGNMENTS,
        ALLOCATION_ROOT / "manifest.json",
        ASSET,
        ASSET_ROOT / "manifest.json",
        ASSET_ROOT / "file_sha256.json",
        FIT,
        CONFIRMATION_ROOT / "manifest.json",
        CONFIRMATION_ROOT / "file_sha256.json",
        CONFIRMATION_ROOT / "checkpoint_04_gate_audit.json",
    ]
    for partition in PARTITIONS:
        paths.extend([
            CONFIRMATION_ROOT / "full" / partition / "state_matrix.parquet",
            CONFIRMATION_ROOT / "full" / partition / "state_audit.json",
        ])
    return paths


def lock_inputs() -> tuple[dict, str]:
    path = OUT / "input_lock.json"
    inputs = {str(item.relative_to(ROOT)): digest(item) for item in input_paths()}
    asset_audit = verify_hash_tree(ASSET_ROOT)
    confirmation_audit = verify_hash_tree(CONFIRMATION_ROOT)
    allocation = read_json(ALLOCATION_ROOT / "manifest.json")
    allocation_ok = all(
        (ALLOCATION_ROOT / name).is_file() and digest(ALLOCATION_ROOT / name) == expected
        for name, expected in allocation["files"].items()
    )
    require(not asset_audit["mismatches"] and asset_audit["manifest_file_sha256_matches"], "e1_asset_hash_chain_mismatch")
    require(not confirmation_audit["mismatches"] and confirmation_audit["manifest_file_sha256_matches"], "confirmation_hash_chain_mismatch")
    require(allocation_ok, "allocation_hash_chain_mismatch")
    payload = {
        "stage": "unified_evidence_two_partition_loss_comparison_v1_input_lock",
        "created_at_utc": now(),
        "classification": "user_authorized_final_source_fixed_model_residual_loss_comparison_with_prior_e1_outcome_access_disclosed",
        "inputs": inputs,
        "e1_asset_hash_audit": asset_audit,
        "confirmation_hash_audit": confirmation_audit,
        "allocation_manifest_files_verified": allocation_ok,
        "seed": SEED,
        "bootstrap_draws": BOOTSTRAPS,
        "partitions": list(PARTITIONS),
    }
    if path.exists():
        saved = read_json(path)
        require(saved["inputs"] == payload["inputs"], "input_hash_changed_after_lock")
        require(saved["seed"] == SEED and saved["bootstrap_draws"] == BOOTSTRAPS, "locked_settings_changed")
        return saved, digest(path)
    write_json_new(path, payload)
    return payload, digest(path)


def reconstruction_ledger() -> pd.DataFrame:
    columns = [
        "physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id",
        "opponent_player_id", "e_t", "residual_valid", "primary_analysis_eligible", "p_support_0p01",
    ]
    asset = pd.read_parquet(ASSET, columns=columns)
    keep = asset["residual_valid"].astype(bool) & asset["primary_analysis_eligible"].astype(bool) & asset["p_support_0p01"].astype(bool)
    asset = asset.loc[keep].copy()
    focal = asset[["physical_event_id", "match_id", "map_id", "event_tick", "focal_player_id", "e_t"]].rename(columns={"focal_player_id": "player_id"})
    opponent = asset[["physical_event_id", "match_id", "map_id", "event_tick", "opponent_player_id", "e_t"]].rename(columns={"opponent_player_id": "player_id"}).copy()
    opponent["e_t"] = -opponent["e_t"].astype(float)
    ledger = pd.concat([focal, opponent], ignore_index=True)
    for column in ("physical_event_id", "match_id", "map_id", "player_id"):
        ledger[column] = ledger[column].astype(str)
    ledger = ledger.sort_values(["match_id", "map_id", "player_id", "event_tick", "physical_event_id"], kind="stable")
    records: list[tuple[str, str, str, str, float, float]] = []
    for _, group in ledger.groupby(["match_id", "map_id", "player_id"], sort=False):
        prior: pd.DataFrame | None = None
        for _, batch in group.groupby("event_tick", sort=False):
            h = np.nan if prior is None or len(prior) != 1 else float(prior["e_t"].iloc[0])
            records.extend(
                (str(row.physical_event_id), str(row.match_id), str(row.map_id), str(row.player_id), float(row.event_tick), h)
                for row in batch.itertuples(index=False)
            )
            prior = batch
    return pd.DataFrame(records, columns=["physical_event_id", "match_id", "map_id", "player_id", "event_tick", "H_t"])


def partition_frame(partition: str, ledger: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    state_path = CONFIRMATION_ROOT / "full" / partition / "state_matrix.parquet"
    state = pd.read_parquet(state_path)
    frame = state.merge(
        ledger,
        on=["physical_event_id", "match_id", "map_id", "player_id", "event_tick"],
        how="left",
        validate="one_to_one",
    )
    source_rows = len(frame)
    shared = frame.dropna(subset=["H_t"]).copy()
    require(len(shared) > 0, f"no_shared_rows:{partition}")
    require(shared["match_id"].nunique() == 100, f"unexpected_match_count:{partition}")
    fit = read_json(FIT)["models"]
    for model, state_column in (("M1_lag1", "H_t"), ("M3_decay_sum", "S_h60")):
        spec = fit[model]
        z = np.log2(1.0 + shared["d_tick"].to_numpy(float)) - float(spec["mu_train"])
        q = shared[state_column].to_numpy(float) / float(spec["state_scale_train"])
        shared[model] = np.column_stack([np.ones(len(shared)), z, q, q * z]) @ np.asarray(spec["coefficients"], dtype=float)
    finite = bool(np.isfinite(shared[["e_t", "H_t", "S_h60", "M1_lag1", "M3_decay_sum"]].to_numpy(float)).all())
    require(finite, f"nonfinite_values:{partition}")
    audit = {
        "partition": partition,
        "source_rows": int(source_rows),
        "shared_rows": int(len(shared)),
        "excluded_ambiguous_or_no_prior_tick": int(source_rows - len(shared)),
        "matches": int(shared["match_id"].nunique()),
        "finite_predictions": finite,
        "state_matrix_sha256": digest(state_path),
    }
    return shared, audit


def smoke(lock_hash: str, frames: dict[str, pd.DataFrame], audits: dict[str, dict]) -> dict:
    path = OUT / "smoke.json"
    smoke_audits = {}
    for partition in PARTITIONS:
        matches = sorted(frames[partition]["match_id"].astype(str).unique())[:30]
        view = frames[partition].loc[frames[partition]["match_id"].astype(str).isin(matches)]
        smoke_audits[partition] = {
            "matches": int(len(matches)),
            "shared_rows": int(len(view)),
            "source_rows": int(audits[partition]["source_rows"]),
            "finite_predictions": bool(np.isfinite(view[["e_t", "M1_lag1", "M3_decay_sum"]].to_numpy(float)).all()),
            "no_effect_estimates_written": True,
        }
        require(len(matches) == 30 and len(view) > 0 and smoke_audits[partition]["finite_predictions"], f"smoke_failed:{partition}")
    payload = {
        "stage": "unified_evidence_two_partition_loss_comparison_v1_smoke",
        "created_at_utc": now(),
        "input_lock_sha256": lock_hash,
        "status": "pass",
        "partitions": smoke_audits,
    }
    if path.exists():
        saved = read_json(path)
        require(saved["input_lock_sha256"] == lock_hash and saved["status"] == "pass", "smoke_checkpoint_mismatch")
        return saved
    write_json_new(path, payload)
    return payload


def one_partition_comparison(frame: pd.DataFrame, partition_index: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scored = frame[["match_id", "e_t", "M1_lag1", "M3_decay_sum"]].copy()
    for model in ("M1_lag1", "M3_decay_sum"):
        scored[model] = (scored["e_t"] - scored[model]) ** 2
    losses = scored.groupby("match_id", as_index=False)[["M1_lag1", "M3_decay_sum"]].mean()
    losses["partition"] = PARTITIONS[partition_index]
    require(len(losses) == 100 and (losses[["M1_lag1", "M3_decay_sum"]] > 0).all().all(), f"invalid_match_losses:{PARTITIONS[partition_index]}")
    def gain(values: pd.DataFrame) -> float:
        mean = values[["M1_lag1", "M3_decay_sum"]].mean()
        return float((mean["M1_lag1"] - mean["M3_decay_sum"]) / mean["M1_lag1"])
    estimate = gain(losses)
    rng = np.random.default_rng(SEED + partition_index)
    draws = np.empty(BOOTSTRAPS, dtype=float)
    for draw in range(BOOTSTRAPS):
        sample = losses.iloc[rng.integers(len(losses), size=len(losses))]
        draws[draw] = gain(sample)
    low, high = np.quantile(draws, [0.025, 0.975])
    sd = float(draws.std(ddof=1))
    p_value = math.erfc(abs(estimate / sd) / math.sqrt(2.0)) if sd > 0 else (0.0 if estimate != 0 else 1.0)
    draw_frame = pd.DataFrame({"partition": PARTITIONS[partition_index], "draw": np.arange(BOOTSTRAPS), "relative_gain": draws})
    summary = {
        "partition": PARTITIONS[partition_index],
        "matches": int(len(losses)),
        "loss_m1_match_equal": float(losses["M1_lag1"].mean()),
        "loss_m3_match_equal": float(losses["M3_decay_sum"].mean()),
        "relative_gain": estimate,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "ci95_half_width": float((high - low) / 2.0),
        "bootstrap_sd": sd,
        "p_value_two_sided_raw": float(p_value),
    }
    return losses, draw_frame, summary


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[name] = running
    return adjusted


def output_hashes() -> dict[str, str]:
    files = [
        OUT / "input_lock.json", OUT / "smoke.json", OUT / "row_audit.json", OUT / "match_losses.parquet",
        OUT / "bootstrap_gains.parquet", OUT / "comparison.json", OUT / "gates.json",
    ]
    return {str(path.relative_to(OUT)): digest(path) for path in files}


def run(phase: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _, lock_hash = lock_inputs()
    ledger = reconstruction_ledger()
    frames, audits = {}, {}
    for partition in PARTITIONS:
        frames[partition], audits[partition] = partition_frame(partition, ledger)
    smoke_payload = smoke(lock_hash, frames, audits)
    if phase == "smoke":
        print(json.dumps({"status": "passed", "phase": "smoke", "smoke": str(OUT / "smoke.json")}, ensure_ascii=True))
        return
    require(digest(OUT / "smoke.json") == digest(OUT / "smoke.json"), "smoke_hash_unreadable")
    require(smoke_payload["input_lock_sha256"] == lock_hash, "smoke_lock_mismatch")
    row_audit_path = OUT / "row_audit.json"
    if not row_audit_path.exists():
        write_json_new(row_audit_path, {"stage": "unified_evidence_two_partition_loss_comparison_v1_rows", "created_at_utc": now(), "input_lock_sha256": lock_hash, "audits": audits})
    all_losses, all_draws, summaries = [], [], {}
    for index, partition in enumerate(PARTITIONS):
        losses, draws, summary = one_partition_comparison(frames[partition], index)
        all_losses.append(losses)
        all_draws.append(draws)
        summaries[partition] = summary
    losses = pd.concat(all_losses, ignore_index=True)
    draws = pd.concat(all_draws, ignore_index=True)
    if not (OUT / "match_losses.parquet").exists():
        write_parquet_new(OUT / "match_losses.parquet", losses)
    if not (OUT / "bootstrap_gains.parquet").exists():
        write_parquet_new(OUT / "bootstrap_gains.parquet", draws)
    adjusted = holm({partition: summaries[partition]["p_value_two_sided_raw"] for partition in PARTITIONS})
    for partition in PARTITIONS:
        summaries[partition]["p_value_holm"] = adjusted[partition]
    comparison_path = OUT / "comparison.json"
    if not comparison_path.exists():
        write_json_new(comparison_path, {"stage": "unified_evidence_two_partition_loss_comparison_v1_comparison", "created_at_utc": now(), "input_lock_sha256": lock_hash, "inference": "2000_match_cluster_bootstrap_percentile_ci_normal_p_two_partition_holm", "comparisons": summaries})
    superiority = {
        partition: bool(
            summaries[partition]["relative_gain"] > 0
            and summaries[partition]["ci95_low"] > 0
            and summaries[partition]["p_value_holm"] < 0.05
        )
        for partition in PARTITIONS
    }
    precision = {partition: summaries[partition]["ci95_half_width"] <= PRECISION_HALF_WIDTH for partition in PARTITIONS}
    gates = {
        "L1_inputs_and_smoke": {"status": "pass", "reason": "verified_input_hash_chains_and_30_match_smoke"},
        "L2_shared_strict_history_rows": {"status": "pass", "metrics": audits},
        "L3_precision": {"status": "pass" if all(precision.values()) else "fail", "threshold_ci_half_width": PRECISION_HALF_WIDTH, "per_partition": precision},
        "L4_positive_m3_superiority": {"status": "pass" if all(superiority.values()) and all(precision.values()) else "fail", "per_partition": superiority},
        "L5_final_source_record": {"status": "user_adjudicated_final_data_source_with_prior_e1_outcome_access_disclosed"},
    }
    gate_path = OUT / "gates.json"
    if not gate_path.exists():
        write_json_new(gate_path, {"stage": "unified_evidence_two_partition_loss_comparison_v1_gates", "created_at_utc": now(), "input_lock_sha256": lock_hash, "gates": gates})
    hashes_path = OUT / "file_sha256.json"
    if not hashes_path.exists():
        write_json_new(hashes_path, output_hashes())
    else:
        for relative, expected in read_json(hashes_path).items():
            require((OUT / relative).is_file() and digest(OUT / relative) == expected, f"output_hash_mismatch:{relative}")
    manifest_path = OUT / "manifest.json"
    status = "passed" if gates["L4_positive_m3_superiority"]["status"] == "pass" else "completed_no_positive_superiority"
    if not manifest_path.exists():
        write_json_new(manifest_path, {
            "stage": "unified_evidence_two_partition_loss_comparison_v1",
            "created_at_utc": now(),
            "status": status,
            "classification": "user_authorized_final_source_fixed_model_residual_loss_comparison_with_prior_e1_outcome_access_disclosed",
            "input_lock_sha256": lock_hash,
            "smoke_sha256": digest(OUT / "smoke.json"),
            "file_sha256_sha256": digest(hashes_path),
            "gate_statuses": {name: value["status"] for name, value in gates.items()},
            "interpretation": "Fixed-model match-equal Pearson-residual squared-loss comparison only; not p_hat performance, association replacement, causal, latent-state, or opportunity-process evidence.",
        })
    print(json.dumps({"status": status, "manifest": str(manifest_path)}, ensure_ascii=True))


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
        write_json_new(failed / "failure_manifest.json", {"stage": "unified_evidence_two_partition_loss_comparison_v1", "status": "failed", "created_at_utc": now(), "error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
