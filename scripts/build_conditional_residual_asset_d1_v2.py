"""Build the resumable D1-only v2 conditional-residual asset."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import build_conditional_residual_asset_all_datasets_v2 as base


MAIN = Path(__file__).resolve().parents[1]
PROTOCOL = MAIN / "docs" / "修订与补丁" / "10_D1条件残差资产重复物理事件修复执行协议_v2_2026-08-26.md"
OUT = MAIN / "results" / "conditional_residual_asset_d1_v2"
HIST, OOF, MODEL, TRAIN_MANIFEST = base.HIST, base.OOF, base.MODEL, base.TRAIN_MANIFEST
D1B, D1C, SCOPES, RAW = base.D1B, base.D1C, base.SCOPES, base.D1_ROOT


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_new(path: Path, value: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required_input_missing:{path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": base.digest(path)}


def raw_sources(frame: pd.DataFrame) -> list[Path]:
    found: set[Path] = set()
    for match_id, map_name in frame[["match_id", "map_name"]].drop_duplicates().itertuples(index=False):
        for directory in base.map_dirs(RAW, str(match_id), str(map_name)):
            for name in ("map_manifest.json", "context.parquet", "outcome_withheld.parquet"):
                path = directory / name
                if not path.is_file():
                    raise RuntimeError(f"required_raw_input_missing:{path}")
                found.add(path)
    return sorted(found)


def verify_checkpoint(dataset: str, input_lock_sha256: str) -> bool:
    folder, path = OUT / dataset, OUT / dataset / "checkpoint.json"
    if not folder.exists():
        return False
    if not path.is_file():
        raise RuntimeError(f"incomplete_checkpoint:{dataset}")
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "input_lock_sha256": input_lock_sha256,
        "asset_sha256": base.digest(folder / "residual_asset.parquet"),
        "exclusion_manifest_sha256": base.digest(folder / "exclusion_manifest.parquet"),
        "audit_sha256": base.digest(folder / "structural_temporal_audit.json"),
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"checkpoint_hash_mismatch:{dataset}")
    return True


def write_checkpoint(dataset: str, asset: pd.DataFrame, exclusions: pd.DataFrame, audit: dict[str, Any], input_lock_sha256: str) -> None:
    folder = OUT / dataset
    if folder.exists():
        raise RuntimeError(f"checkpoint_destination_exists:{dataset}")
    folder.mkdir()
    asset.to_parquet(folder / "residual_asset.parquet", index=False, compression="zstd")
    exclusions.to_parquet(folder / "exclusion_manifest.parquet", index=False, compression="zstd")
    write_new(folder / "structural_temporal_audit.json", audit)
    write_new(folder / "checkpoint.json", {
        "stage": f"conditional_residual_asset_d1_v2_{dataset}",
        "completed_at_utc": utc_now(),
        "input_lock_sha256": input_lock_sha256,
        "asset_sha256": base.digest(folder / "residual_asset.parquet"),
        "exclusion_manifest_sha256": base.digest(folder / "exclusion_manifest.parquet"),
        "audit_sha256": base.digest(folder / "structural_temporal_audit.json"),
    })


def external_provenance(candidate_sha256: str, raw_event_sha256: str) -> dict[str, str]:
    return {
        "model_version": "D1-A_final_model_selected_C_0p01",
        "model_artifact_sha256": base.digest(MODEL),
        "feature_contract_version": "D1-A_frozen_training_v1",
        "scorer_sha256": base.digest(Path(__file__).resolve()),
        "input_candidate_sha256": candidate_sha256,
        "model_selection_status": "frozen_model_external_nonconfirmatory",
        "field_source_version": "frozen_external_candidate_and_withheld_outcome_v1",
        "raw_event_source_sha256": raw_event_sha256,
    }


def deduplicate_exact_physical_events(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep min(duel_id) only when every other candidate field is identical."""
    keys = ["match_id", "map_id", "event_tick", "focal_player_key", "opponent_player_key"]
    duplicate = frame.loc[frame.duplicated(keys, keep=False)].copy()
    if duplicate.empty:
        return frame, pd.DataFrame(columns=list(frame.columns) + ["representative_duel_id", "exclusion_code"])
    value_columns = [column for column in frame.columns if column != "duel_id"]
    changed = duplicate.groupby(keys, dropna=False)[value_columns].nunique(dropna=False).max()
    changed = changed.loc[changed.gt(1)].index.tolist()
    if changed:
        raise RuntimeError(f"duplicate_physical_event_fields_differ:{','.join(changed)}")
    if not duplicate.groupby(keys, dropna=False).size().eq(2).all():
        raise RuntimeError("duplicate_physical_event_not_exactly_two_rows")
    representative = duplicate.groupby(keys, dropna=False)["duel_id"].transform("min")
    removed = duplicate.loc[duplicate["duel_id"].ne(representative)].copy()
    removed["representative_duel_id"] = representative.loc[removed.index].to_numpy()
    removed["exclusion_code"] = "duplicate_physical_event_representative_min_duel_id"
    return frame.loc[~frame["duel_id"].isin(removed["duel_id"])].copy(), removed


def main() -> None:
    d1b = pd.read_parquet(D1B / "scored.parquet")
    d1c = pd.read_parquet(D1C / "candidate.parquet")
    d1b_scope = pd.read_parquet(SCOPES / "D1-B" / "candidate_scope_outcome_free.parquet")
    d1c_scope = pd.read_parquet(SCOPES / "D1-C" / "candidate_scope_outcome_free.parquet")
    d1b_candidate = d1b.drop(columns=["focal_wins", "p_hat"])
    base.canonical_candidate(d1b_candidate, "D1-B")
    base.canonical_candidate(d1c, "D1-C")
    d1c, d1c_duplicate_records = deduplicate_exact_physical_events(d1c)
    if not d1b["duel_id"].isin(d1b_scope["duel_id"]).all() or not d1c["duel_id"].isin(d1c_scope["duel_id"]).all():
        raise RuntimeError("candidate_scope_join_failed")
    paths = raw_sources(d1b) + raw_sources(d1c)
    raw_hashes = {str(path): base.digest(path) for path in paths}
    inputs = [
        PROTOCOL, HIST, OOF, MODEL, TRAIN_MANIFEST,
        MAIN / "data" / "d1a_internal_validation_input_mapping_v1.json",
        D1B / "scored.parquet", D1B / "manifest.json",
        D1C / "candidate.parquet", D1C / "candidate_manifest.json",
        SCOPES / "D1-B" / "candidate_scope_outcome_free.parquet",
        SCOPES / "D1-C" / "candidate_scope_outcome_free.parquet",
        RAW / "access_control" / "d1_player_hmac_salt.bin",
        RAW / "d1_exhaustive_v3_manifest_20260819.json",
        MAIN / "scripts" / "build_p_hat_external_candidate_v1.py",
        MAIN / "scripts" / "build_p_hat_external_candidate_v3.py",
        Path(__file__).resolve(),
    ]
    lock = {
        "stage": "conditional_residual_asset_d1_v2_input_lock",
        "created_at_utc": utc_now(),
        "classification": "exploratory_nonconfirmatory_all_datasets",
        "inputs": {str(path): file_record(path) for path in inputs},
        "raw_input_sha256": raw_hashes,
        "outcome_access_before_run": {"D1-A": "previously_authorized", "D1-B": "previously_released", "D1-C": "not_read"},
    }
    if not OUT.exists():
        OUT.mkdir(parents=True)
        write_new(OUT / "input_lock.json", lock)
    elif not (OUT / "input_lock.json").is_file():
        raise RuntimeError("output_without_input_lock")
    input_lock_sha256 = base.digest(OUT / "input_lock.json")
    saved_lock = json.loads((OUT / "input_lock.json").read_text(encoding="utf-8"))
    if saved_lock.get("raw_input_sha256") != raw_hashes:
        raise RuntimeError("resume_input_lock_hash_mismatch")
    if base.digest(MODEL) != json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))["files"]["full_model.joblib"]:
        raise RuntimeError("frozen_model_hash_mismatch")
    if not verify_checkpoint("D1-A", input_lock_sha256):
        frame, excluded = base.d1a_inputs()
        provenance = {
            "model_version": "D1-A_selected_C_0p01_time_forward_oof",
            "model_artifact_sha256": base.digest(TRAIN_MANIFEST),
            "feature_contract_version": "D1-A_frozen_training_v1",
            "scorer_sha256": base.digest(MAIN / "scripts" / "train_d1a_c_selection_v1.py"),
            "input_candidate_sha256": base.digest(HIST),
            "oof_file_sha256": base.digest(OOF),
            "model_selection_status": "selected_c_oof_used_for_C_selection",
            "field_source_version": "historical_candidate_plus_saved_oof_v1",
            "raw_event_source_sha256": base.digest(HIST),
        }
        write_checkpoint("D1-A", base.asset(frame, "D1-A", "forward_oof_post_selection", provenance), excluded, {"status": "passed", "round_source": "historical_candidate", "same_tick_rule": "concurrent_group_key=match_id:map_id:event_tick"}, input_lock_sha256)
    if not verify_checkpoint("D1-B", input_lock_sha256):
        receipt = OUT / "label_access_receipts" / "D1-B.json"
        if not receipt.exists():
            write_new(receipt, {"stage": "conditional_residual_label_access_receipt_v1", "dataset": "D1-B", "created_at_utc": utc_now(), "outcome_access": "previously_released_reconciled_after_input_lock", "candidate_rows": int(len(d1b_candidate))})
        labeled = base.labels_and_rounds(d1b_candidate, RAW, raw_hashes, "D1-B")
        labeled["p_hat"] = d1b.set_index("duel_id").loc[labeled["duel_id"], "p_hat"].to_numpy()
        expected = d1b.set_index("duel_id").loc[labeled["duel_id"], "focal_wins"].to_numpy(int)
        if not np.array_equal(labeled["Y_t"].to_numpy(int), expected):
            raise RuntimeError("d1b_saved_score_label_mismatch")
        labeled = labeled.rename(columns={"focal_steamid": "focal_player_id", "opponent_steamid": "opponent_player_id"})
        for column in ("oof_fold_id", "oof_training_match_range", "oof_validation_match_range"):
            labeled[column] = ""
        write_checkpoint("D1-B", base.asset(labeled, "D1-B", "fixed_model_external_score_previously_released", external_provenance(base.digest(D1B / "scored.parquet"), base.json_digest(json.loads(receipt.read_text(encoding="utf-8"))))), base.exclusions(d1b_scope, d1b, "D1-B", "not_in_saved_scored_candidate"), {"status": "passed", "round_source": "raw_context_round_start", "same_tick_rule": "concurrent_group_key=match_id:map_id:event_tick"}, input_lock_sha256)
    if not verify_checkpoint("D1-C", input_lock_sha256):
        frozen = joblib.load(MODEL)
        d1c["p_hat"] = frozen["model"].predict_proba(base.matrix(d1c, frozen["design"]))[:, 1]
        if not d1c["p_hat"].between(0, 1, inclusive="neither").all():
            raise RuntimeError("invalid_score:D1-C")
        receipt = OUT / "label_access_receipts" / "D1-C.json"
        if not receipt.exists():
            write_new(receipt, {"stage": "conditional_residual_label_access_receipt_v2", "dataset": "D1-C", "created_at_utc": utc_now(), "outcome_access": "previously_accessed_in_d1_v1_hard_stop_reconnected_after_v2_input_lock_candidate_preflight_and_frozen_score", "candidate_rows": int(len(d1c)), "duplicate_physical_event_rows_excluded_before_label_connection": int(len(d1c_duplicate_records)), "probability_role": "fixed_model_external_score_nonconfirmatory"})
        labeled = base.labels_and_rounds(d1c, RAW, raw_hashes, "D1-C").rename(columns={"focal_player_key": "focal_player_id", "opponent_player_key": "opponent_player_id"})
        for column in ("oof_fold_id", "oof_training_match_range", "oof_validation_match_range"):
            labeled[column] = ""
        d1c_exclusions = base.exclusions(d1c_scope, d1c, "D1-C", "candidate_feature_construction_exclusion")
        duplicate_ids = set(d1c_duplicate_records["duel_id"])
        d1c_exclusions.loc[d1c_exclusions["duel_id"].isin(duplicate_ids), "exclusion_code"] = "duplicate_physical_event_representative_min_duel_id"
        d1c_exclusions = d1c_exclusions.merge(d1c_duplicate_records[["duel_id", "representative_duel_id"]], on="duel_id", how="left", validate="one_to_one")
        audit = {"status": "passed", "round_source": "raw_context_round_start", "same_tick_rule": "concurrent_group_key=match_id:map_id:event_tick", "confirmatory_status": "ineligible_nonconfirmatory", "duplicate_physical_event_rule": "identical non-duel_id candidate fields; retain min(duel_id), record the other row as an exclusion", "duplicate_physical_event_rows_excluded": int(len(d1c_duplicate_records))}
        write_checkpoint("D1-C", base.asset(labeled, "D1-C", "fixed_model_external_score_nonconfirmatory", external_provenance(base.digest(D1C / "candidate.parquet"), base.json_digest(json.loads(receipt.read_text(encoding="utf-8"))))), d1c_exclusions, audit, input_lock_sha256)
    if not (OUT / "field_dictionary.json").exists():
        write_new(OUT / "field_dictionary.json", {"formula": "r_t=Y_t-p_hat; e_t=(Y_t-p_hat)/sqrt(p_hat*(1-p_hat))", "p_support_0p01": "0.01 <= p_hat <= 0.99", "physical_event_id": "dataset + ':' + duel_id", "classification": "exploratory_nonconfirmatory_all_datasets"})
    if not (OUT / "file_sha256.json").exists():
        write_new(OUT / "file_sha256.json", {str(path.relative_to(OUT)): base.digest(path) for path in sorted(OUT.rglob("*")) if path.is_file()})
    if not (OUT / "manifest.json").exists():
        summary = {}
        for dataset in ("D1-A", "D1-B", "D1-C"):
            data, exclusions = pd.read_parquet(OUT / dataset / "residual_asset.parquet"), pd.read_parquet(OUT / dataset / "exclusion_manifest.parquet")
            summary[dataset] = {"rows": int(len(data)), "matches": int(data["match_id"].nunique()), "maps": int(data["map_id"].nunique()), "primary_analysis_eligible": int(data["primary_analysis_eligible"].sum()), "outside_p_support_0p01": int((~data["p_support_0p01"]).sum()), "structural_exclusions": int(len(exclusions)), "probability_role": str(data["probability_role"].iloc[0])}
        write_new(OUT / "manifest.json", {"stage": "conditional_residual_asset_d1_v2", "created_at_utc": utc_now(), "status": "passed", "classification": "exploratory_nonconfirmatory_all_datasets", "interpretation": "Auditable D1 conditional p_hat/r_t/e_t asset only; not a hot-hand, state, causal, calibration, or predictive-gain result.", "input_lock_sha256": input_lock_sha256, "partitions": summary, "file_sha256_sha256": base.digest(OUT / "file_sha256.json")})
    print((OUT / "manifest.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
