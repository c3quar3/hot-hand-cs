"""Build the E1-v2 exploratory conditional-residual asset from a locked cache."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

import build_conditional_residual_asset_all_datasets_v3 as base
from build_conditional_residual_asset_d1_v2 import deduplicate_exact_physical_events


MAIN = Path(__file__).resolve().parents[1]
PROTOCOL = MAIN / "docs" / "修订与补丁" / "11_E1条件残差资产执行协议_v1_2026-08-26.md"
OUT = MAIN / "results" / "conditional_residual_asset_e1_v1"
CACHE = MAIN / "results" / "conditional_residual_asset_candidate_cache_v1" / "E1-v2"
SCOPE = MAIN / "data" / "confirmation_scopes_v1" / "E1-v2" / "candidate_scope_outcome_free.parquet"


def now() -> str:
    return datetime.now(UTC).isoformat()


def write(path: Path, value: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"required_input_missing:{path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": base.digest(path)}


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing_to_overwrite_output:{OUT}")
    made = False
    try:
        required_cache = (CACHE / "candidate.parquet", CACHE / "candidate_manifest.json", CACHE / "cache_provenance.json")
        if not all(path.is_file() for path in required_cache):
            raise RuntimeError("e1_candidate_cache_incomplete")
        scope = pd.read_parquet(SCOPE)
        candidate = pd.read_parquet(CACHE / "candidate.parquet")
        manifest = json.loads((CACHE / "candidate_manifest.json").read_text(encoding="utf-8"))
        cache = json.loads((CACHE / "cache_provenance.json").read_text(encoding="utf-8"))
        base.canonical_candidate(candidate, "E1-v2")
        candidate, duplicate_rows = deduplicate_exact_physical_events(candidate)
        if not candidate.duel_id.isin(scope.duel_id).all():
            raise RuntimeError("candidate_scope_join_failed:E1-v2")
        raw_paths = base.raw_files(base.E1_ROOT, scope, True)
        raw_hashes = {str(path): base.digest(path) for path in raw_paths}
        cache_hashes = {path: value for path, value in cache.get("raw_input_sha256", {}).items() if path.startswith(str(base.E1_ROOT))}
        if raw_hashes != cache_hashes:
            raise RuntimeError("e1_candidate_cache_raw_input_hash_mismatch")
        if manifest.get("status") != "passed" or manifest.get("candidate_sha256") != base.digest(CACHE / "candidate.parquet") or cache.get("candidate_sha256") != base.digest(CACHE / "candidate.parquet"):
            raise RuntimeError("e1_candidate_cache_manifest_mismatch")
        inputs = [
            PROTOCOL, Path(__file__).resolve(), CACHE / "candidate.parquet", CACHE / "candidate_manifest.json", CACHE / "cache_provenance.json", SCOPE,
            base.MODEL, base.TRAIN_MANIFEST, base.E1_ROOT / "access_control" / "d1_player_hmac_salt.bin", base.E1_ROOT / "e1v2_exhaustive_manifest_20260822.json",
            MAIN / "scripts" / "build_p_hat_external_candidate_v4.py", MAIN / "scripts" / "build_conditional_residual_asset_d1_v2.py",
        ]
        lock = {
            "stage": "conditional_residual_asset_e1_v1_input_lock", "created_at_utc": now(), "classification": "exploratory_nonconfirmatory_all_datasets",
            "inputs": {str(path): record(path) for path in inputs}, "raw_input_sha256": raw_hashes,
            "outcome_access_before_run": "not_read", "duplicate_physical_event_rule": "identical non-duel_id candidate fields; retain min(duel_id), record the other row as an exclusion",
        }
        OUT.mkdir(parents=True); made = True; write(OUT / "input_lock.json", lock)
        if base.digest(base.MODEL) != json.loads(base.TRAIN_MANIFEST.read_text(encoding="utf-8"))["files"]["full_model.joblib"]:
            raise RuntimeError("frozen_model_hash_mismatch")
        frozen = joblib.load(base.MODEL)
        candidate["p_hat"] = frozen["model"].predict_proba(base.matrix(candidate, frozen["design"]))[:, 1]
        if not candidate.p_hat.between(0, 1, inclusive="neither").all():
            raise RuntimeError("invalid_score:E1-v2")
        receipt = OUT / "label_access_receipt.json"
        write(receipt, {
            "stage": "conditional_residual_label_access_receipt_e1_v1", "dataset": "E1-v2", "created_at_utc": now(),
            "outcome_access": "authorized_after_input_lock_cache_preflight_and_frozen_score", "candidate_rows": len(candidate),
            "duplicate_physical_event_rows_excluded_before_label_connection": len(duplicate_rows), "probability_role": "fixed_model_external_score_nonconfirmatory",
        })
        labeled = base.labels_and_rounds(candidate, base.E1_ROOT, raw_hashes, "E1-v2")
        labeled = labeled.rename(columns={"focal_player_key": "focal_player_id", "opponent_player_key": "opponent_player_id"})
        for column in ("oof_fold_id", "oof_training_match_range", "oof_validation_match_range"):
            labeled[column] = ""
        provenance = {
            "model_version": "D1-A_final_model_selected_C_0p01", "model_artifact_sha256": base.digest(base.MODEL),
            "feature_contract_version": "D1-A_frozen_training_v1", "scorer_sha256": base.digest(Path(__file__).resolve()),
            "input_candidate_sha256": base.digest(CACHE / "candidate.parquet"), "oof_file_sha256": "",
            "model_selection_status": "frozen_model_external_nonconfirmatory", "field_source_version": "cached_outcome_free_candidate_and_withheld_outcome_v1",
            "raw_event_source_sha256": base.json_digest(json.loads(receipt.read_text(encoding="utf-8"))),
        }
        residual = base.asset(labeled, "E1-v2", "fixed_model_external_score_nonconfirmatory", provenance)
        excluded = base.exclusions(scope, candidate, "E1-v2", "candidate_feature_construction_exclusion")
        duplicate_ids = set(duplicate_rows.duel_id)
        excluded.loc[excluded.duel_id.isin(duplicate_ids), "exclusion_code"] = "duplicate_physical_event_representative_min_duel_id"
        excluded = excluded.merge(duplicate_rows[["duel_id", "representative_duel_id"]], on="duel_id", how="left", validate="one_to_one")
        folder = OUT / "E1-v2"; folder.mkdir()
        residual.to_parquet(folder / "residual_asset.parquet", index=False, compression="zstd")
        excluded.to_parquet(folder / "exclusion_manifest.parquet", index=False, compression="zstd")
        write(folder / "structural_temporal_audit.json", {
            "status": "passed", "round_source": "raw_context_round_start", "same_tick_rule": "concurrent_group_key=match_id:map_id:event_tick",
            "duplicate_physical_event_rule": "identical non-duel_id candidate fields; retain min(duel_id), record the other row as an exclusion",
            "duplicate_physical_event_rows_excluded": len(duplicate_rows), "candidate_cache_sha256": base.digest(CACHE / "candidate.parquet"),
        })
        write(folder / "input_candidate_reference.json", {"cache_path": str(CACHE), "candidate_sha256": base.digest(CACHE / "candidate.parquet"), "candidate_manifest_sha256": base.digest(CACHE / "candidate_manifest.json"), "cache_provenance_sha256": base.digest(CACHE / "cache_provenance.json")})
        write(OUT / "field_dictionary.json", {"formula": "r_t=Y_t-p_hat; e_t=(Y_t-p_hat)/sqrt(p_hat*(1-p_hat))", "p_support_0p01": "0.01 <= p_hat <= 0.99", "physical_event_id": "dataset + ':' + duel_id", "classification": "exploratory_nonconfirmatory_all_datasets"})
        hashes = {str(path.relative_to(OUT)): base.digest(path) for path in sorted(OUT.rglob("*")) if path.is_file()}; write(OUT / "file_sha256.json", hashes)
        summary = {"rows": len(residual), "matches": residual.match_id.nunique(), "maps": residual.map_id.nunique(), "primary_analysis_eligible": int(residual.primary_analysis_eligible.sum()), "outside_p_support_0p01": int((~residual.p_support_0p01).sum()), "structural_exclusions": len(excluded), "probability_role": residual.probability_role.iloc[0]}
        write(OUT / "manifest.json", {"stage": "conditional_residual_asset_e1_v1", "created_at_utc": now(), "status": "passed", "classification": "exploratory_nonconfirmatory_all_datasets", "interpretation": "Auditable E1-v2 conditional p_hat/r_t/e_t asset only; not a hot-hand, state, causal, calibration, or predictive-gain result.", "input_lock_sha256": base.digest(OUT / "input_lock.json"), "partition": summary, "file_sha256_sha256": base.digest(OUT / "file_sha256.json")})
        print(json.dumps({"status": "passed", "E1-v2": summary}, ensure_ascii=True, indent=2))
    except Exception as error:
        if made and OUT.exists(): shutil.rmtree(OUT)
        failure = OUT / f"failed_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"; failure.mkdir(parents=True, exist_ok=False)
        write(failure / "failure_manifest.json", {"stage": "conditional_residual_asset_e1_v1", "status": "failed", "created_at_utc": now(), "error_type": type(error).__name__, "reason": str(error), "sensitive_outputs_written": False, "candidate_cache_retained": CACHE.exists()})
        raise


if __name__ == "__main__":
    main()
