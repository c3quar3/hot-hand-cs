"""Construct the cache-resumable v3 conditional-residual asset."""

from __future__ import annotations

import hashlib
import hmac
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


MAIN = Path(__file__).resolve().parents[1]
PROTOCOL = MAIN / "docs" / "修订与补丁" / "08_条件残差资产缓存恢复执行协议_v3_2026-08-26.md"
OUT = MAIN / "results" / "conditional_residual_asset_all_datasets_v3"
CACHE = MAIN / "results" / "conditional_residual_asset_candidate_cache_v1" / "E1-v2"
HIST = Path(r"C:\Users\Admin\Desktop\information flow for hot-hand-effect\history\2026-08-25_main_before_restart_20260825\results\p_hat_final_v8_position_continuity\candidate.parquet")
TRAIN = MAIN / "results" / "d1a_c_selection_training_v1"
OOF, MODEL, TRAIN_MANIFEST = TRAIN / "selected_c_oof.parquet", TRAIN / "full_model.joblib", TRAIN / "manifest.json"
D1B = MAIN / "results" / "external_generalization_v3" / "D1-B"
D1C = MAIN / "results" / "external_candidates_v3" / "D1-C"
SCOPES = MAIN / "data" / "confirmation_scopes_v1"
D1_ROOT, E1_ROOT = Path(r"D:\CS2_p_hat_v3_d1_extracted_v1"), Path(r"D:\CS2_p_hat_v3_d1_extracted_e1v2")


def now() -> str: return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path: Path, value: Any) -> None:
    if path.exists(): raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def record(path: Path) -> dict[str, Any]:
    if not path.is_file(): raise RuntimeError(f"required_input_missing:{path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def map_dirs(root: Path, match: str, map_name: str) -> list[Path]:
    base = root / "maps" / match
    if not base.is_dir(): raise RuntimeError(f"missing_raw_match:{match}")
    result = []
    for item in sorted(base.iterdir()):
        manifest = item / "map_manifest.json"
        if manifest.is_file() and str(json.loads(manifest.read_text(encoding="utf-8")).get("header", {}).get("map_name", "")).removeprefix("de_") == map_name: result.append(item)
    if not result: raise RuntimeError(f"missing_raw_map:{match}:{map_name}")
    return result


def raw_files(root: Path, rows: pd.DataFrame, features: bool) -> list[Path]:
    names = ["map_manifest.json", "context.parquet", "outcome_withheld.parquet"]
    if features: names += ["event_stream.parquet", "full_tick_player_state.parquet", "position_remedy.parquet", "weapon_inventory.parquet"]
    found: set[Path] = set()
    for match, map_name in rows[["match_id", "map_name"]].drop_duplicates().itertuples(index=False):
        for part in map_dirs(root, str(match), str(map_name)):
            for name in names:
                path = part / name
                if not path.is_file(): raise RuntimeError(f"required_raw_input_missing:{path}")
                found.add(path)
    return sorted(found)


def canonical_candidate(frame: pd.DataFrame, name: str) -> None:
    fields = {"duel_id", "match_id", "map_id", "map_name", "event_tick", "anchor_tick", "focal_player_key", "opponent_player_key"}
    if fields - set(frame) or frame.empty or frame.duel_id.duplicated().any(): raise RuntimeError(f"candidate_identity_gate_failed:{name}")
    if not frame.anchor_tick.lt(frame.event_tick).all(): raise RuntimeError(f"candidate_time_gate_failed:{name}")
    if not frame.focal_player_key.astype(str).lt(frame.opponent_player_key.astype(str)).all(): raise RuntimeError(f"focal_orientation_not_canonical:{name}")


def matrix(frame: pd.DataFrame, design: dict[str, Any]) -> np.ndarray:
    cols = []
    for col in design["numeric"]:
        x = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
        if not np.isfinite(x).all(): raise RuntimeError(f"nonfinite_feature:{col}")
        cols.append((x - float(design["means"][col])) / float(design["stds"][col]))
    for col in design["categorical"]:
        x, ref = frame[col].astype(str), str(design["references"][col])
        for category in design["categories"][col]:
            if category != ref: cols.append(x.eq(category).to_numpy(float))
    return np.column_stack(cols)


def labels_and_rounds(candidate: pd.DataFrame, root: Path, hashes: dict[str, str], name: str) -> pd.DataFrame:
    salt = (root / "access_control" / "d1_player_hmac_salt.bin").read_bytes()
    output = []
    def key(value: object) -> str:
        try:
            n = int(value)
            return hmac.new(salt, str(n).encode("ascii"), hashlib.sha256).hexdigest() if n > 0 else ""
        except (TypeError, ValueError): return ""
    for (match, map_name), group in candidate.groupby(["match_id", "map_name"], sort=True):
        deaths, starts, sources = [], [], []
        for part in map_dirs(root, str(match), str(map_name)):
            outcome, context = part / "outcome_withheld.parquet", part / "context.parquet"
            sources.append(str(outcome))
            raw = pd.read_parquet(outcome, columns=["tick", "event_type", "attacker_steamid", "user_steamid"])
            raw = raw.loc[raw.event_type.eq("player_death")].copy()
            raw["event_tick"] = pd.to_numeric(raw.tick, errors="coerce")
            raw["attacker_key"], raw["victim_key"] = raw.attacker_steamid.map(key), raw.user_steamid.map(key)
            raw = raw.loc[raw.event_tick.notna() & raw.attacker_key.ne("") & raw.victim_key.ne("")].copy()
            raw.event_tick = raw.event_tick.astype("int64")
            raw["focal_player_key"], raw["opponent_player_key"] = np.minimum(raw.attacker_key, raw.victim_key), np.maximum(raw.attacker_key, raw.victim_key)
            raw["Y_t"] = raw.focal_player_key.eq(raw.attacker_key).astype("int8")
            deaths.append(raw[["event_tick", "focal_player_key", "opponent_player_key", "Y_t"]])
            ctx = pd.read_parquet(context, columns=["tick", "event_type"])
            starts += pd.to_numeric(ctx.loc[ctx.event_type.eq("round_start"), "tick"], errors="coerce").dropna().astype(int).tolist()
        if not deaths or not starts: raise RuntimeError(f"label_or_round_source_missing:{name}:{match}:{map_name}")
        event = pd.concat(deaths, ignore_index=True).drop_duplicates(["event_tick", "focal_player_key", "opponent_player_key"], keep="first")
        joined = group.merge(event, on=["event_tick", "focal_player_key", "opponent_player_key"], how="left", validate="one_to_one")
        if joined.Y_t.isna().any(): raise RuntimeError(f"candidate_label_join_not_exact:{name}:{match}:{map_name}")
        starts_a = np.array(sorted(set(starts)), dtype="int64")
        joined["round_id"] = np.searchsorted(starts_a, joined.event_tick.to_numpy("int64"), side="right") - 1
        if (joined.round_id < 0).any(): raise RuntimeError(f"round_mapping_failed:{name}:{match}:{map_name}")
        joined["raw_event_source_set_sha256"] = json_digest({p: hashes[p] for p in sorted(sources)})
        output.append(joined)
    result = pd.concat(output, ignore_index=True)
    if len(result) != len(candidate) or result.duel_id.duplicated().any() or not set(result.Y_t.unique()).issubset({0, 1}): raise RuntimeError(f"label_integrity_failed:{name}")
    return result


def asset(frame: pd.DataFrame, dataset: str, role: str, provenance: dict[str, str]) -> pd.DataFrame:
    if not frame.p_hat.between(0, 1, inclusive="neither").all() or not set(pd.to_numeric(frame.Y_t, errors="raise").unique()).issubset({0, 1}): raise RuntimeError(f"probability_or_outcome_gate_failed:{dataset}")
    x = frame.copy()
    x["dataset"], x["physical_event_id"] = dataset, dataset + ":" + x.duel_id.astype(str)
    if x.physical_event_id.duplicated().any(): raise RuntimeError(f"physical_event_id_not_unique:{dataset}")
    x["concurrent_group_key"] = x.match_id.astype(str) + ":" + x.map_id.astype(str) + ":" + x.event_tick.astype(str)
    x["order_key"], x["r_t"] = x.concurrent_group_key, x.Y_t.astype(float) - x.p_hat.astype(float)
    x["e_t"] = x.r_t / np.sqrt(x.p_hat.astype(float) * (1 - x.p_hat.astype(float)))
    if not np.isfinite(x[["r_t", "e_t"]].to_numpy(float)).all(): raise RuntimeError(f"residual_numeric_gate_failed:{dataset}")
    x["p_support_0p01"] = x.p_hat.between(.01, .99, inclusive="both")
    x["residual_valid"], x["primary_analysis_eligible"] = True, x.p_support_0p01
    x["exclusion_code"] = np.where(x.p_support_0p01, "", "outside_p_support_0p01")
    x["probability_role"], x["residual_formula_version"], x["source_row_key"] = role, "r=Y-p;e=(Y-p)/sqrt(p*(1-p));v1", x.duel_id.astype(str)
    for col, value in provenance.items(): x[col] = value
    columns = ["dataset", "physical_event_id", "duel_id", "match_id", "map_id", "round_id", "event_tick", "concurrent_group_key", "order_key", "anchor_tick", "focal_player_id", "opponent_player_id", "focal_side", "source_row_key", "Y_t", "p_hat", "r_t", "e_t", "p_support_0p01", "residual_valid", "primary_analysis_eligible", "exclusion_code", "probability_role", "residual_formula_version", "oof_fold_id", "oof_training_match_range", "oof_validation_match_range"] + list(provenance)
    return x[columns].sort_values(["match_id", "map_id", "event_tick", "duel_id"], kind="stable").reset_index(drop=True)


def exclusions(scope: pd.DataFrame, candidate: pd.DataFrame, dataset: str, code: str) -> pd.DataFrame:
    x = scope.loc[~scope.duel_id.isin(candidate.duel_id)].copy()
    x["dataset"], x["physical_event_id"], x["exclusion_code"] = dataset, dataset + ":" + x.duel_id.astype(str), code
    x["residual_valid"], x["primary_analysis_eligible"] = False, False
    return x


def d1a_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = pd.read_parquet(HIST)
    candidate = candidate.loc[candidate.partition.eq("D1-A")].copy()
    oof = pd.read_parquet(OOF)
    if candidate.duel_id.duplicated().any() or not candidate.anchor_tick.lt(candidate.event_tick).all() or not candidate.focal_steamid.astype(str).lt(candidate.opponent_steamid.astype(str)).all(): raise RuntimeError("d1a_candidate_preflight_failed")
    if oof.duel_id.duplicated().any() or not oof.queue_order.between(101, 300).all(): raise RuntimeError("d1a_oof_preflight_failed")
    x = candidate.merge(oof, on=["duel_id", "match_id", "map_id", "queue_order", "focal_wins"], validate="one_to_one")
    if len(x) != len(oof): raise RuntimeError("d1a_oof_candidate_join_not_exact")
    x = x.rename(columns={"focal_wins": "Y_t", "focal_steamid": "focal_player_id", "opponent_steamid": "opponent_player_id"})
    ranges = {"fold_1": ("1-100", "101-150"), "fold_2": ("1-150", "151-200"), "fold_3": ("1-200", "201-250"), "fold_4": ("1-250", "251-300")}
    x["oof_fold_id"] = x.fold; x["oof_training_match_range"] = x.fold.map(lambda f: ranges[f][0]); x["oof_validation_match_range"] = x.fold.map(lambda f: ranges[f][1])
    excluded = candidate.loc[~candidate.duel_id.isin(oof.duel_id)].copy()
    excluded["dataset"], excluded["physical_event_id"], excluded["exclusion_code"] = "D1-A", "D1-A:" + excluded.duel_id.astype(str), "not_saved_forward_oof_coverage"
    excluded["residual_valid"], excluded["primary_analysis_eligible"] = False, False
    return x, excluded


def partition(dataset: str, data: pd.DataFrame, excluded: pd.DataFrame, audit: dict[str, Any]) -> None:
    folder = OUT / dataset
    folder.mkdir(exist_ok=False)
    data.to_parquet(folder / "residual_asset.parquet", index=False, compression="zstd")
    excluded.to_parquet(folder / "exclusion_manifest.parquet", index=False, compression="zstd")
    write(folder / "structural_temporal_audit.json", audit)


def main() -> None:
    if OUT.exists(): raise RuntimeError(f"refusing_to_overwrite_output:{OUT}")
    made = False
    try:
        d1b = pd.read_parquet(D1B / "scored.parquet")
        d1c, d1b_scope, d1c_scope, e1_scope = pd.read_parquet(D1C / "candidate.parquet"), pd.read_parquet(SCOPES / "D1-B" / "candidate_scope_outcome_free.parquet"), pd.read_parquet(SCOPES / "D1-C" / "candidate_scope_outcome_free.parquet"), pd.read_parquet(SCOPES / "E1-v2" / "candidate_scope_outcome_free.parquet")
        canonical_candidate(d1c, "D1-C")
        d1b_candidate = d1b.drop(columns=["focal_wins", "p_hat"]); canonical_candidate(d1b_candidate, "D1-B")
        if not d1c.duel_id.isin(d1c_scope.duel_id).all() or not d1b.duel_id.isin(d1b_scope.duel_id).all(): raise RuntimeError("candidate_scope_join_failed")
        files_d1b, files_d1c, files_e1 = raw_files(D1_ROOT, d1b, False), raw_files(D1_ROOT, d1c, False), raw_files(E1_ROOT, e1_scope, True)
        inputs = [PROTOCOL, HIST, OOF, MODEL, TRAIN_MANIFEST, MAIN / "data" / "d1a_internal_validation_input_mapping_v1.json", D1B / "scored.parquet", D1B / "manifest.json", D1C / "candidate.parquet", D1C / "candidate_manifest.json", SCOPES / "D1-B" / "candidate_scope_outcome_free.parquet", SCOPES / "D1-C" / "candidate_scope_outcome_free.parquet", SCOPES / "E1-v2" / "candidate_scope_outcome_free.parquet", D1_ROOT / "access_control" / "d1_player_hmac_salt.bin", E1_ROOT / "access_control" / "d1_player_hmac_salt.bin", D1_ROOT / "d1_exhaustive_v3_manifest_20260819.json", E1_ROOT / "e1v2_exhaustive_manifest_20260822.json", MAIN / "scripts" / "build_p_hat_external_candidate_v1.py", MAIN / "scripts" / "build_p_hat_external_candidate_v3.py", MAIN / "scripts" / "build_p_hat_external_candidate_v4.py", Path(__file__).resolve()]
        lock = {"stage": "conditional_residual_asset_all_datasets_v3_input_lock", "created_at_utc": now(), "classification": "exploratory_nonconfirmatory_all_datasets", "inputs": {str(p): record(p) for p in inputs}, "raw_input_sha256": {str(p): digest(p) for p in files_d1b + files_d1c + files_e1}, "outcome_access_before_run": {"D1-A": "previously_authorized", "D1-B": "previously_released", "D1-C": "not_read", "E1-v2": "not_read"}}
        OUT.mkdir(parents=True); made = True; write(OUT / "input_lock.json", lock)
        sys.path.insert(0, str(MAIN / "scripts")); import build_p_hat_external_candidate_v4 as builder
        if CACHE.exists():
            provenance = CACHE / "cache_provenance.json"
            if not (CACHE / "candidate.parquet").is_file() or not (CACHE / "candidate_manifest.json").is_file() or not provenance.is_file(): raise RuntimeError("e1_candidate_cache_incomplete")
            cache_lock = json.loads(provenance.read_text(encoding="utf-8"))
            if cache_lock.get("raw_input_sha256") != lock["raw_input_sha256"]: raise RuntimeError("e1_candidate_cache_raw_input_hash_mismatch")
            e1 = pd.read_parquet(CACHE / "candidate.parquet"); e1_manifest = json.loads((CACHE / "candidate_manifest.json").read_text(encoding="utf-8"))
        else:
            staging = OUT / "_staging_e1"; builder.build("E1-v2", staging, 0)
            built = staging / "E1-v2"; e1 = pd.read_parquet(built / "candidate.parquet"); e1_manifest = json.loads((built / "candidate_manifest.json").read_text(encoding="utf-8"))
            canonical_candidate(e1, "E1-v2")
            if e1_manifest.get("status") != "passed" or not e1.duel_id.isin(e1_scope.duel_id).all(): raise RuntimeError("e1_full_candidate_preflight_failed")
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(built), str(CACHE)); staging.rmdir()
            write(CACHE / "cache_provenance.json", {"stage": "conditional_residual_e1_candidate_cache_v1", "created_at_utc": now(), "candidate_sha256": digest(CACHE / "candidate.parquet"), "candidate_manifest_sha256": digest(CACHE / "candidate_manifest.json"), "raw_input_sha256": lock["raw_input_sha256"], "outcome_access": "not_read"})
        canonical_candidate(e1, "E1-v2")
        if e1_manifest.get("status") != "passed" or e1_manifest.get("candidate_sha256") != digest(CACHE / "candidate.parquet") or not e1.duel_id.isin(e1_scope.duel_id).all(): raise RuntimeError("e1_full_candidate_preflight_failed")
        frozen = joblib.load(MODEL)
        if digest(MODEL) != json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))["files"]["full_model.joblib"]: raise RuntimeError("frozen_model_hash_mismatch")
        for frame, name in [(d1c, "D1-C"), (e1, "E1-v2")]:
            frame["p_hat"] = frozen["model"].predict_proba(matrix(frame, frozen["design"]))[:, 1]
            if not frame.p_hat.between(0, 1, inclusive="neither").all(): raise RuntimeError(f"invalid_score:{name}")
        receipts = OUT / "label_access_receipts"; receipts.mkdir()
        for name, rows, root in [("D1-B", d1b_candidate, D1_ROOT), ("D1-C", d1c, D1_ROOT), ("E1-v2", e1, E1_ROOT)]: write(receipts / f"{name}.json", {"stage": "conditional_residual_label_access_receipt_v1", "dataset": name, "created_at_utc": now(), "outcome_access": "authorized_after_input_lock_and_preflight", "candidate_rows": len(rows), "outcome_source_file_count": len([p for p in raw_files(root, rows, False) if p.name == "outcome_withheld.parquet"])})
        d1b_labeled = labels_and_rounds(d1b_candidate, D1_ROOT, lock["raw_input_sha256"], "D1-B")
        d1b_labeled["p_hat"] = d1b.set_index("duel_id").loc[d1b_labeled.duel_id, "p_hat"].to_numpy()
        if not np.array_equal(d1b_labeled.Y_t.to_numpy(int), d1b.set_index("duel_id").loc[d1b_labeled.duel_id, "focal_wins"].to_numpy(int)): raise RuntimeError("d1b_saved_score_label_mismatch")
        d1c_labeled, e1_labeled = labels_and_rounds(d1c, D1_ROOT, lock["raw_input_sha256"], "D1-C"), labels_and_rounds(e1, E1_ROOT, lock["raw_input_sha256"], "E1-v2")
        d1a, d1a_x = d1a_inputs()
        d1a_p = {"model_version": "D1-A_selected_C_0p01_time_forward_oof", "model_artifact_sha256": digest(TRAIN_MANIFEST), "feature_contract_version": "D1-A_frozen_training_v1", "scorer_sha256": digest(MAIN / "scripts" / "train_d1a_c_selection_v1.py"), "input_candidate_sha256": digest(HIST), "oof_file_sha256": digest(OOF), "model_selection_status": "selected_c_oof_used_for_C_selection", "field_source_version": "historical_candidate_plus_saved_oof_v1", "raw_event_source_sha256": digest(HIST)}
        common = {"model_version": "D1-A_final_model_selected_C_0p01", "model_artifact_sha256": digest(MODEL), "feature_contract_version": "D1-A_frozen_training_v1", "scorer_sha256": digest(Path(__file__).resolve()), "oof_file_sha256": "", "oof_fold_id": "", "oof_training_match_range": "", "oof_validation_match_range": "", "model_selection_status": "frozen_model_external_nonconfirmatory", "field_source_version": "frozen_external_candidate_and_withheld_outcome_v1"}
        a = asset(d1a, "D1-A", "forward_oof_post_selection", d1a_p)
        d1b_labeled = d1b_labeled.rename(columns={"focal_steamid": "focal_player_id", "opponent_steamid": "opponent_player_id"})
        d1c_labeled = d1c_labeled.rename(columns={"focal_player_key": "focal_player_id", "opponent_player_key": "opponent_player_id"})
        e1_labeled = e1_labeled.rename(columns={"focal_player_key": "focal_player_id", "opponent_player_key": "opponent_player_id"})
        b = asset(d1b_labeled, "D1-B", "fixed_model_external_score_previously_released", common | {"input_candidate_sha256": digest(D1B / "scored.parquet"), "raw_event_source_sha256": json_digest(json.loads((receipts / "D1-B.json").read_text()))})
        c = asset(d1c_labeled, "D1-C", "fixed_model_external_score_nonconfirmatory", common | {"input_candidate_sha256": digest(D1C / "candidate.parquet"), "raw_event_source_sha256": json_digest(json.loads((receipts / "D1-C.json").read_text()))})
        e = asset(e1_labeled, "E1-v2", "fixed_model_external_score_nonconfirmatory", common | {"input_candidate_sha256": digest(CACHE / "candidate.parquet"), "raw_event_source_sha256": json_digest(json.loads((receipts / "E1-v2.json").read_text()))})
        b_x, c_x, e_x = exclusions(d1b_scope, d1b, "D1-B", "not_in_saved_scored_candidate"), exclusions(d1c_scope, d1c, "D1-C", "candidate_feature_construction_exclusion"), exclusions(e1_scope, e1, "E1-v2", "candidate_feature_construction_exclusion")
        for name, data, excluded in [("D1-A", a, d1a_x), ("D1-B", b, b_x), ("D1-C", c, c_x), ("E1-v2", e, e_x)]: partition(name, data, excluded, {"status": "passed", "rows": len(data), "round_source": "historical_candidate" if name == "D1-A" else "raw_context_round_start", "same_tick_rule": "concurrent_group_key=match_id:map_id:event_tick"})
        write(OUT / "E1-v2" / "input_candidate_reference.json", {"cache_path": str(CACHE), "candidate_sha256": digest(CACHE / "candidate.parquet"), "candidate_manifest_sha256": digest(CACHE / "candidate_manifest.json"), "cache_provenance_sha256": digest(CACHE / "cache_provenance.json")})
        write(OUT / "field_dictionary.json", {"formula": "r_t=Y_t-p_hat; e_t=(Y_t-p_hat)/sqrt(p_hat*(1-p_hat))", "p_support_0p01": "0.01 <= p_hat <= 0.99", "physical_event_id": "dataset + ':' + duel_id", "classification": "exploratory_nonconfirmatory_all_datasets"})
        hashes = {str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob("*")) if p.is_file()}; write(OUT / "file_sha256.json", hashes)
        stats = {name: {"rows": len(data), "matches": data.match_id.nunique(), "maps": data.map_id.nunique(), "primary_analysis_eligible": int(data.primary_analysis_eligible.sum()), "outside_p_support_0p01": int((~data.p_support_0p01).sum()), "structural_exclusions": len(ex), "probability_role": data.probability_role.iloc[0]} for name, data, ex in [("D1-A", a, d1a_x), ("D1-B", b, b_x), ("D1-C", c, c_x), ("E1-v2", e, e_x)]}
        write(OUT / "manifest.json", {"stage": "conditional_residual_asset_all_datasets_v3", "created_at_utc": now(), "status": "passed", "classification": "exploratory_nonconfirmatory_all_datasets", "interpretation": "Auditable conditional p_hat/r_t/e_t asset only; not a hot-hand, state, causal, calibration, or predictive-gain result.", "input_lock_sha256": digest(OUT / "input_lock.json"), "partitions": stats, "file_sha256_sha256": digest(OUT / "file_sha256.json")})
        print(json.dumps({"status": "passed", "partitions": stats}, ensure_ascii=True, indent=2))
    except Exception as error:
        if made and OUT.exists(): shutil.rmtree(OUT)
        failure = OUT / f"failed_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"; failure.mkdir(parents=True, exist_ok=False)
        write(failure / "failure_manifest.json", {"stage": "conditional_residual_asset_all_datasets_v3", "status": "failed", "created_at_utc": now(), "error_type": type(error).__name__, "reason": str(error), "sensitive_outputs_written": False})
        raise


if __name__ == "__main__": main()
