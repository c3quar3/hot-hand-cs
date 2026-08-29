"""Create an outcome-free, nonconfirmatory D1-C exclusion variant.

This script never changes the frozen D1-C scope or candidate. It derives a
separate package excluding one whole match and refuses to score or read labels.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


MAIN = Path(__file__).resolve().parents[1]
DATASET = "D1-C"
EXCLUDED_MATCH_ID = "2394779"
PARENT_SCOPE = MAIN / "data" / "confirmation_scopes_v1" / DATASET / "candidate_scope_outcome_free.parquet"
PARENT_CANDIDATE = MAIN / "results" / "external_candidates_v3" / DATASET / "candidate.parquet"
PARENT_MANIFEST = MAIN / "results" / "external_candidates_v3" / DATASET / "candidate_manifest.json"
OUTPUT = MAIN / "results" / "nonconfirmatory_d1c_match_2394779_exclusion_v1" / DATASET


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing_to_overwrite:{OUTPUT}")
    parent_manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    if parent_manifest.get("outcome_access") != "not_read" or parent_manifest.get("status") != "passed":
        raise RuntimeError("parent_candidate_is_not_a_passed_outcome_free_artifact")
    scope = pd.read_parquet(PARENT_SCOPE)
    candidate = pd.read_parquet(PARENT_CANDIDATE)
    excluded_scope = scope.loc[scope["match_id"].astype(str).eq(EXCLUDED_MATCH_ID)].copy()
    excluded_candidate = candidate.loc[candidate["match_id"].astype(str).eq(EXCLUDED_MATCH_ID)].copy()
    if excluded_scope.empty or excluded_candidate.empty:
        raise RuntimeError("excluded_match_not_present_in_both_parent_artifacts")
    if set(excluded_scope["map_id"].astype(str)) != {"2394779_ancient"}:
        raise RuntimeError("unexpected_match_map_membership")
    new_scope = scope.loc[~scope["match_id"].astype(str).eq(EXCLUDED_MATCH_ID)].copy()
    new_candidate = candidate.loc[~candidate["match_id"].astype(str).eq(EXCLUDED_MATCH_ID)].copy()
    if new_scope["duel_id"].duplicated().any() or new_candidate["duel_id"].duplicated().any():
        raise RuntimeError("duplicate_duel_id_after_exclusion")
    if set(new_candidate["duel_id"]) - set(new_scope["duel_id"]):
        raise RuntimeError("candidate_not_subset_of_new_scope")
    if not new_candidate["anchor_tick"].lt(new_candidate["event_tick"]).all():
        raise RuntimeError("temporal_gate_failed")
    forbidden = [column for column in new_candidate.columns if any(token in column.lower() for token in ("outcome", "winner", "label", "target", "win_"))]
    if forbidden:
        raise RuntimeError(f"outcome_like_columns_detected:{forbidden}")
    OUTPUT.mkdir(parents=True)
    new_scope.to_parquet(OUTPUT / "candidate_scope_outcome_free.parquet", index=False, compression="zstd")
    new_candidate.to_parquet(OUTPUT / "candidate.parquet", index=False, compression="zstd")
    manifest = {
        "stage": "d1c_nonconfirmatory_match_exclusion_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "passed_nonconfirmatory_preparation_only",
        "outcome_access": "not_read",
        "dataset": DATASET,
        "classification": "post_D1-B_result specification variant; ineligible for the frozen D1-C confirmatory scoring protocol",
        "prohibited_actions_under_frozen_confirmatory_protocol": ["D1-C_label_materialization", "D1-C_scoring", "model_adjustment", "confirmation_claim"],
        "future_research_boundary": "Any label materialization or score requires a separately authorized protocol and must be reported as nonconfirmatory.",
        "parent_scope": {"path": str(PARENT_SCOPE), "sha256": sha256(PARENT_SCOPE), "rows": int(len(scope))},
        "parent_candidate": {"path": str(PARENT_CANDIDATE), "sha256": sha256(PARENT_CANDIDATE), "rows": int(len(candidate))},
        "parent_candidate_manifest": {"path": str(PARENT_MANIFEST), "sha256": sha256(PARENT_MANIFEST)},
        "excluded_match": {
            "match_id": EXCLUDED_MATCH_ID,
            "queue_order": sorted(excluded_scope["queue_order"].astype(int).unique().tolist()),
            "map_ids": sorted(excluded_scope["map_id"].astype(str).unique().tolist()),
            "scope_rows": int(len(excluded_scope)),
            "candidate_rows": int(len(excluded_candidate)),
        },
        "output": {
            "scope_sha256": sha256(OUTPUT / "candidate_scope_outcome_free.parquet"),
            "candidate_sha256": sha256(OUTPUT / "candidate.parquet"),
            "scope_rows": int(len(new_scope)),
            "candidate_rows": int(len(new_candidate)),
            "matches": int(new_candidate["match_id"].nunique()),
            "maps": int(new_candidate["map_id"].nunique()),
        },
        "audits": {
            "excluded_match_absent_from_scope": bool(not new_scope["match_id"].astype(str).eq(EXCLUDED_MATCH_ID).any()),
            "excluded_match_absent_from_candidate": bool(not new_candidate["match_id"].astype(str).eq(EXCLUDED_MATCH_ID).any()),
            "candidate_duel_ids_unique": True,
            "candidate_subset_of_scope": True,
            "anchor_strictly_before_event": True,
            "outcome_like_columns": forbidden,
        },
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "scope_rows": manifest["output"]["scope_rows"], "candidate_rows": manifest["output"]["candidate_rows"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
