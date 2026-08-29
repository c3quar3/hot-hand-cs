"""Versioned legal-empty-strict-scope repair for the R1 core extractor."""

from __future__ import annotations

import pandas as pd

import extract_r1_70_core_preoutcome_v3 as v3


base = v3.v2.base
KEEP = ["duel_id", "match_id", "map_id", "round", "event_tick", "pre_event_tick", "focal_player_key", "opponent_player_key", "atomic_tick_batch_id", "atomic_tick_batch_size", "atomic_tick_participant_count", "atomic_tick_is_multidual", "atomic_tick_shared_participant", "atomic_tick_history_policy"]


def scoped_duels_with_legal_empty_scope(pilot, staged, item, salt):
    snapshots, duels, _, summary = pilot.parse_map(
        demo=staged,
        source_match_id=str(item["source_match_id"]),
        source_map_id=f"r1m{int(item['map_ordinal']):04d}",
        match_date_utc=str(item["match_date_utc"]),
        tick_rate=64.0,
        player_id_key=salt,
    )
    audit = {"pilot_summary": {"map_name": summary.get("map_name"), "missing_parser_fields": summary.get("missing_parser_fields", [])}}
    if duels.empty:
        return pd.DataFrame({column: pd.Series(dtype="object") for column in KEEP}), audit | {"strict_duel_count": 0, "strict_scope_status": "legal_empty"}
    focal = snapshots[["event_tick", "player_slot", "player_id_hash"]].drop_duplicates().rename(columns={"player_slot": "focal_player_slot", "player_id_hash": "focal_player_key"})
    opponent = snapshots[["event_tick", "player_slot", "player_id_hash"]].drop_duplicates().rename(columns={"player_slot": "opponent_player_slot", "player_id_hash": "opponent_player_key"})
    scope = duels.merge(focal, on=["event_tick", "focal_player_slot"], how="left", validate="many_to_one")
    scope = scope.merge(opponent, on=["event_tick", "opponent_player_slot"], how="left", validate="many_to_one")
    scope = scope.dropna(subset=["focal_player_key", "opponent_player_key"]).copy()
    scope = scope[KEEP].drop_duplicates("duel_id").sort_values(["event_tick", "duel_id"], kind="stable")
    return scope, audit | {"strict_duel_count": int(len(scope)), "strict_scope_status": "nonempty"}


base.scoped_duels = scoped_duels_with_legal_empty_scope
# The base executor records its module-level __file__; bind it to this repair.
base.__file__ = __file__


if __name__ == "__main__":
    base.main()
