"""Build the D1-C outcome-free external p_hat candidate from raw inputs.

This version reuses the frozen D1-B v1 feature implementation without changing
it.  It is intentionally authorized for D1-C only and never opens an outcome
file.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import build_p_hat_external_candidate_v1 as base


OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "results" / "external_candidates_v2"


def build(dataset: str, output_root: Path, max_maps: int) -> None:
    if dataset != "D1-C":
        raise RuntimeError("v2_is_authorized_for_D1-C_only")
    scope_path = base.SCOPE_ROOT / dataset / "candidate_scope_outcome_free.parquet"
    output = output_root / dataset
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    scope = pd.read_parquet(scope_path)
    if scope["duel_id"].duplicated().any():
        raise RuntimeError("scope_duplicate_duel_id")
    salt = (base.DATA_ROOT / "access_control" / "d1_player_hmac_salt.bin").read_bytes()
    all_rows, exclusions, map_audits = [], Counter(), []
    groups = list(scope.groupby(["match_id", "map_name"], sort=True))
    for index, ((match_id, map_name), map_scope) in enumerate(groups, start=1):
        if max_maps and index > max_maps:
            break
        paths = base.map_directories(str(match_id), str(map_name))
        context = base.concat_tables(paths, "context.parquet", ["tick", "event_type"])
        fires, identities = base.map_events(paths, salt)
        planned, removed = base.planned_rows(map_scope, fires, context)
        exclusions.update(removed)
        if planned.empty:
            map_audits.append({"map_id": str(map_scope.iloc[0].map_id), "input_rows": len(map_scope), "output_rows": 0, "exclusions": dict(removed)})
            continue
        ticks = set(planned["anchor_tick"].astype(int)) | set(planned["post_buy_snapshot_tick"].astype(int))
        states = base.load_states(paths, ticks)
        motion = base.continuous_motion(paths, set(planned["anchor_tick"].astype(int)))
        anchor_ticks = sorted(set(planned["anchor_tick"].astype(int)))
        weapon_raw = base.concat_tables(paths, "weapon_inventory.parquet", ["tick", "player_id_hash", "Weapon.m_iItemDefinitionIndex", "Weapon.m_zoomLevel"], [("tick", "in", anchor_ticks)])
        weapons = {(int(tick), str(key)): (int(item) if pd.notna(item) else 0, float(zoom) if pd.notna(zoom) else 0.0) for tick, key, item, zoom in weapon_raw.drop_duplicates(["tick", "player_id_hash"], keep="last").itertuples(index=False, name=None)}
        rebuilt, removed = base.candidate_features(planned, states, motion, weapons, identities, context)
        exclusions.update(removed)
        all_rows.extend(rebuilt)
        map_audits.append({"map_id": str(map_scope.iloc[0].map_id), "input_rows": int(len(map_scope)), "output_rows": int(len(rebuilt)), "exclusions": dict(removed)})
        print(f"maps={index}/{min(len(groups), max_maps or len(groups))} rows={len(all_rows)}", flush=True)
    partial = bool(max_maps and max_maps < len(groups))
    candidate = pd.DataFrame(all_rows)
    if candidate.empty or candidate["duel_id"].duplicated().any():
        raise RuntimeError("invalid_rebuilt_candidate")
    required = set(base.NUMERIC) | set(base.CATEGORICAL) | {"duel_id", "match_id", "map_id", "event_tick", "anchor_tick"}
    if required - set(candidate):
        raise RuntimeError("candidate_contract_missing")
    if not np.isfinite(candidate[base.NUMERIC].to_numpy(float)).all() or not candidate["anchor_tick"].lt(candidate["event_tick"]).all():
        raise RuntimeError("candidate_feature_or_timing_gate_failed")
    output.mkdir(parents=True)
    candidate.to_parquet(output / "candidate.parquet", index=False, compression="zstd")
    manifest = {
        "stage": "p_hat_external_raw_candidate_v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "partial_smoke" if partial else "passed",
        "outcome_access": "not_read",
        "dataset": dataset,
        "source_scope_sha256": base.sha256(scope_path),
        "candidate_sha256": base.sha256(output / "candidate.parquet"),
        "builder_sha256": base.sha256(Path(__file__).resolve()),
        "reused_feature_builder_path": str(Path(base.__file__).resolve()),
        "reused_feature_builder_sha256": base.sha256(Path(base.__file__).resolve()),
        "anchor_rule": "first weapon_fire by either locked participant in [event_tick-640,event_tick], minus 32 ticks",
        "temporal_audit": {"status": "passed", "anchor_strictly_before_event": True},
        "input_rows": int(len(scope)),
        "processed_maps": int(len(map_audits)),
        "output_rows": int(len(candidate)),
        "excluded_rows": int(sum(exclusions.values())),
        "exclusions": dict(exclusions),
        "map_audits": map_audits,
        "feature_contract": {"numeric": base.NUMERIC, "categorical": base.CATEGORICAL},
    }
    base.write_new_json(output / "candidate_manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "dataset", "processed_maps", "output_rows", "excluded_rows")}, ensure_ascii=True, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("D1-C",))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-maps", type=int, default=0)
    args = parser.parse_args()
    build(args.dataset, args.output_root, args.max_maps)


if __name__ == "__main__":
    main()
