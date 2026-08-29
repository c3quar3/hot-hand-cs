"""Materialize one locked external p_hat label table after an explicit release."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MAIN = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"D:\CS2_p_hat_v3_d1_extracted_v1")
SALT_PATH = DATA_ROOT / "access_control" / "d1_player_hmac_salt.bin"
AUTHORIZATION_ENV = "PHAT_EXTERNAL_SCORE_AUTHORIZATION"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def load_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    scripts = lock.get("scripts", {})
    actual = sha256(Path(__file__).resolve())
    if scripts.get(Path(__file__).name) != actual:
        raise RuntimeError("execution_lock_script_hash_mismatch")
    if lock.get("dataset") != "D1-B" or lock.get("outcome_access_before_run") != "not_read":
        raise RuntimeError("execution_lock_invalid")
    return lock


def player_key(salt: bytes, value: object) -> str:
    try:
        steamid = int(value)
    except (TypeError, ValueError):
        return ""
    if steamid <= 0:
        return ""
    return hmac.new(salt, str(steamid).encode("ascii"), hashlib.sha256).hexdigest()


def map_dirs(match_id: str, map_name: str) -> list[Path]:
    root = DATA_ROOT / "maps" / match_id
    if not root.is_dir():
        raise RuntimeError(f"missing_raw_match:{match_id}")
    found: list[Path] = []
    for directory in sorted(root.iterdir()):
        manifest = directory / "map_manifest.json"
        if not manifest.is_file():
            continue
        name = str(json.loads(manifest.read_text(encoding="utf-8")).get("header", {}).get("map_name", "")).removeprefix("de_")
        if name == map_name:
            found.append(directory)
    if not found:
        raise RuntimeError(f"missing_raw_map:{match_id}:{map_name}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("D1-B",))
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--execution-lock", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    output = args.output_root / args.dataset
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    if os.environ.get(AUTHORIZATION_ENV) != f"release_{args.dataset}_once":
        raise RuntimeError(f"missing_dataset_release_authorization:{args.dataset}")
    if not all(path.is_file() for path in (args.candidate, args.candidate_manifest, args.execution_lock, SALT_PATH)):
        raise RuntimeError("required_input_missing")
    lock = load_lock(args.execution_lock)
    candidate_manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    if (
        lock.get("candidate_sha256") != sha256(args.candidate)
        or lock.get("candidate_manifest_sha256") != sha256(args.candidate_manifest)
        or lock.get("hmac_salt_sha256") != sha256(SALT_PATH)
        or candidate_manifest.get("candidate_sha256") != sha256(args.candidate)
        or candidate_manifest.get("source_scope_sha256") != lock.get("source_scope_sha256")
    ):
        raise RuntimeError("candidate_hash_mismatch")
    candidate = pd.read_parquet(args.candidate, columns=["duel_id", "match_id", "map_id", "map_name", "event_tick", "focal_player_key", "opponent_player_key"])
    if candidate.empty or candidate["duel_id"].duplicated().any():
        raise RuntimeError("invalid_candidate_keys")

    sources: list[Path] = []
    for (match_id, map_name), _ in candidate.groupby(["match_id", "map_name"], sort=True):
        for directory in map_dirs(str(match_id), str(map_name)):
            path = directory / "outcome_withheld.parquet"
            if not path.is_file():
                raise RuntimeError(f"missing_outcome_source:{path}")
            sources.append(path)
    sources = sorted(set(sources))
    output.mkdir(parents=True)
    receipt = {
        "stage": "p_hat_external_label_release_receipt_v1",
        "dataset": args.dataset,
        "authorization": f"release_{args.dataset}_once",
        "created_at_utc": utc_now(),
        "outcome_access": "authorized_before_read",
        "execution_lock_sha256": sha256(args.execution_lock),
        "candidate_sha256": sha256(args.candidate),
        "candidate_manifest_sha256": sha256(args.candidate_manifest),
        "materializer_sha256": sha256(Path(__file__).resolve()),
        "source_files_planned": [str(path) for path in sources],
        "source_file_count": len(sources),
    }
    write_new_json(output / "access_receipt.json", receipt)

    salt = SALT_PATH.read_bytes()
    raw_frames: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}
    for path in sources:
        source_hashes[str(path)] = sha256(path)
        frame = pd.read_parquet(path, columns=["tick", "event_type", "attacker_steamid", "user_steamid"])
        frame = frame.loc[frame["event_type"].eq("player_death")].copy()
        if not frame.empty:
            frame["source_path"] = str(path)
            raw_frames.append(frame)
    deaths = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    if deaths.empty:
        raise RuntimeError("no_player_death_labels")
    deaths["event_tick"] = pd.to_numeric(deaths["tick"], errors="coerce")
    deaths = deaths.dropna(subset=["event_tick"]).copy()
    deaths["event_tick"] = deaths["event_tick"].astype("int64")
    deaths["attacker_key"] = deaths["attacker_steamid"].map(lambda value: player_key(salt, value))
    deaths["victim_key"] = deaths["user_steamid"].map(lambda value: player_key(salt, value))
    deaths = deaths.loc[deaths["attacker_key"].ne("") & deaths["victim_key"].ne("")].copy()
    deaths["focal_player_key"] = np.minimum(deaths["attacker_key"], deaths["victim_key"])
    deaths["opponent_player_key"] = np.maximum(deaths["attacker_key"], deaths["victim_key"])
    deaths["focal_wins"] = deaths["focal_player_key"].eq(deaths["attacker_key"]).astype("int8")

    labels: list[pd.DataFrame] = []
    for (match_id, map_name), group in candidate.groupby(["match_id", "map_name"], sort=True):
        paths = {str(directory / "outcome_withheld.parquet") for directory in map_dirs(str(match_id), str(map_name))}
        outcome = deaths.loc[deaths["source_path"].isin(paths), ["event_tick", "focal_player_key", "opponent_player_key", "focal_wins"]].drop_duplicates(
            subset=["event_tick", "focal_player_key", "opponent_player_key"], keep="first"
        )
        joined = group.merge(outcome, on=["event_tick", "focal_player_key", "opponent_player_key"], how="left", validate="one_to_one")
        if joined["focal_wins"].isna().any():
            raise RuntimeError(f"candidate_label_join_not_exact:{match_id}:{map_name}")
        labels.append(joined[["duel_id", "focal_wins"]])
    label_frame = pd.concat(labels, ignore_index=True)
    if label_frame["duel_id"].duplicated().any() or not set(label_frame["focal_wins"].unique()).issubset({0, 1}):
        raise RuntimeError("labels_invalid")
    if len(label_frame) != len(candidate):
        raise RuntimeError("label_row_count_mismatch")
    label_path = output / "labels.parquet"
    label_frame.to_parquet(label_path, index=False, compression="zstd")
    manifest = receipt | {
        "stage": "p_hat_external_label_materialization_v1",
        "outcome_access": "released_once",
        "labels_sha256": sha256(label_path),
        "labels_rows": int(len(label_frame)),
        "labels_columns": list(label_frame.columns),
        "source_file_sha256": source_hashes,
    }
    write_new_json(output / "label_manifest.json", manifest)
    print(json.dumps({"dataset": args.dataset, "labels_rows": len(label_frame), "labels_sha256": manifest["labels_sha256"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
