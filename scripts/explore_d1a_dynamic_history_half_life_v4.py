"""v4 execution wrapper: use the authorized three-fold D1-A development windows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import explore_d1a_dynamic_history_half_life_v1 as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "d1a_dynamic_history_half_life_v4"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "74_D1A动态历史残差概率模型半衰期开发执行锁_v4_2026-08-29.md"
V3 = ROOT / "results" / "d1a_dynamic_history_half_life_v3"
EXECUTOR = Path(__file__).resolve()
IMPLEMENTATION = Path(base.__file__).resolve()
FOLDS = (("F1", 101, 150, 151, 200), ("F2", 101, 200, 201, 250), ("F3", 101, 250, 251, 300))


def input_paths_v4() -> list[Path]:
    paths = [PROTOCOL, EXECUTOR, IMPLEMENTATION, V3 / "checkpoint_01_input_lock.json"]
    for dataset in base.DATASETS:
        paths += [V3 / "state" / dataset / "model_matrix.parquet", V3 / "state" / dataset / f"checkpoint_02_matrix_{dataset}.json"]
    return paths


def build_matrix_v4(dataset: str, lock_hash: str) -> tuple[pd.DataFrame, dict]:
    source = V3 / "state" / dataset / "model_matrix.parquet"
    source_check = V3 / "state" / dataset / f"checkpoint_02_matrix_{dataset}.json"
    path = base.OUT / "state" / dataset / f"checkpoint_02_reused_matrix_{dataset}.json"
    require_source = base.require
    require_source(source.is_file() and source_check.is_file(), f"missing_v3_state:{dataset}")
    if path.exists():
        checkpoint = base.read_json(path)
        require_source(checkpoint.get("input_lock_sha256") == lock_hash and checkpoint.get("source_matrix_sha256") == base.digest(source), f"reused_matrix_checkpoint_mismatch:{dataset}")
        return pd.read_parquet(source), checkpoint
    prior = base.read_json(source_check)
    require_source(prior.get("G1", {}).get("status") == "pass" and prior.get("matrix_sha256") == base.digest(source), f"v3_state_gate_or_hash_failure:{dataset}")
    checkpoint = {"stage": "d1a_dynamic_history_half_life_v4_reused_matrix", "created_at_utc": base.now(), "dataset": dataset, "input_lock_sha256": lock_hash, "source_matrix_path": str(source.relative_to(ROOT)), "source_matrix_sha256": base.digest(source), "source_checkpoint_sha256": base.digest(source_check), "G1": prior["G1"]}
    base.write_json(path, checkpoint)
    return pd.read_parquet(source), checkpoint


def smoke_v4(matrices: dict[str, pd.DataFrame], lock_hash: str) -> None:
    path = base.OUT / "checkpoint_03_smoke.json"
    if path.exists():
        base.require(base.read_json(path).get("input_lock_sha256") == lock_hash, "smoke_input_lock_mismatch")
        return
    train = matrices["D1-A"].loc[matrices["D1-A"]["queue_order"].between(101, 150)].copy()
    matches = sorted(matrices["D1-B"]["match_id"].unique())[:30]
    evaluation = matrices["D1-B"].loc[matrices["D1-B"]["match_id"].isin(matches)].copy()
    base.require(len(matches) == 30 and not train.empty and not evaluation.empty, "smoke_support_failure")
    details = {}
    for candidate in base.candidates():
        probability, fitted = base.fit_predict(train, evaluation, candidate)
        details[candidate["name"]] = {"n_iter": fitted["n_iter"], "probability_range": [float(probability.min()), float(probability.max())], "design_shape": fitted["design_shape"]}
    base.write_json(path, {"stage": "d1a_dynamic_history_half_life_v4_smoke", "created_at_utc": base.now(), "input_lock_sha256": lock_hash, "G0": {"status": "pass"}, "G1": {"status": "pass"}, "G2": {"status": "pass", "training_queue_order": [101, 150], "evaluation_matches": len(matches)}, "G3": {"status": "pass", "all_candidates": details}})


base.OUT = OUT
base.PROTOCOL = PROTOCOL
base.FOLDS = FOLDS
base.input_paths = input_paths_v4
base.build_matrix = build_matrix_v4
base.smoke = smoke_v4


if __name__ == "__main__":
    base.main()
