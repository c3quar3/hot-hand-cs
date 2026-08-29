"""Score one released external p_hat label table with a frozen execution lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse


AUTHORIZATION_ENV = "PHAT_EXTERNAL_SCORE_AUTHORIZATION"
EPSILON = 1e-15
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260825


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def load_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("dataset") != "D1-B" or lock.get("outcome_access_before_run") != "not_read":
        raise RuntimeError("execution_lock_invalid")
    if lock.get("scripts", {}).get(Path(__file__).name) != sha256(Path(__file__).resolve()):
        raise RuntimeError("execution_lock_script_hash_mismatch")
    return lock


def verify_model(model_path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("files", {}).get("full_model.joblib") != sha256(model_path):
        raise RuntimeError("model_manifest_hash_mismatch")
    artifact = joblib.load(model_path)
    if not {"model", "design", "selected_c"}.issubset(artifact):
        raise RuntimeError("invalid_model_artifact")
    required = {"numeric", "categorical", "means", "stds", "categories", "references"}
    if not required.issubset(artifact["design"]):
        raise RuntimeError("invalid_model_design")
    return artifact


def verify_candidate(candidate_path: Path, manifest_path: Path, expected_scope: str, design: dict[str, Any]) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_scope_sha256") != expected_scope or manifest.get("candidate_sha256") != sha256(candidate_path):
        raise RuntimeError("candidate_manifest_hash_mismatch")
    if manifest.get("outcome_access") not in {"not_read", "outcome_free"} or manifest.get("temporal_audit", {}).get("status") != "passed":
        raise RuntimeError("candidate_manifest_not_eligible")
    frame = pd.read_parquet(candidate_path)
    required = set(design["numeric"]) | set(design["categorical"]) | {"duel_id", "match_id", "map_id", "anchor_tick", "event_tick"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError("candidate_missing_columns:" + ",".join(missing))
    if frame.empty or frame["duel_id"].duplicated().any() or not (frame["anchor_tick"] < frame["event_tick"]).all():
        raise RuntimeError("candidate_invalid_keys_or_time")
    for column in design["numeric"]:
        if not np.isfinite(pd.to_numeric(frame[column], errors="raise").to_numpy(float)).all():
            raise RuntimeError(f"candidate_nonfinite_numeric:{column}")
    for column in design["categorical"]:
        values = frame[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise RuntimeError(f"candidate_missing_category:{column}")
    return frame


def design_matrix(frame: pd.DataFrame, design: dict[str, Any]) -> tuple[sparse.csr_matrix, dict[str, int]]:
    numeric = np.column_stack([
        (pd.to_numeric(frame[column], errors="raise").to_numpy(float) - float(design["means"][column])) / float(design["stds"][column])
        for column in design["numeric"]
    ])
    blocks: list[sparse.csr_matrix] = [sparse.csr_matrix(numeric)]
    unknown: dict[str, int] = {}
    for column in design["categorical"]:
        categories = [str(value) for value in design["categories"][column]]
        reference = str(design["references"][column])
        lookup = {value: index for index, value in enumerate(value for value in categories if value != reference)}
        values = frame[column].astype(str).to_numpy()
        unknown[column] = int(sum(value not in categories for value in values))
        rows = np.fromiter((index for index, value in enumerate(values) if value in lookup), dtype=np.int64)
        columns = np.fromiter((lookup[value] for value in values if value in lookup), dtype=np.int64)
        blocks.append(sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(frame), len(lookup))))
    return sparse.hstack(blocks, format="csr"), unknown


def calibration(probability: np.ndarray, outcome: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    logits = np.log(probability / (1 - probability))
    matrix = np.column_stack((np.ones(len(logits)), logits))
    beta = np.zeros(2)
    for _ in range(100):
        fitted = 1 / (1 + np.exp(-np.clip(matrix @ beta, -35, 35)))
        weight = np.maximum(fitted * (1 - fitted), 1e-9)
        step = np.linalg.solve(matrix.T @ (weight[:, None] * matrix), matrix.T @ (outcome - fitted))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    bins = pd.qcut(pd.Series(probability), q=10, duplicates="drop")
    table = pd.DataFrame({"bin": bins, "p_hat": probability, "focal_wins": outcome}).groupby("bin", observed=True).agg(p_hat=("p_hat", "mean"), focal_wins=("focal_wins", "mean"), rows=("focal_wins", "size")).reset_index(drop=True)
    ece = float((table["rows"] * (table["p_hat"] - table["focal_wins"]).abs()).sum() / len(probability))
    return {"calibration_intercept": float(beta[0]), "calibration_slope": float(beta[1]), "calibration_ece_10_bin": ece, "calibration_bin_count": int(len(table))}, table


def match_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    p = np.clip(scored["p_hat"].to_numpy(float), EPSILON, 1 - EPSILON)
    y = scored["focal_wins"].to_numpy(int)
    frame = pd.DataFrame({"match_id": scored["match_id"].astype(str), "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)), "brier": (p - y) ** 2})
    return frame.groupby("match_id", sort=True).agg(rows=("log_loss", "size"), match_log_loss=("log_loss", "mean"), match_brier=("brier", "mean")).reset_index()


def bootstrap(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    values = metrics[["match_log_loss", "match_brier"]].to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records: list[dict[str, float | int]] = []
    for iteration in range(BOOTSTRAP_ITERATIONS):
        sample = values[rng.integers(0, len(values), size=len(values))].mean(axis=0)
        records.append({"iteration": iteration + 1, "match_equal_log_loss": float(sample[0]), "match_equal_brier": float(sample[1])})
    draws = pd.DataFrame(records)
    summary = {
        column: {"p2_5": float(draws[column].quantile(0.025)), "p50": float(draws[column].quantile(0.5)), "p97_5": float(draws[column].quantile(0.975))}
        for column in ("match_equal_log_loss", "match_equal_brier")
    }
    return draws, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("D1-B",))
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--label-manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--execution-lock", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root / args.dataset
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    if os.environ.get(AUTHORIZATION_ENV) != f"release_{args.dataset}_once":
        raise RuntimeError(f"missing_dataset_release_authorization:{args.dataset}")
    if not all(path.is_file() for path in (args.candidate, args.candidate_manifest, args.labels, args.label_manifest, args.model, args.model_manifest, args.execution_lock)):
        raise RuntimeError("scoring_input_missing")
    lock = load_lock(args.execution_lock)
    if (
        lock.get("candidate_sha256") != sha256(args.candidate)
        or lock.get("candidate_manifest_sha256") != sha256(args.candidate_manifest)
        or lock.get("model_sha256") != sha256(args.model)
        or lock.get("model_manifest_sha256") != sha256(args.model_manifest)
    ):
        raise RuntimeError("execution_lock_input_hash_mismatch")
    label_manifest = json.loads(args.label_manifest.read_text(encoding="utf-8"))
    if label_manifest.get("labels_sha256") != sha256(args.labels) or label_manifest.get("candidate_sha256") != sha256(args.candidate):
        raise RuntimeError("label_manifest_hash_mismatch")
    artifact = verify_model(args.model, args.model_manifest)
    candidate = verify_candidate(args.candidate, args.candidate_manifest, str(lock["source_scope_sha256"]), artifact["design"])
    labels = pd.read_parquet(args.labels, columns=["duel_id", "focal_wins"])
    if labels["duel_id"].duplicated().any() or not set(pd.to_numeric(labels["focal_wins"], errors="raise").unique()).issubset({0, 1}):
        raise RuntimeError("labels_invalid")
    scored = candidate.merge(labels, on="duel_id", how="inner", validate="one_to_one")
    if len(scored) != len(candidate) or len(scored) != len(labels):
        raise RuntimeError("candidate_label_join_not_exact")
    matrix, unknown = design_matrix(scored, artifact["design"])
    scored["p_hat"] = artifact["model"].predict_proba(matrix)[:, 1]
    if not np.isfinite(scored["p_hat"]).all() or not scored["p_hat"].between(0, 1, inclusive="neither").all():
        raise RuntimeError("invalid_probability")
    per_match = match_metrics(scored)
    draws, intervals = bootstrap(per_match)
    calibrate, bins = calibration(scored["p_hat"].to_numpy(float), scored["focal_wins"].to_numpy(int))
    output.mkdir(parents=True)
    scored.to_parquet(output / "scored.parquet", index=False, compression="zstd")
    per_match.to_parquet(output / "per_match_metrics.parquet", index=False, compression="zstd")
    bins.to_parquet(output / "calibration_bins.parquet", index=False, compression="zstd")
    draws.to_parquet(output / "bootstrap_draws.parquet", index=False, compression="zstd")
    report: dict[str, Any] = {
        "stage": "p_hat_external_score_v3", "dataset": args.dataset, "created_at_utc": datetime.now(UTC).isoformat(),
        "outcome_access": "released_once", "rows": int(len(scored)), "matches": int(len(per_match)), "maps": int(scored["map_id"].nunique()),
        "match_equal_log_loss": float(per_match["match_log_loss"].mean()), "match_equal_brier": float(per_match["match_brier"].mean()),
        "row_brier": float(((scored["p_hat"] - scored["focal_wins"]) ** 2).mean()), "unknown_categories": unknown,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "iterations": BOOTSTRAP_ITERATIONS, "intervals": intervals}, **calibrate,
        "execution_lock_sha256": sha256(args.execution_lock), "candidate_sha256": sha256(args.candidate), "candidate_manifest_sha256": sha256(args.candidate_manifest),
        "labels_sha256": sha256(args.labels), "label_manifest_sha256": sha256(args.label_manifest), "model_sha256": sha256(args.model),
        "model_manifest_sha256": sha256(args.model_manifest), "script_sha256": sha256(Path(__file__).resolve()),
        "scored_sha256": sha256(output / "scored.parquet"), "per_match_metrics_sha256": sha256(output / "per_match_metrics.parquet"),
        "calibration_bins_sha256": sha256(output / "calibration_bins.parquet"), "bootstrap_draws_sha256": sha256(output / "bootstrap_draws.parquet"),
    }
    write_new_json(output / "report.json", report)
    write_new_json(output / "manifest.json", report | {"report_sha256": sha256(output / "report.json")})
    print(json.dumps({key: report[key] for key in ("dataset", "rows", "matches", "match_equal_log_loss", "match_equal_brier")}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
