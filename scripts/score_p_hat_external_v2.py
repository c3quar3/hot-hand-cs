"""Preflight or score one external p_hat candidate with a fixed D1-A model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse


MAIN = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = MAIN / "data" / "confirmation_scopes_v1" / "manifest.json"
DEFAULT_MODEL = MAIN / "results" / "d1a_c_selection_training_v1" / "full_model.joblib"
DEFAULT_MODEL_MANIFEST = MAIN / "results" / "d1a_c_selection_training_v1" / "manifest.json"
DEFAULT_OUTPUT = MAIN / "results" / "external_generalization_v2"
AUTHORIZATION_ENV = "PHAT_EXTERNAL_SCORE_AUTHORIZATION"
EPSILON = 1e-15


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


def load_expected(lock_path: Path, dataset: str) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("outcome_access") != "not_read":
        raise RuntimeError("scope_lock_outcome_access_not_clean")
    return next((item for item in lock["datasets"] if item["dataset"] == dataset), None) or (_ for _ in ()).throw(RuntimeError("dataset_not_in_scope_lock"))


def verify_model(model_path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("files", {}).get("full_model.joblib")
    if not isinstance(expected, str) or expected != sha256(model_path):
        raise RuntimeError("model_manifest_hash_mismatch")
    artifact = joblib.load(model_path)
    if not {"model", "design", "selected_c"}.issubset(artifact):
        raise RuntimeError("invalid_model_artifact")
    design = artifact["design"]
    needed = {"numeric", "categorical", "means", "stds", "categories", "references"}
    if not needed.issubset(design):
        raise RuntimeError("invalid_model_design")
    return artifact


def verify_candidate(candidate_path: Path, candidate_manifest_path: Path, expected: dict[str, Any], artifact: dict[str, Any]) -> pd.DataFrame:
    manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_scope_sha256") != expected["locked_scope"]["sha256"]:
        raise RuntimeError("candidate_not_built_from_locked_scope")
    if manifest.get("candidate_sha256") != sha256(candidate_path):
        raise RuntimeError("candidate_manifest_hash_mismatch")
    if manifest.get("outcome_access") not in ("not_read", "outcome_free"):
        raise RuntimeError("candidate_manifest_outcome_access_invalid")
    if manifest.get("temporal_audit", {}).get("status") != "passed":
        raise RuntimeError("candidate_temporal_audit_not_passed")
    frame = pd.read_parquet(candidate_path)
    design = artifact["design"]
    required = set(design["numeric"]) | set(design["categorical"]) | {"duel_id", "match_id", "map_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError("candidate_missing_columns:" + ",".join(missing))
    if frame.empty or frame["duel_id"].duplicated().any():
        raise RuntimeError("candidate_invalid_duel_id")
    for column in design["numeric"]:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
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


def report_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    p = np.clip(frame["p_hat"].to_numpy(float), EPSILON, 1 - EPSILON)
    y = frame["focal_wins"].to_numpy(int)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    brier = (p - y) ** 2
    grouped = pd.DataFrame({"match_id": frame["match_id"], "loss": loss, "brier": brier}).groupby("match_id", sort=True).mean()
    logit = np.log(p / (1 - p))
    matrix = np.column_stack((np.ones(len(p)), logit))
    beta = np.zeros(2)
    for _ in range(100):
        fitted = 1 / (1 + np.exp(-np.clip(matrix @ beta, -35, 35)))
        weight = np.maximum(fitted * (1 - fitted), 1e-9)
        step = np.linalg.solve(matrix.T @ (weight[:, None] * matrix), matrix.T @ (y - fitted))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    bins = pd.qcut(pd.Series(p), q=10, duplicates="drop")
    calibration = pd.DataFrame({"bin": bins, "p_hat": p, "focal_wins": y}).groupby("bin", observed=True).agg(p_hat=("p_hat", "mean"), focal_wins=("focal_wins", "mean"), rows=("focal_wins", "size")).reset_index(drop=True)
    return {
        "rows": int(len(frame)), "matches": int(frame["match_id"].nunique()),
        "match_equal_log_loss": float(grouped["loss"].mean()), "match_equal_brier": float(grouped["brier"].mean()),
        "row_brier": float(brier.mean()), "calibration_intercept": float(beta[0]), "calibration_slope": float(beta[1]),
        "calibration_ece_10_bin": float((calibration["rows"] * (calibration["p_hat"] - calibration["focal_wins"]).abs()).sum() / len(frame)),
        "calibration_bin_count": int(len(calibration)), "calibration_bins": calibration.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "score"))
    parser.add_argument("dataset", choices=("D1-B", "D1-C", "E1-v2"))
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--scope-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not all(path.is_file() for path in (args.scope_lock, args.model, args.model_manifest, args.candidate, args.candidate_manifest)):
        raise RuntimeError("required_nonoutcome_input_missing")
    expected = load_expected(args.scope_lock, args.dataset)
    artifact = verify_model(args.model, args.model_manifest)
    candidate = verify_candidate(args.candidate, args.candidate_manifest, expected, artifact)
    output = args.output_root / args.dataset
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    common = {
        "dataset": args.dataset, "created_at_utc": datetime.now(UTC).isoformat(), "scope_lock_sha256": sha256(args.scope_lock),
        "source_scope_sha256": expected["locked_scope"]["sha256"], "candidate_sha256": sha256(args.candidate),
        "candidate_manifest_sha256": sha256(args.candidate_manifest), "model_sha256": sha256(args.model),
        "model_manifest_sha256": sha256(args.model_manifest), "script_sha256": sha256(Path(__file__).resolve()),
        "candidate_rows": int(len(candidate)), "candidate_matches": int(candidate["match_id"].nunique()),
    }
    if args.action == "preflight":
        common["outcome_access"] = "not_read"
        write_new_json(output / "preflight.json", common)
        print(json.dumps(common, ensure_ascii=True, indent=2))
        return
    if args.labels is None or not args.labels.is_file():
        raise RuntimeError("labels_required_for_score")
    if os.environ.get(AUTHORIZATION_ENV) != f"release_{args.dataset}_once":
        raise RuntimeError(f"missing_dataset_release_authorization:{args.dataset}")
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
    output.mkdir(parents=True)
    scored.to_parquet(output / "scored.parquet", index=False, compression="zstd")
    report = report_metrics(scored)
    report.update(common | {"outcome_access": "released_once", "labels_sha256": sha256(args.labels), "scored_sha256": sha256(output / "scored.parquet"), "unknown_categories": unknown})
    write_new_json(output / "report.json", report)
    print(json.dumps({key: report[key] for key in ("dataset", "rows", "matches", "match_equal_log_loss", "match_equal_brier")}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
