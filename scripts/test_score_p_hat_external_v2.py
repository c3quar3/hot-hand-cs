"""Synthetic, outcome-free integration tests for score_p_hat_external_v2."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


SCRIPT = Path(__file__).with_name("score_p_hat_external_v2.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], text=True, capture_output=True, check=False, env=env)


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    scope = root / "scope.json"
    scope.write_text(json.dumps({"outcome_access": "not_read", "datasets": [{"dataset": "D1-B", "locked_scope": {"sha256": "scope-hash"}}]}), encoding="utf-8")
    candidate = root / "candidate.parquet"
    frame = pd.DataFrame({"duel_id": ["a", "b", "c", "d"], "match_id": ["m1", "m1", "m2", "m2"], "map_id": ["x", "x", "y", "y"], "x": [0.0, 1.0, 0.5, -0.5], "side": ["ct", "t", "ct", "t"]})
    frame.to_parquet(candidate, index=False)
    manifest = root / "candidate.json"
    manifest.write_text(json.dumps({"source_scope_sha256": "scope-hash", "candidate_sha256": sha256(candidate), "outcome_access": "not_read", "temporal_audit": {"status": "passed"}}), encoding="utf-8")
    model = LogisticRegression(random_state=1).fit(np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.0], [-0.5, 1.0]]), np.array([0, 1, 1, 0]))
    artifact = {"model": model, "selected_c": 0.01, "design": {"numeric": ["x"], "categorical": ["side"], "means": {"x": 0.25}, "stds": {"x": 0.559016994}, "categories": {"side": ["ct", "t"]}, "references": {"side": "ct"}}}
    model_path = root / "model.joblib"; joblib.dump(artifact, model_path)
    model_manifest = root / "model.json"; model_manifest.write_text(json.dumps({"files": {"full_model.joblib": sha256(model_path)}}), encoding="utf-8")
    common = ["D1-B", "--candidate", str(candidate), "--candidate-manifest", str(manifest), "--scope-lock", str(scope), "--model", str(model_path), "--model-manifest", str(model_manifest)]
    preflight_root = root / "preflight"
    result = run(["preflight", *common, "--output-root", str(preflight_root)])
    assert result.returncode == 0, result.stderr
    assert json.loads((preflight_root / "D1-B" / "preflight.json").read_text(encoding="utf-8"))["outcome_access"] == "not_read"
    labels = root / "labels.parquet"; pd.DataFrame({"duel_id": ["a", "b", "c", "d"], "focal_wins": [0, 1, 1, 0]}).to_parquet(labels, index=False)
    denied_root = root / "denied"
    result = run(["score", *common, "--labels", str(labels), "--output-root", str(denied_root)])
    assert result.returncode != 0 and "missing_dataset_release_authorization:D1-B" in result.stderr
    assert not denied_root.exists()
    score_root = root / "score"; environment = os.environ.copy(); environment["PHAT_EXTERNAL_SCORE_AUTHORIZATION"] = "release_D1-B_once"
    result = run(["score", *common, "--labels", str(labels), "--output-root", str(score_root)], environment)
    assert result.returncode == 0, result.stderr
    report = json.loads((score_root / "D1-B" / "report.json").read_text(encoding="utf-8"))
    assert report["outcome_access"] == "released_once" and report["rows"] == 4
print("score_p_hat_external_v2 synthetic tests passed")
