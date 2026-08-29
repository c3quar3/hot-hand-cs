"""Synthetic test for the locked one-time external p_hat scorer."""

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


SCRIPT = Path(__file__).with_name("score_p_hat_external_v3.py")


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
    candidate = root / "candidate.parquet"
    frame = pd.DataFrame({
        "duel_id": ["a", "b", "c", "d"], "match_id": ["m1", "m1", "m2", "m2"], "map_id": ["x", "x", "y", "y"],
        "anchor_tick": [1, 2, 3, 4], "event_tick": [2, 3, 4, 5], "x": [0.0, 1.0, 0.5, -0.5], "side": ["ct", "t", "ct", "t"],
    })
    frame.to_parquet(candidate, index=False)
    candidate_manifest = root / "candidate.json"
    candidate_manifest.write_text(json.dumps({"source_scope_sha256": "scope-hash", "candidate_sha256": sha256(candidate), "outcome_access": "not_read", "temporal_audit": {"status": "passed"}}), encoding="utf-8")
    model = LogisticRegression(random_state=1).fit(np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.0], [-0.5, 1.0]]), np.array([0, 1, 1, 0]))
    artifact = {"model": model, "selected_c": 0.01, "design": {"numeric": ["x"], "categorical": ["side"], "means": {"x": 0.25}, "stds": {"x": 0.559016994}, "categories": {"side": ["ct", "t"]}, "references": {"side": "ct"}}}
    model_path = root / "model.joblib"; joblib.dump(artifact, model_path)
    model_manifest = root / "model.json"; model_manifest.write_text(json.dumps({"files": {"full_model.joblib": sha256(model_path)}}), encoding="utf-8")
    labels = root / "labels.parquet"; pd.DataFrame({"duel_id": ["a", "b", "c", "d"], "focal_wins": [0, 1, 1, 0]}).to_parquet(labels, index=False)
    label_manifest = root / "labels.json"; label_manifest.write_text(json.dumps({"labels_sha256": sha256(labels), "candidate_sha256": sha256(candidate)}), encoding="utf-8")
    lock = root / "execution_lock.json"
    lock.write_text(json.dumps({"dataset": "D1-B", "outcome_access_before_run": "not_read", "candidate_sha256": sha256(candidate), "candidate_manifest_sha256": sha256(candidate_manifest), "model_sha256": sha256(model_path), "model_manifest_sha256": sha256(model_manifest), "source_scope_sha256": "scope-hash", "scripts": {"score_p_hat_external_v3.py": sha256(SCRIPT)}}), encoding="utf-8")
    common = ["D1-B", "--candidate", str(candidate), "--candidate-manifest", str(candidate_manifest), "--labels", str(labels), "--label-manifest", str(label_manifest), "--model", str(model_path), "--model-manifest", str(model_manifest), "--execution-lock", str(lock)]
    denied = root / "denied"
    result = run([*common, "--output-root", str(denied)])
    assert result.returncode != 0 and "missing_dataset_release_authorization:D1-B" in result.stderr
    assert not denied.exists()
    output = root / "score"
    environment = os.environ.copy(); environment["PHAT_EXTERNAL_SCORE_AUTHORIZATION"] = "release_D1-B_once"
    result = run([*common, "--output-root", str(output)], environment)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "D1-B" / "report.json").read_text(encoding="utf-8"))
    assert report["rows"] == 4 and report["bootstrap"]["iterations"] == 2000
    assert len(pd.read_parquet(output / "D1-B" / "bootstrap_draws.parquet")) == 2000
print("score_p_hat_external_v3 synthetic tests passed")
