"""Audit D1-A p_hat function form with a fixed solver budget and in-place sparse scaling."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "docs" / "修订与补丁" / "39_D1A既有变量函数形式充分性与模型升格分析协议_v4_2026-08-26.md"
PROTOCOL = ROOT / "docs" / "修订与补丁" / "40_D1A既有变量函数形式充分性与模型升格执行协议_v4_2026-08-26.md"
MAPPING = ROOT / "data" / "d1a_internal_validation_input_mapping_v1.json"
INPUT_LOCK = ROOT / "data" / "d1a_c_selection_input_lock_v1.json"
TRAINING = ROOT / "results" / "d1a_c_selection_training_v1"
RESULT = ROOT / "results" / "p_hat_function_form_audit_d1a_v4"
STATE = RESULT / "state"

SEED = 20260826
EPS = 1e-15
BOOTSTRAPS = 2000
BASELINE_C = 0.01
MAX_ITER = 20000
C_GRID = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
OUTER_FOLDS = (
    ("outer_1", 100, 101, 150),
    ("outer_2", 150, 151, 200),
    ("outer_3", 200, 201, 250),
    ("outer_4", 250, 251, 300),
)
SPLINE = (
    "focal_minus_opponent_health", "focal_speed", "opponent_speed", "position_distance",
    "focal_minus_opponent_post_buy_money", "focal_minus_opponent_post_buy_equipment_value",
    "focal_minus_opponent_team_post_buy_equipment_value", "absolute_vertical_position_difference",
    "relative_radial_speed", "relative_transverse_speed", "relative_angular_speed",
    "focal_facing_error_radians", "opponent_facing_error_radians",
    "focal_nearest_living_teammate_distance", "opponent_nearest_living_teammate_distance",
    "focal_nearest_living_enemy_distance", "opponent_nearest_living_enemy_distance",
    "focal_flash_remaining_seconds", "opponent_flash_remaining_seconds", "bomb_time_remaining_seconds",
)
TEAM_ALIVE = "focal_minus_opponent_team_alive"


class AuditFailure(RuntimeError):
    pass


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise AuditFailure(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise AuditFailure(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    write_json_once(path.with_suffix(path.suffix + ".sha256.json"), {"sha256": sha256(path), "rows": int(len(frame)), "columns": frame.columns.tolist()})


def verify_parquet(path: Path) -> pd.DataFrame:
    sidecar = path.with_suffix(path.suffix + ".sha256.json")
    ensure(path.is_file() and sidecar.is_file(), f"missing_state_file:{path.name}")
    record = read_json(sidecar)
    ensure(record["sha256"] == sha256(path), f"state_hash_mismatch:{path.name}")
    frame = pd.read_parquet(path)
    ensure(int(record["rows"]) == len(frame) and record["columns"] == frame.columns.tolist(), f"state_schema_mismatch:{path.name}")
    return frame


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(value, -35, 35)))


def log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    probability = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(probability) + (1 - y) * np.log(1 - probability))


class CategoryDesign:
    def __init__(self, categorical: tuple[str, ...]) -> None:
        self.categorical = categorical
        self.categories: dict[str, list[str]] = {}
        self.references: dict[str, str] = {}

    def fit(self, frame: pd.DataFrame) -> "CategoryDesign":
        for column in self.categorical:
            levels = sorted(frame[column].astype(str).unique().tolist())
            ensure(bool(levels), f"empty_category:{column}")
            self.categories[column] = levels
            self.references[column] = levels[0]
        return self

    def transform_column(self, frame: pd.DataFrame, column: str) -> sparse.csr_matrix:
        levels = self.categories[column]
        lookup = {value: index for index, value in enumerate(levels[1:])}
        raw = frame[column].astype(str).to_numpy()
        rows = np.fromiter((index for index, value in enumerate(raw) if value in lookup), dtype=np.int64)
        cols = np.fromiter((lookup[value] for value in raw if value in lookup), dtype=np.int64)
        return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(frame), len(lookup)))

    def transform_all(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        return sparse.hstack([self.transform_column(frame, column) for column in self.categorical], format="csr")


class BaselineDesign:
    """Exact numerical/category construction used by the frozen D1-A trainer."""

    def __init__(self, numeric: tuple[str, ...], categorical: tuple[str, ...]) -> None:
        self.numeric = numeric
        self.categories = CategoryDesign(categorical)
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame) -> "BaselineDesign":
        for column in self.numeric:
            value = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
            mean, std = float(value.mean()), float(value.std())
            ensure(np.isfinite(value).all() and math.isfinite(mean) and std > 0, f"invalid_numeric:{column}")
            self.means[column], self.stds[column] = mean, std
        self.categories.fit(frame)
        return self

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        numeric = np.column_stack([(pd.to_numeric(frame[column], errors="raise").to_numpy(float) - self.means[column]) / self.stds[column] for column in self.numeric])
        return sparse.hstack([sparse.csr_matrix(numeric), self.categories.transform_all(frame)], format="csr")


class FlexibleDesign:
    def __init__(self, numeric: tuple[str, ...], categorical: tuple[str, ...]) -> None:
        self.numeric = numeric
        self.categorical = categorical
        self.binary = tuple(column for column in numeric if column not in SPLINE and column != TEAM_ALIVE)
        self.categories = CategoryDesign(categorical)
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.splines: dict[str, SplineTransformer] = {}
        self.team_alive_levels: list[str] = []
        self.weapon_levels: list[str] = []
        self.weapon_reference = ""
        self.scaler: StandardScaler | None = None

    def fit(self, frame: pd.DataFrame) -> "FlexibleDesign":
        for column in self.numeric:
            value = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
            mean, std = float(value.mean()), float(value.std())
            ensure(np.isfinite(value).all() and std > 0, f"invalid_numeric:{column}")
            self.means[column], self.stds[column] = mean, std
        for column in SPLINE:
            transformer = SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="linear", include_bias=False)
            transformer.fit(pd.to_numeric(frame[column], errors="raise").to_numpy(float).reshape(-1, 1))
            self.splines[column] = transformer
        self.team_alive_levels = sorted(frame[TEAM_ALIVE].astype(str).unique().tolist())
        ensure(len(self.team_alive_levels) >= 2, "team_alive_levels_insufficient")
        self.categories.fit(frame)
        weapon_counts = frame["weapon_pair"].astype(str).value_counts()
        self.weapon_reference = self.categories.references["weapon_pair"]
        retained = sorted(value for value, count in weapon_counts.items() if count >= 200 and value != self.weapon_reference)
        rare = sorted(value for value, count in weapon_counts.items() if count < 200 and value != self.weapon_reference)
        self.weapon_levels = retained + (["__other__"] if rare else [])
        raw = self._raw_matrix(frame)
        self.scaler = StandardScaler(with_mean=False, copy=False).fit(raw)
        return self

    def _standardized_numeric(self, frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack([(pd.to_numeric(frame[column], errors="raise").to_numpy(float) - self.means[column]) / self.stds[column] for column in self.numeric])

    def _spline_block(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        blocks = []
        for column in SPLINE:
            transformed = self.splines[column].transform(pd.to_numeric(frame[column], errors="raise").to_numpy(float).reshape(-1, 1))
            blocks.append(sparse.csr_matrix(transformed))
        return sparse.hstack(blocks, format="csr")

    def _team_alive_block(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        lookup = {value: index for index, value in enumerate(self.team_alive_levels[1:])}
        raw = frame[TEAM_ALIVE].astype(str).to_numpy()
        rows = np.fromiter((index for index, value in enumerate(raw) if value in lookup), dtype=np.int64)
        cols = np.fromiter((lookup[value] for value in raw if value in lookup), dtype=np.int64)
        return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(frame), len(lookup)))

    def _weapon_distance_block(self, frame: pd.DataFrame, distance_basis: sparse.csr_matrix) -> sparse.csr_matrix:
        if not self.weapon_levels:
            return sparse.csr_matrix((len(frame), 0))
        raw = frame["weapon_pair"].astype(str).to_numpy()
        counts = {value: int((raw == value).sum()) for value in np.unique(raw)}
        retained = set(self.weapon_levels)
        labels = np.array([value if value in retained else "__other__" if "__other__" in retained and value != self.weapon_reference else "" for value in raw], dtype=object)
        blocks = [distance_basis.multiply((labels == level).astype(float).reshape(-1, 1)) for level in self.weapon_levels]
        return sparse.hstack(blocks, format="csr")

    def _map_distance_block(self, frame: pd.DataFrame, distance_basis: sparse.csr_matrix) -> sparse.csr_matrix:
        levels = self.categories.categories["map_name"][1:]
        raw = frame["map_name"].astype(str).to_numpy()
        blocks = [distance_basis.multiply((raw == level).astype(float).reshape(-1, 1)) for level in levels]
        return sparse.hstack(blocks, format="csr") if blocks else sparse.csr_matrix((len(frame), 0))

    def _raw_matrix(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        numeric = self._standardized_numeric(frame)
        interactions = np.column_stack([numeric[:, left] * numeric[:, right] for left in range(numeric.shape[1]) for right in range(left + 1, numeric.shape[1])]).astype(np.float32)
        distance_basis = sparse.csr_matrix(self.splines["position_distance"].transform(pd.to_numeric(frame["position_distance"], errors="raise").to_numpy(float).reshape(-1, 1)))
        binary = np.column_stack([pd.to_numeric(frame[column], errors="raise").to_numpy(float) for column in self.binary])
        return sparse.hstack([
            self._spline_block(frame), self._team_alive_block(frame), sparse.csr_matrix(binary), self.categories.transform_all(frame),
            sparse.csr_matrix(interactions), self._weapon_distance_block(frame, distance_basis), self._map_distance_block(frame, distance_basis),
        ], format="csr")

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        ensure(self.scaler is not None, "flexible_design_not_fitted")
        return self.scaler.transform(self._raw_matrix(frame))

    def summary(self) -> dict[str, Any]:
        ensure(self.scaler is not None, "flexible_design_not_fitted")
        return {"feature_count": int(self.scaler.n_features_in_), "spline_variables": list(SPLINE), "binary_variables": list(self.binary), "weapon_distance_levels": self.weapon_levels}


def fit_probability(design: Any, train: pd.DataFrame, valid: pd.DataFrame, c_value: float) -> tuple[np.ndarray, dict[str, Any]]:
    matrix_train = design.fit(train).transform(train)
    matrix_valid = design.transform(valid)
    model = LogisticRegression(C=c_value, penalty="l2", solver="lbfgs", max_iter=MAX_ITER, tol=1e-8, random_state=SEED)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(matrix_train, train["focal_wins"].to_numpy(int))
    ensure(not any(issubclass(item.category, ConvergenceWarning) for item in caught), f"convergence_warning:C={c_value}")
    probability = model.predict_proba(matrix_valid)[:, 1]
    ensure(np.isfinite(probability).all() and np.all((probability > 0) & (probability < 1)), "nonfinite_or_boundary_probability")
    extra = design.summary() if isinstance(design, FlexibleDesign) else {"feature_count": int(matrix_train.shape[1])}
    return probability, {"c": c_value, "train_rows": int(len(train)), "valid_rows": int(len(valid)), **extra}


def make_design(model_kind: str, numeric: tuple[str, ...], categorical: tuple[str, ...]) -> Any:
    if model_kind == "baseline":
        return BaselineDesign(numeric, categorical)
    if model_kind == "flexible":
        return FlexibleDesign(numeric, categorical)
    raise AuditFailure(f"unknown_model_kind:{model_kind}")


def per_match_metrics(frame: pd.DataFrame, p_column: str) -> pd.DataFrame:
    y = frame["focal_wins"].to_numpy(float)
    probability = frame[p_column].to_numpy(float)
    metric = pd.DataFrame({"match_id": frame["match_id"].astype(str), "loss": log_loss(y, probability), "brier": (y - probability) ** 2})
    return metric.groupby("match_id", sort=True).mean().reset_index()


def select_c(frame: pd.DataFrame, outer_end: int, model_kind: str, numeric: tuple[str, ...], categorical: tuple[str, ...], outer_name: str) -> tuple[float, pd.DataFrame]:
    first_end = outer_end // 2
    second_end = (3 * outer_end) // 4
    inner_folds = (("inner_1", 1, first_end, first_end + 1, second_end), ("inner_2", 1, second_end, second_end + 1, outer_end))
    records: list[dict[str, Any]] = []
    for c_value in C_GRID:
        for inner_name, train_start, train_end, valid_start, valid_end in inner_folds:
            state_path = STATE / f"{outer_name}_{model_kind}_c_{c_value:g}_{inner_name}.json"
            if state_path.exists():
                record = read_json(state_path)
                ensure(record["input_lock_sha256"] == sha256(RESULT / "checkpoint_01_input_lock.json"), f"inner_state_lock_mismatch:{state_path.name}")
            else:
                train = frame.loc[frame["queue_order"].between(train_start, train_end)].copy()
                valid = frame.loc[frame["queue_order"].between(valid_start, valid_end)].copy()
                ensure(not train.empty and not valid.empty and not (set(train["match_id"]) & set(valid["match_id"])), f"inner_fold_invalid:{outer_name}:{inner_name}")
                probability, summary = fit_probability(make_design(model_kind, numeric, categorical), train, valid, c_value)
                oof = valid[["match_id", "focal_wins"]].copy()
                oof["p_hat"] = probability
                metrics = per_match_metrics(oof, "p_hat")
                record = {"input_lock_sha256": sha256(RESULT / "checkpoint_01_input_lock.json"), "outer_fold": outer_name, "model": model_kind, "c": c_value, "inner_fold": inner_name, "train_queue": [train_start, train_end], "valid_queue": [valid_start, valid_end], "match_equal_log_loss": float(metrics["loss"].mean()), "match_equal_brier": float(metrics["brier"].mean()), **summary}
                write_json_once(state_path, record)
            records.append(record)
    metrics = pd.DataFrame(records)
    summary = metrics.groupby("c", as_index=False)[["match_equal_log_loss", "match_equal_brier"]].mean().sort_values(["match_equal_log_loss", "c"], kind="stable")
    best = float(summary.iloc[0]["c"])
    tied = summary[np.isclose(summary["match_equal_log_loss"], summary.iloc[0]["match_equal_log_loss"], atol=1e-10, rtol=0)]
    return float(tied["c"].min()), metrics


def input_paths() -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    mapping, input_lock, training = read_json(MAPPING), read_json(INPUT_LOCK), read_json(TRAINING / "manifest.json")
    candidate = Path(mapping["mapping"]["candidate"]["historical_read_only_path"])
    ensure(candidate.is_file(), "historical_candidate_missing")
    ensure(sha256(candidate) == mapping["mapping"]["candidate"]["sha256"] == input_lock["candidate_sha256"], "historical_candidate_hash_mismatch")
    ensure(training["selected_c"] == BASELINE_C, "unexpected_frozen_baseline_c")
    return candidate, mapping, input_lock, training


def load_frame() -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    candidate, _, input_lock, _ = input_paths()
    numeric = tuple(input_lock["feature_contract"]["numeric"])
    categorical = tuple(input_lock["feature_contract"]["categorical"])
    columns = list(dict.fromkeys(("duel_id", "match_id", "map_id", "queue_order", "partition", "focal_wins", *numeric, *categorical)))
    frame = pd.read_parquet(candidate, columns=columns)
    frame = frame.loc[frame["partition"].eq("D1-A")].copy()
    ensure(len(frame) == input_lock["rows"] == 86685, "d1a_row_count_mismatch")
    ensure(frame["duel_id"].is_unique and frame["match_id"].nunique() == 300, "d1a_key_or_match_mismatch")
    ensure(frame.groupby("match_id")["queue_order"].nunique().eq(1).all(), "match_queue_nonunique")
    ensure(set(frame["queue_order"].astype(int)) == set(range(1, 301)), "queue_scope_mismatch")
    ensure(set(frame["focal_wins"].astype(int).unique()).issubset({0, 1}), "outcome_not_binary")
    for column in numeric:
        ensure(np.isfinite(pd.to_numeric(frame[column], errors="raise").to_numpy(float)).all(), f"nonfinite_numeric:{column}")
    for column in categorical:
        values = frame[column].astype("string")
        ensure(not values.isna().any() and not values.str.strip().eq("").any(), f"invalid_category:{column}")
    return frame, numeric, categorical


def lock() -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    checkpoint = RESULT / "checkpoint_01_input_lock.json"
    candidate, mapping, input_lock, training = input_paths()
    if checkpoint.exists():
        record = read_json(checkpoint)
        ensure(record["candidate_sha256"] == sha256(candidate) and record["script_sha256"] == sha256(Path(__file__).resolve()) and record["protocol_sha256"] == sha256(PROTOCOL), "existing_input_lock_mismatch")
        return
    ensure(ANALYSIS.is_file() and PROTOCOL.is_file() and MAPPING.is_file() and INPUT_LOCK.is_file() and (TRAINING / "selected_c_oof.parquet").is_file(), "required_input_missing")
    write_json_once(checkpoint, {"stage": "p_hat_function_form_audit_d1a_v4_input_lock", "created_at_utc": datetime.now(UTC).isoformat(), "outcome_access": "D1-A_previously_authorized_only", "candidate_path": str(candidate), "candidate_sha256": sha256(candidate), "mapping_sha256": sha256(MAPPING), "training_input_lock_sha256": sha256(INPUT_LOCK), "training_manifest_sha256": sha256(TRAINING / "manifest.json"), "training_oof_sha256": sha256(TRAINING / "selected_c_oof.parquet"), "analysis_protocol_sha256": sha256(ANALYSIS), "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__).resolve()), "solver": "lbfgs", "max_iter": MAX_ITER, "tol": 1e-8, "flexible_scaler_copy": False, "rows": input_lock["rows"], "matches": input_lock["matches"], "numeric": input_lock["feature_contract"]["numeric"], "categorical": input_lock["feature_contract"]["categorical"], "mapping_stage": mapping["stage"], "training_stage": training["stage"]})


def verify_lock() -> dict[str, Any]:
    lock_path = RESULT / "checkpoint_01_input_lock.json"
    ensure(lock_path.is_file(), "input_lock_missing")
    record = read_json(lock_path)
    candidate, _, _, _ = input_paths()
    checks = {"candidate_sha256": sha256(candidate), "mapping_sha256": sha256(MAPPING), "training_input_lock_sha256": sha256(INPUT_LOCK), "training_manifest_sha256": sha256(TRAINING / "manifest.json"), "training_oof_sha256": sha256(TRAINING / "selected_c_oof.parquet"), "analysis_protocol_sha256": sha256(ANALYSIS), "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__).resolve())}
    for key, value in checks.items():
        ensure(record[key] == value, f"input_lock_mismatch:{key}")
    return record


def smoke() -> None:
    verify_lock()
    checkpoint = RESULT / "checkpoint_03_smoke.json"
    if checkpoint.exists():
        return
    frame, numeric, categorical = load_frame()
    match_order = sorted(frame["match_id"].astype(str).unique().tolist())[:30]
    subset = frame.loc[frame["match_id"].astype(str).isin(match_order)].sort_values(["queue_order", "duel_id"])
    selected = sorted(subset["queue_order"].unique().tolist())
    split = selected[20]
    train, valid = subset.loc[subset["queue_order"] < split], subset.loc[subset["queue_order"] >= split]
    ensure(not train.empty and not valid.empty and not (set(train["match_id"]) & set(valid["match_id"])), "smoke_split_invalid")
    _, base_summary = fit_probability(make_design("baseline", numeric, categorical), train, valid, BASELINE_C)
    _, flex_summary = fit_probability(make_design("flexible", numeric, categorical), train, valid, 0.001)
    write_json_once(checkpoint, {"stage": "p_hat_function_form_audit_d1a_v4_smoke", "created_at_utc": datetime.now(UTC).isoformat(), "input_lock_sha256": sha256(RESULT / "checkpoint_01_input_lock.json"), "status": "passed", "matches": 30, "train_rows": int(len(train)), "valid_rows": int(len(valid)), "baseline": base_summary, "flexible": flex_summary, "solver": "lbfgs", "max_iter": MAX_ITER, "interpretation": "No OOF comparison, interval, adequacy, or promotion decision was calculated in smoke."})


def stress() -> None:
    verify_lock()
    checkpoint = RESULT / "checkpoint_04_stress.json"
    if checkpoint.exists():
        return
    frame, numeric, categorical = load_frame()
    train = frame.loc[frame["queue_order"].between(1, 150)].copy()
    valid = frame.loc[frame["queue_order"].between(151, 200)].copy()
    ensure(not train.empty and not valid.empty and not (set(train["match_id"]) & set(valid["match_id"])), "stress_split_invalid")
    design = make_design("flexible", numeric, categorical)
    matrix_train = design.fit(train).transform(train)
    matrix_valid = design.transform(valid)
    ensure(matrix_train.shape[1] == matrix_valid.shape[1] and matrix_train.shape[1] > 0, "stress_design_shape_invalid")
    ensure(np.isfinite(matrix_train.data).all() and np.isfinite(matrix_valid.data).all(), "stress_design_nonfinite")
    write_json_once(checkpoint, {"stage": "p_hat_function_form_audit_d1a_v4_stress", "created_at_utc": datetime.now(UTC).isoformat(), "input_lock_sha256": sha256(RESULT / "checkpoint_01_input_lock.json"), "status": "passed", "train_queue": [1, 150], "valid_queue": [151, 200], "train_rows": int(len(train)), "valid_rows": int(len(valid)), "train_shape": [int(matrix_train.shape[0]), int(matrix_train.shape[1])], "valid_shape": [int(matrix_valid.shape[0]), int(matrix_valid.shape[1])], "flexible_scaler_copy": False, "interpretation": "Memory-path transformation test only; no logistic fit, probability, C selection, OOF comparison, interval, adequacy, or promotion decision was calculated."})


def outer_oof(frame: pd.DataFrame, numeric: tuple[str, ...], categorical: tuple[str, ...], outer_name: str, outer_end: int, valid_start: int, valid_end: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_path = STATE / f"{outer_name}_selection.json"
    oof_path = STATE / f"{outer_name}_oof.parquet"
    reproduction_path = STATE / f"{outer_name}_baseline_reproduction.parquet"
    if selection_path.exists() and oof_path.exists() and reproduction_path.exists():
        selection = read_json(selection_path)
        ensure(selection["input_lock_sha256"] == sha256(RESULT / "checkpoint_01_input_lock.json"), f"selection_lock_mismatch:{outer_name}")
        return verify_parquet(oof_path), verify_parquet(reproduction_path)
    train = frame.loc[frame["queue_order"].between(1, outer_end)].copy()
    valid = frame.loc[frame["queue_order"].between(valid_start, valid_end)].copy()
    ensure(not train.empty and not valid.empty and not (set(train["match_id"]) & set(valid["match_id"])), f"outer_fold_invalid:{outer_name}")
    base_c, base_inner = select_c(frame, outer_end, "baseline", numeric, categorical, outer_name)
    flex_c, flex_inner = select_c(frame, outer_end, "flexible", numeric, categorical, outer_name)
    base_p, base_summary = fit_probability(make_design("baseline", numeric, categorical), train, valid, base_c)
    flex_p, flex_summary = fit_probability(make_design("flexible", numeric, categorical), train, valid, flex_c)
    reproduction_p, reproduction_summary = fit_probability(make_design("baseline", numeric, categorical), train, valid, BASELINE_C)
    oof = valid[["duel_id", "match_id", "map_id", "queue_order", "focal_wins"]].copy()
    oof["outer_fold"] = outer_name
    oof["p_hat_baseline"] = base_p
    oof["p_hat_flexible"] = flex_p
    oof["baseline_c"] = base_c
    oof["flexible_c"] = flex_c
    reproduction = valid[["duel_id", "match_id", "map_id", "queue_order", "focal_wins"]].copy()
    reproduction["outer_fold"] = outer_name
    reproduction["p_hat"] = reproduction_p
    if not selection_path.exists():
        write_json_once(selection_path, {"input_lock_sha256": sha256(RESULT / "checkpoint_01_input_lock.json"), "outer_fold": outer_name, "outer_train_queue": [1, outer_end], "outer_valid_queue": [valid_start, valid_end], "baseline_selected_c": base_c, "flexible_selected_c": flex_c, "baseline_inner": base_inner.to_dict(orient="records"), "flexible_inner": flex_inner.to_dict(orient="records"), "baseline_outer_summary": base_summary, "flexible_outer_summary": flex_summary, "baseline_reproduction_summary": reproduction_summary})
    if not oof_path.exists():
        write_parquet_once(oof_path, oof)
    if not reproduction_path.exists():
        write_parquet_once(reproduction_path, reproduction)
    return oof, reproduction


def calibration(probability: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    x = np.log(np.clip(probability, EPS, 1 - EPS) / np.clip(1 - probability, EPS, 1 - EPS))
    beta = np.zeros(2)
    for _ in range(100):
        p = sigmoid(beta[0] + beta[1] * x)
        w = np.maximum(p * (1 - p), 1e-9)
        hessian = np.array(((w.sum(), np.dot(w, x)), (np.dot(w, x), np.dot(w, x * x))))
        step = np.linalg.solve(hessian, np.array(((outcome - p).sum(), np.dot(outcome - p, x))))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            return {"intercept": float(beta[0]), "slope": float(beta[1])}
    raise AuditFailure("calibration_nonconvergence")


def reliability(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    probability = frame[column].to_numpy(float)
    groups = pd.qcut(pd.Series(probability), q=10, duplicates="drop")
    return pd.DataFrame({"bin": groups, "p_hat": probability, "focal_wins": frame["focal_wins"].to_numpy(float)}).groupby("bin", observed=True).agg(p_hat=("p_hat", "mean"), focal_wins=("focal_wins", "mean"), rows=("focal_wins", "size")).reset_index(drop=True)


def full() -> dict[str, Any]:
    lock = verify_lock()
    ensure((RESULT / "checkpoint_03_smoke.json").is_file(), "smoke_missing")
    ensure((RESULT / "checkpoint_04_stress.json").is_file(), "stress_missing")
    smoke_record = read_json(RESULT / "checkpoint_03_smoke.json")
    ensure(smoke_record["status"] == "passed" and smoke_record["input_lock_sha256"] == sha256(RESULT / "checkpoint_01_input_lock.json"), "smoke_invalid")
    frame, numeric, categorical = load_frame()
    outer, reproduction = zip(*(outer_oof(frame, numeric, categorical, *fold) for fold in OUTER_FOLDS))
    outer_frame = pd.concat(outer, ignore_index=True).sort_values("duel_id").reset_index(drop=True)
    reproduction_frame = pd.concat(reproduction, ignore_index=True).sort_values("duel_id").reset_index(drop=True)
    frozen = pd.read_parquet(TRAINING / "selected_c_oof.parquet").sort_values("duel_id").reset_index(drop=True)
    merged = reproduction_frame.merge(frozen[["duel_id", "p_hat"]], on="duel_id", how="inner", suffixes=("_rebuilt", "_frozen"), validate="one_to_one")
    ensure(len(merged) == len(reproduction_frame) == len(frozen), "baseline_reproduction_coverage_failure")
    rebuilt_loss = per_match_metrics(reproduction_frame, "p_hat")["loss"].mean()
    frozen_loss = per_match_metrics(frozen, "p_hat")["loss"].mean()
    ensure(abs(float(rebuilt_loss) - 0.6471161295553121) <= 1e-10 and abs(float(rebuilt_loss) - float(frozen_loss)) <= 1e-10, "baseline_reproduction_metric_failure")
    y = outer_frame["focal_wins"].to_numpy(float)
    outer_frame["loss_baseline"] = log_loss(y, outer_frame["p_hat_baseline"].to_numpy(float))
    outer_frame["loss_flexible"] = log_loss(y, outer_frame["p_hat_flexible"].to_numpy(float))
    outer_frame["brier_baseline"] = (y - outer_frame["p_hat_baseline"].to_numpy(float)) ** 2
    outer_frame["brier_flexible"] = (y - outer_frame["p_hat_flexible"].to_numpy(float)) ** 2
    by_match = outer_frame.groupby(["outer_fold", "match_id"], sort=True)[["loss_baseline", "loss_flexible", "brier_baseline", "brier_flexible"]].mean().reset_index()
    by_match["d_log_loss"] = by_match["loss_baseline"] - by_match["loss_flexible"]
    fold_metrics = by_match.groupby("outer_fold", sort=True).mean(numeric_only=True).reset_index()
    point_d = float(fold_metrics["d_log_loss"].mean())
    rng = np.random.default_rng(SEED)
    boot_records: list[dict[str, float | int]] = []
    fold_arrays = [group["d_log_loss"].to_numpy(float) for _, group in by_match.groupby("outer_fold", sort=True)]
    for replicate in range(1, BOOTSTRAPS + 1):
        means = [float(values[rng.integers(0, len(values), len(values))].mean()) for values in fold_arrays]
        boot_records.append({"replicate": replicate, "d_log_loss": float(np.mean(means))})
    boot = pd.DataFrame(boot_records)
    lower, upper = float(boot["d_log_loss"].quantile(0.025)), float(boot["d_log_loss"].quantile(0.975))
    positive_folds = int((fold_metrics["d_log_loss"] > 0).sum())
    if upper < 0.001:
        decision = "baseline_function_form_adequate"
    elif lower >= 0.001 and positive_folds >= 3:
        decision = "flexible_model_upgrade_eligible"
    else:
        decision = "indeterminate"
    comparison = {"input_lock_sha256": sha256(RESULT / "checkpoint_01_input_lock.json"), "baseline_reproduction": {"rebuilt_match_equal_log_loss": float(rebuilt_loss), "frozen_match_equal_log_loss": float(frozen_loss), "max_abs_probability_difference": float(np.max(np.abs(merged["p_hat_rebuilt"] - merged["p_hat_frozen"])))}, "outer_rows": int(len(outer_frame)), "outer_matches": int(outer_frame["match_id"].nunique()), "d_log_loss": point_d, "d_log_loss_ci95": [lower, upper], "positive_outer_folds": positive_folds, "fold_metrics": fold_metrics.to_dict(orient="records"), "decision": decision, "baseline_calibration": calibration(outer_frame["p_hat_baseline"].to_numpy(float), y), "flexible_calibration": calibration(outer_frame["p_hat_flexible"].to_numpy(float), y), "baseline_match_equal_brier": float(fold_metrics["brier_baseline"].mean()), "flexible_match_equal_brier": float(fold_metrics["brier_flexible"].mean()), "gates": {"G0": "pass", "G1": "pass", "G2": "pass", "G3": "pass", "G4": decision, "G5": "not_authorized"}}
    comparison_path = RESULT / "checkpoint_04_comparison.json"
    if not comparison_path.exists():
        write_json_once(comparison_path, comparison)
    if not (RESULT / "outer_oof.parquet").exists():
        write_parquet_once(RESULT / "outer_oof.parquet", outer_frame)
    if not (RESULT / "match_level_comparison.parquet").exists():
        write_parquet_once(RESULT / "match_level_comparison.parquet", by_match)
    if not (RESULT / "bootstrap_comparison.parquet").exists():
        write_parquet_once(RESULT / "bootstrap_comparison.parquet", boot)
    if not (RESULT / "reliability_baseline.parquet").exists():
        write_parquet_once(RESULT / "reliability_baseline.parquet", reliability(outer_frame, "p_hat_baseline"))
    if not (RESULT / "reliability_flexible.parquet").exists():
        write_parquet_once(RESULT / "reliability_flexible.parquet", reliability(outer_frame, "p_hat_flexible"))
    manifest_path = RESULT / "manifest.json"
    if not manifest_path.exists():
        files = {path.name: sha256(path) for path in RESULT.iterdir() if path.is_file()}
        write_json_once(manifest_path, {"stage": "p_hat_function_form_audit_d1a_v4", "created_at_utc": datetime.now(UTC).isoformat(), "status": "passed", "outcome_access": "D1-A_previously_authorized_only", "input_lock_sha256": sha256(RESULT / "checkpoint_01_input_lock.json"), "comparison_sha256": sha256(comparison_path), "output_hashes": files, "random_seed": SEED, "bootstrap_replicates": BOOTSTRAPS, "solver": "lbfgs", "max_iter": MAX_ITER, "tol": 1e-8, "flexible_scaler_copy": False, "python": sys.version, "platform": platform.platform(), "decision": decision, "interpretation": "D1-A internal nonconfirmatory function-form audit only; it neither changes the frozen p_hat nor re-estimates the E1-v2 hot-hand result."})
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("lock", "smoke", "stress", "full"))
    args = parser.parse_args()
    try:
        if args.phase == "lock":
            lock(); result = {"status": "locked"}
        elif args.phase == "smoke":
            smoke(); result = {"status": "smoke_passed"}
        elif args.phase == "stress":
            stress(); result = {"status": "stress_passed"}
        else:
            result = full()
    except Exception as error:
        RESULT.mkdir(parents=True, exist_ok=True)
        failure = RESULT / f"failure_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
        write_json_once(failure, {"stage": "p_hat_function_form_audit_d1a_v4", "created_at_utc": datetime.now(UTC).isoformat(), "status": "failed", "error": f"{type(error).__name__}:{error}"})
        raise
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
