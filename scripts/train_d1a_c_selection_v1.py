"""Lock, select C by forward CV, and train the restart-track D1-A probability model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


REPO_ROOT = Path(__file__).resolve().parents[3]
TRACK = REPO_ROOT / "main" / "restart_20260825"
PROTOCOL = TRACK / "docs" / "02_D1A重新训练与C选择协议_v1_2026-08-25.md"
CANDIDATE = REPO_ROOT / "main" / "results" / "p_hat_final_v8_position_continuity" / "candidate.parquet"
CANDIDATE_AUDIT = REPO_ROOT / "main" / "results" / "p_hat_final_v8_position_continuity" / "candidate_audit.json"
SOURCE_LOCK = REPO_ROOT / "main" / "data" / "p_hat_final_v8_position_continuity_source_manifest.json"
INPUT_LOCK = TRACK / "data" / "d1a_c_selection_input_lock_v1.json"
RESULT = TRACK / "results" / "d1a_c_selection_training_v1"
STAGING = RESULT / "staging"
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
FOLDS = (("fold_1", 1, 100, 101, 150), ("fold_2", 1, 150, 151, 200), ("fold_3", 1, 200, 201, 250), ("fold_4", 1, 250, 251, 300))
SEED, MAX_ITER, TOL, EPS = 20260825, 5000, 1e-8, 1e-15
NUMERIC = (
    "focal_minus_opponent_health", "focal_minus_opponent_team_alive", "focal_speed", "opponent_speed", "position_distance",
    "focal_has_armor", "opponent_has_armor", "focal_has_helmet", "opponent_has_helmet", "focal_has_defuser", "opponent_has_defuser",
    "focal_minus_opponent_post_buy_money", "focal_minus_opponent_post_buy_equipment_value", "focal_minus_opponent_team_post_buy_equipment_value",
    "absolute_vertical_position_difference", "relative_radial_speed", "relative_transverse_speed", "relative_angular_speed", "focal_facing_error_radians", "opponent_facing_error_radians", "focal_nearest_living_teammate_distance", "opponent_nearest_living_teammate_distance", "focal_nearest_living_enemy_distance", "opponent_nearest_living_enemy_distance", "focal_is_scoped", "opponent_is_scoped", "focal_is_alone", "opponent_is_alone",
    "focal_is_flashed", "opponent_is_flashed", "focal_flash_remaining_seconds", "opponent_flash_remaining_seconds", "bomb_planted", "bomb_time_remaining_seconds", "focal_is_defusing", "opponent_is_defusing",
)
CATEGORICAL = ("focal_side", "map_name", "focal_speed_source", "opponent_speed_source", "focal_weapon", "opponent_weapon", "weapon_pair", "focal_steamid", "opponent_steamid")
META = ("duel_id", "match_id", "map_id", "queue_order", "partition", "focal_wins")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def new_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


class Design:
    def __init__(self) -> None:
        self.means: dict[str, float] = {}; self.stds: dict[str, float] = {}; self.categories: dict[str, list[str]] = {}; self.references: dict[str, str] = {}

    def fit(self, frame: pd.DataFrame) -> "Design":
        for col in NUMERIC:
            x = pd.to_numeric(frame[col], errors="raise").to_numpy(float); mean, std = float(x.mean()), float(x.std())
            if not np.isfinite(x).all() or not math.isfinite(mean) or std == 0:
                raise RuntimeError(f"invalid_numeric:{col}")
            self.means[col], self.stds[col] = mean, std
        for col in CATEGORICAL:
            levels = sorted(frame[col].astype(str).unique().tolist())
            if not levels:
                raise RuntimeError(f"empty_category:{col}")
            self.categories[col], self.references[col] = levels, levels[0]
        return self

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        numeric = np.column_stack([(pd.to_numeric(frame[c], errors="raise").to_numpy(float) - self.means[c]) / self.stds[c] for c in NUMERIC])
        blocks: list[sparse.csr_matrix] = [sparse.csr_matrix(numeric)]
        for col in CATEGORICAL:
            lookup = {v: i for i, v in enumerate(self.categories[col][1:])}; raw = frame[col].astype(str).to_numpy()
            rows = np.fromiter((i for i, v in enumerate(raw) if v in lookup), dtype=np.int64)
            cols = np.fromiter((lookup[v] for v in raw if v in lookup), dtype=np.int64)
            blocks.append(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(frame), len(lookup))))
        return sparse.hstack(blocks, format="csr")

    def as_dict(self) -> dict[str, Any]:
        return {"numeric": list(NUMERIC), "categorical": list(CATEGORICAL), "means": self.means, "stds": self.stds, "categories": self.categories, "references": self.references}


def load() -> pd.DataFrame:
    cols = list(dict.fromkeys((*META, *NUMERIC, *CATEGORICAL)))
    frame = pd.read_parquet(CANDIDATE, columns=cols).query("partition == 'D1-A'").copy()
    if frame.empty or frame.duel_id.duplicated().any(): raise RuntimeError("invalid_d1a_scope")
    q = pd.to_numeric(frame.queue_order, errors="raise")
    if q.min() != 1 or q.max() != 300 or frame.groupby("match_id").queue_order.nunique().gt(1).any(): raise RuntimeError("queue_integrity_failure")
    if not set(pd.to_numeric(frame.focal_wins, errors="raise").unique()).issubset({0,1}): raise RuntimeError("outcome_not_binary")
    for col in NUMERIC:
        if not np.isfinite(pd.to_numeric(frame[col], errors="raise").to_numpy(float)).all(): raise RuntimeError(f"nonfinite:{col}")
    for col in CATEGORICAL:
        x = frame[col].astype("string")
        if x.isna().any() or x.str.strip().eq("").any(): raise RuntimeError(f"missing_category:{col}")
    return frame


def lock() -> None:
    if INPUT_LOCK.exists(): raise RuntimeError("input_lock_exists")
    for p in (PROTOCOL, CANDIDATE, CANDIDATE_AUDIT, SOURCE_LOCK):
        if not p.is_file(): raise RuntimeError(f"missing_input:{p}")
    frame = load()
    new_json(INPUT_LOCK, {"stage":"restart_d1a_c_selection_input_lock_v1","created_at_utc":datetime.now(UTC).isoformat(),"protocol_path":str(PROTOCOL),"protocol_sha256":sha(PROTOCOL),"script_path":str(Path(__file__).resolve()),"script_sha256":sha(Path(__file__).resolve()),"candidate_path":str(CANDIDATE),"candidate_sha256":sha(CANDIDATE),"candidate_audit_sha256":sha(CANDIDATE_AUDIT),"source_lock_sha256":sha(SOURCE_LOCK),"rows":len(frame),"matches":int(frame.match_id.nunique()),"queue_gap":[129],"feature_contract":{"numeric":list(NUMERIC),"categorical":list(CATEGORICAL)},"c_grid":list(C_GRID),"folds":[list(x) for x in FOLDS]})


def verify_lock() -> dict[str, Any]:
    lock = json.loads(INPUT_LOCK.read_text(encoding="utf-8"))
    for key, path in (("protocol_sha256", PROTOCOL), ("script_sha256", Path(__file__).resolve()), ("candidate_sha256", CANDIDATE), ("candidate_audit_sha256", CANDIDATE_AUDIT), ("source_lock_sha256", SOURCE_LOCK)):
        if lock.get(key) != sha(path): raise RuntimeError(f"lock_mismatch:{key}")
    return lock


def metrics(frame: pd.DataFrame) -> tuple[float,float]:
    p = np.clip(frame.p_hat.to_numpy(float), EPS, 1-EPS); y=frame.focal_wins.to_numpy(int)
    loss=-(y*np.log(p)+(1-y)*np.log(1-p)); brier=(y-p)**2
    x=pd.DataFrame({"match_id":frame.match_id,"loss":loss,"brier":brier}).groupby("match_id").mean()
    return float(x.loss.mean()),float(x.brier.mean())


def fit_predict(train: pd.DataFrame, valid: pd.DataFrame, c: float) -> tuple[np.ndarray, Design, LogisticRegression]:
    design=Design().fit(train); model=LogisticRegression(C=c, penalty="l2",solver="lbfgs",max_iter=MAX_ITER,tol=TOL,random_state=SEED)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning); model.fit(design.transform(train),train.focal_wins.to_numpy(int))
    if any(issubclass(x.category,ConvergenceWarning) for x in caught): raise RuntimeError(f"convergence_warning:C={c}")
    return model.predict_proba(design.transform(valid))[:,1], design, model


def train() -> None:
    lock=verify_lock(); frame=load(); RESULT.mkdir(parents=True,exist_ok=True); STAGING.mkdir(exist_ok=True)
    grid=[]
    for c in C_GRID:
        for name,ts,te,vs,ve in FOLDS:
            path=STAGING/f"c_{c:g}_{name}.parquet"; train=frame[frame.queue_order.between(ts,te)]; valid=frame[frame.queue_order.between(vs,ve)]
            if set(train.match_id)&set(valid.match_id) or train.empty or valid.empty: raise RuntimeError(f"fold_failure:{name}")
            if path.exists(): oof=pd.read_parquet(path)
            else:
                p,_,_=fit_predict(train,valid,c); oof=valid[["duel_id","match_id","map_id","queue_order","focal_wins"]].copy(); oof["p_hat"]=p; oof["c"]=c; oof["fold"]=name; oof.to_parquet(path,index=False,compression="zstd")
            ll,brier=metrics(oof); grid.append({"c":c,"fold":name,"rows":len(oof),"matches":int(oof.match_id.nunique()),"match_equal_log_loss":ll,"match_equal_brier":brier,"checkpoint_sha256":sha(path)})
    grid_frame=pd.DataFrame(grid); summary=grid_frame.groupby("c",as_index=False)[["match_equal_log_loss","match_equal_brier"]].mean().sort_values(["match_equal_log_loss","c"],kind="stable")
    best=float(summary.iloc[0].c)
    tied=summary[np.isclose(summary.match_equal_log_loss,summary.iloc[0].match_equal_log_loss,atol=1e-10,rtol=0)]
    best=float(tied.c.min())
    if (RESULT/"manifest.json").exists(): raise RuntimeError("final_output_exists")
    oof=pd.concat([pd.read_parquet(STAGING/f"c_{best:g}_{name}.parquet") for name,*_ in FOLDS],ignore_index=True)
    design=Design().fit(frame); model=LogisticRegression(C=best,penalty="l2",solver="lbfgs",max_iter=MAX_ITER,tol=TOL,random_state=SEED); model.fit(design.transform(frame),frame.focal_wins.to_numpy(int))
    beta=model.coef_.ravel(); rows=[{"model_column":"intercept","source_column":"intercept","category":None,"coefficient":float(model.intercept_[0]),"is_reference":False}]; i=0
    for col in NUMERIC: rows.append({"model_column":f"num_{col}","source_column":col,"category":None,"coefficient":float(beta[i]),"is_reference":False}); i+=1
    for col in CATEGORICAL:
        for cat in design.categories[col][1:]: rows.append({"model_column":f"cat_{col}={cat}","source_column":col,"category":cat,"coefficient":float(beta[i]),"is_reference":False}); i+=1
        rows.append({"model_column":f"cat_{col}={design.references[col]}","source_column":col,"category":design.references[col],"coefficient":0.0,"is_reference":True})
    coeff=pd.DataFrame(rows); grid_frame.to_parquet(RESULT/"c_grid_fold_metrics.parquet",index=False,compression="zstd"); summary.to_parquet(RESULT/"c_grid_summary.parquet",index=False,compression="zstd"); oof.to_parquet(RESULT/"selected_c_oof.parquet",index=False,compression="zstd"); coeff.to_parquet(RESULT/"coefficients_all.parquet",index=False,compression="zstd"); coeff[coeff.source_column.isin(["focal_steamid","opponent_steamid"])].to_parquet(RESULT/"coefficients_players_real_id.parquet",index=False,compression="zstd"); coeff[coeff.source_column.eq("map_name")].to_parquet(RESULT/"coefficients_maps.parquet",index=False,compression="zstd"); coeff[coeff.source_column.eq("weapon_pair")].to_parquet(RESULT/"coefficients_weapon_pairs.parquet",index=False,compression="zstd"); joblib.dump({"model":model,"design":design.as_dict(),"selected_c":best},RESULT/"full_model.joblib")
    files={p.name:sha(p) for p in RESULT.iterdir() if p.is_file()}; new_json(RESULT/"manifest.json",{"stage":"restart_d1a_c_selection_training_v1","created_at_utc":datetime.now(UTC).isoformat(),"input_lock_sha256":sha(INPUT_LOCK),"script_sha256":sha(Path(__file__).resolve()),"selected_c":best,"selection_rule":"min_mean_match_equal_log_loss; ties_within_1e-10_choose_smaller_C","grid_summary":summary.to_dict(orient="records"),"selected_oof_metrics":dict(zip(["match_equal_log_loss","match_equal_brier"],metrics(oof))),"files":files})
    print(json.dumps({"selected_c":best,"summary":summary.to_dict(orient="records")},ensure_ascii=True,indent=2))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("action",choices=("lock","train")); a=p.parse_args()
    if a.action=="lock": lock()
    else: train()

if __name__=="__main__": main()
