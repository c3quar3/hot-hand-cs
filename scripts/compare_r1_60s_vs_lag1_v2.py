"""Repair R1 M3-versus-M1 comparison using the complete residual ledger."""
from __future__ import annotations
import hashlib, json, math, os
from datetime import UTC, datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=Path(r"D:\CS2_60s_confirmation_r1_70_preoutcome_v1\analysis_60s_v1")
STATE, ASSET=BASE/'state_matrix.parquet', BASE/'residual_asset.parquet'
FIT=ROOT/'results/history_residual_state_oof_d1_v2/state/F1/training_fit.json'
PROTOCOL=ROOT/'docs/协议/23_R1_60秒相对lag-1直接比较修复执行锁_v2_2026-08-27.md'
OUT=ROOT/'results/r1_60s_vs_lag1_head_to_head_v2'; SEED=20260827; BOOTSTRAPS=2000

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    os.replace(temp,path)

def reconstruct_h(asset):
    focal=asset[['physical_event_id','match_id','map_id','event_tick','focal_player_id','e_t']].rename(columns={'focal_player_id':'player_id'})
    opponent=asset[['physical_event_id','match_id','map_id','event_tick','opponent_player_id','e_t']].rename(columns={'opponent_player_id':'player_id'}).copy()
    opponent['e_t']=-opponent['e_t']
    ledger=pd.concat([focal,opponent],ignore_index=True)
    ledger['player_id']=ledger['player_id'].astype(str)
    ledger=ledger.sort_values(['match_id','map_id','player_id','event_tick','physical_event_id'],kind='stable')
    rows=[]
    for _, group in ledger.groupby(['match_id','map_id','player_id'],sort=False):
        prior=None
        for _, batch in group.groupby('event_tick',sort=False):
            h=np.nan if prior is None or len(prior)!=1 else float(prior.e_t.iloc[0])
            rows.extend((str(r.physical_event_id),str(r.match_id),str(r.map_id),str(r.player_id),float(r.event_tick),h) for r in batch.itertuples(index=False))
            prior=batch
    return pd.DataFrame(rows,columns=['physical_event_id','match_id','map_id','player_id','event_tick','H_t'])

def main():
    if (OUT/'manifest.json').is_file(): print('completed_verified'); return
    if OUT.exists() and any(OUT.iterdir()): raise RuntimeError('incomplete_output')
    if not all(p.is_file() for p in (STATE,ASSET,FIT,PROTOCOL)): raise RuntimeError('input_missing')
    OUT.mkdir(parents=True)
    lock={'stage':'r1_60s_vs_lag1_head_to_head_v2','created_at_utc':datetime.now(UTC).isoformat(),'state_sha256':sha(STATE),'asset_sha256':sha(ASSET),'fit_sha256':sha(FIT),'protocol_sha256':sha(PROTOCOL),'script_sha256':sha(Path(__file__).resolve()),'seed':SEED,'draws':BOOTSTRAPS}
    write(OUT/'input_lock.json',lock)
    state=pd.read_parquet(STATE)
    frame=state.merge(reconstruct_h(pd.read_parquet(ASSET)),on=['physical_event_id','match_id','map_id','player_id','event_tick'],how='left',validate='one_to_one').dropna(subset=['H_t']).copy()
    frame['S_t']=frame.S_h60
    fit=json.loads(FIT.read_text(encoding='utf-8'))['models']
    for model in ('M1_lag1','M3_decay_sum'):
        spec=fit[model]; z=np.log2(1+frame.d_tick.to_numpy(float))-spec['mu_train']; state_value=frame[spec['state']].to_numpy(float)/spec['state_scale_train']
        frame[model]=np.column_stack([np.ones(len(frame)),z,state_value,state_value*z])@np.asarray(spec['coefficients'],float)
    if not np.isfinite(frame[['e_t','M1_lag1','M3_decay_sum']].to_numpy(float)).all(): raise RuntimeError('nonfinite_prediction')
    matches=sorted(frame.match_id.astype(str).unique())
    if len(matches)!=70: raise RuntimeError('unexpected_match_count')
    frame['fold']=np.where(frame.match_id.astype(str).isin(set(matches[:35])),'F1','F2')
    for model in ('M1_lag1','M3_decay_sum'): frame[model]=(frame.e_t-frame[model])**2
    losses=frame.groupby(['fold','match_id'],as_index=False)[['M1_lag1','M3_decay_sum']].mean(); losses.to_parquet(OUT/'match_losses.parquet',index=False)
    def gain(value):
        mean=value.groupby('fold')[['M1_lag1','M3_decay_sum']].mean()
        return float((mean.M1_lag1.mean()-mean.M3_decay_sum.mean())/mean.M1_lag1.mean())
    estimate=gain(losses); rng=np.random.default_rng(SEED); draws=[]
    for _ in range(BOOTSTRAPS):
        sample=pd.concat([part.iloc[rng.integers(len(part),size=len(part))] for _,part in losses.groupby('fold',sort=True)],ignore_index=True)
        draws.append(gain(sample))
    draws=np.asarray(draws); np.save(OUT/'bootstrap_gains.npy',draws)
    low,high=np.quantile(draws,[.025,.975]); sd=float(draws.std(ddof=1)); p=math.erfc(abs(estimate/sd)/math.sqrt(2)); fold_gain={name:gain(part) for name,part in losses.groupby('fold',sort=True)}
    audit={'source_rows':int(len(state)),'shared_rows':int(len(frame)),'excluded_ambiguous_prior_tick':int(len(state)-len(frame)),'matches':70,'fold_matches':{name:int(part.match_id.nunique()) for name,part in losses.groupby('fold')},'fold_gain':fold_gain,'relative_gain':estimate,'ci95_low':float(low),'ci95_high':float(high),'ci95_half_width':float((high-low)/2),'bootstrap_sd':sd,'p_value_two_sided':p}
    write(OUT/'comparison.json',audit)
    passed=all(value>0 for value in fold_gain.values()) and estimate>0 and low>0 and p<.05 and (high-low)/2<=.002
    gates={'H1_shared_rows':True,'H2_two_35_match_blocks':True,'H3_precision':bool((high-low)/2<=.002),'H4_positive_superiority':bool(passed)}
    write(OUT/'gates.json',gates)
    hashes={path.name:sha(path) for path in (OUT/'input_lock.json',OUT/'match_losses.parquet',OUT/'bootstrap_gains.npy',OUT/'comparison.json',OUT/'gates.json')}
    write(OUT/'file_sha256.json',hashes)
    write(OUT/'manifest.json',{'stage':lock['stage'],'status':'passed' if passed else 'completed_no_positive_superiority_confirmation','classification':'confirmatory_by_user_final_adjudication; R1_labels_previously_released_no_design_adjustment; v2_repairs_lag1_reconstruction','input_lock_sha256':sha(OUT/'input_lock.json'),'file_sha256_sha256':sha(OUT/'file_sha256.json'),'gates':gates})
    print('passed' if passed else 'completed_no_positive_superiority_confirmation')

if __name__=='__main__': main()
