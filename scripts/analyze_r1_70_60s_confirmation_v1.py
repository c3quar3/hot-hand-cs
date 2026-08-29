"""Locked R1 70-map 60-second residual-state confirmation analysis."""
from __future__ import annotations
import argparse, json, math, os
from datetime import UTC,datetime
from pathlib import Path
import numpy as np,pandas as pd
import analyze_exponential_history_state_sequence_v1 as base

ROOT=Path(r"D:\CS2_60s_confirmation_r1_70_preoutcome_v1"); SCORES=ROOT/'p_hat_preoutcome_v1'; LABELS=ROOT/'label_release_v1'; OUT=ROOT/'analysis_60s_v1'
REF={'mu_z':11.538069409706317,'mu_S':0.34010149087253094,'sigma_S_ddof1':0.6490311893067302}
def sha(p): return base.digest(p)
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('smoke','full'),required=True);ap.add_argument('--out',type=Path,default=OUT);a=ap.parse_args()
 if a.out.exists() and (a.out/'manifest.json').exists(): raise RuntimeError('immutable_completed_output')
 for p in (SCORES/'p_hat_preoutcome.parquet',SCORES/'manifest.json',LABELS/'labels.parquet',LABELS/'manifest.json'): 
  if not p.is_file(): raise RuntimeError('locked_input_missing')
 a.out.mkdir(parents=True,exist_ok=True); lock={'stage':'r1_70_60s_confirmation_v1','created_at_utc':datetime.now(UTC).isoformat(),'scores_manifest_sha256':sha(SCORES/'manifest.json'),'labels_manifest_sha256':sha(LABELS/'manifest.json'),'script_sha256':sha(Path(__file__).resolve()),'reference':REF,'half_life_seconds':60.0,'tick_rate':64.0,'model':'e~1+z+S_star+S_star:z; OLS two-way CRV1 match/player'}
 write(a.out/'input_lock.json',lock)
 x=pd.read_parquet(SCORES/'p_hat_preoutcome.parquet').merge(pd.read_parquet(LABELS/'labels.parquet'),on='duel_id',validate='one_to_one')
 x=x.loc[x.p_hat.between(.01,.99)].copy(); x['e_t']=(x.focal_wins-x.p_hat)/np.sqrt(x.p_hat*(1-x.p_hat)); x=x.rename(columns={'duel_id':'physical_event_id','focal_player_key':'focal_player_id','opponent_player_key':'opponent_player_id'}); x['residual_valid']=True;x['primary_analysis_eligible']=True;x['p_support_0p01']=True
 asset=x[['physical_event_id','match_id','map_id','event_tick','focal_player_id','opponent_player_id','e_t','residual_valid','primary_analysis_eligible','p_support_0p01']]; asset.to_parquet(a.out/'residual_asset.parquet',index=False,compression='zstd')
 state,audit=base.build_state(asset,'R1-70'); state.to_parquet(a.out/'state_matrix.parquet',index=False,compression='zstd'); gates={'C1':{'status':'pass' if audit['maximum_direct_sum_error']<=1e-10 else 'fail','audit':audit},'C2':{'status':'pass' if audit['match_clusters']>=30 and audit['player_clusters']>=30 else 'fail'}}; write(a.out/'preflight.json',gates)
 if any(v['status']!='pass' for v in gates.values()): raise RuntimeError('preflight_failed')
 smoke=base.fit(state.loc[state.match_id.isin(sorted(state.match_id.unique())[:30])],REF,False); write(a.out/'smoke.json',{'effect_output_withheld':True,'metrics':smoke});
 if not smoke['g3_pass']: raise RuntimeError('smoke_g3_failed')
 if a.phase=='smoke': print('smoke_passed');return
 model=base.fit(state,REF,True); write(a.out/'model.json',model); d4=model['ci_half_width_95']<=.020 and model['mde_80_single']<=.030; gate={'C3':model['g3_pass'],'C4_precision':d4,'C5_positive_confirmation':bool(d4 and model['beta']['estimate']>0 and model['beta']['p_value']<.05 and model['beta']['ci95_low']>0)}; write(a.out/'manifest.json',{'stage':'r1_70_60s_confirmation_v1','status':'passed','rows':len(state),'reference':REF,'gates':gate,'input_lock_sha256':sha(a.out/'input_lock.json'),'model_sha256':sha(a.out/'model.json')}); print('full_passed')
if __name__=='__main__': main()
