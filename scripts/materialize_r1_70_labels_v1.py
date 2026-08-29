"""One-time authorized R1 outcome materialization keyed to frozen p_hat rows."""
from __future__ import annotations
import argparse, json, os
from datetime import UTC, datetime
from pathlib import Path
import pandas as pd
import extract_r1_70_core_preoutcome_v1 as base

CORE=Path(r"D:\CS2_60s_confirmation_r1_70_preoutcome_v1\core_tables_v1")
SCORES=Path(r"D:\CS2_60s_confirmation_r1_70_preoutcome_v1\p_hat_preoutcome_v1")
OUT=Path(r"D:\CS2_60s_confirmation_r1_70_preoutcome_v1\label_release_v1")

def now(): return datetime.now(UTC).isoformat()
def sha(p): return base.sha256(p)
def write(path, value, replace=False):
    if path.exists() and not replace: raise RuntimeError(f"refusing_to_overwrite:{path}")
    temp=path.with_name(path.name+'.tmp'); temp.write_text(json.dumps(value,ensure_ascii=True,indent=2)+'\n',encoding='utf-8'); os.replace(temp,path)

def labels(core_map: Path, score_map: Path, salt: bytes, staging: Path) -> pd.DataFrame:
    source=json.loads((core_map/'map_manifest.json').read_text(encoding='utf-8')); candidate=pd.read_parquet(score_map/'p_hat_preoutcome.parquet')
    if candidate.empty: return pd.DataFrame(columns=['duel_id','focal_wins'])
    archive=Path(source['selection']['archive']['path']); demo=base.extract_member(archive,source['selection']['demo_member'],staging)
    try:
        deaths=base.event_frame(base.DemoParser(str(demo)),'player_death')
    finally: demo.unlink(missing_ok=True)
    required={'tick','attacker_steamid','user_steamid'}
    if required-set(deaths): raise RuntimeError('label_event_fields_missing')
    deaths=deaths.loc[:,['tick','attacker_steamid','user_steamid']].copy(); deaths['event_tick']=pd.to_numeric(deaths.tick,errors='coerce'); deaths=deaths.dropna(subset=['event_tick']); deaths['event_tick']=deaths.event_tick.astype('int64')
    deaths['attacker_key']=deaths.attacker_steamid.map(lambda x:base.hmac_key(salt,x)); deaths['victim_key']=deaths.user_steamid.map(lambda x:base.hmac_key(salt,x)); deaths=deaths.loc[deaths.attacker_key.ne('')&deaths.victim_key.ne('')]
    keys=pd.concat([deaths[['event_tick','attacker_key']].rename(columns={'attacker_key':'player_key'}),deaths[['event_tick','victim_key']].rename(columns={'victim_key':'player_key'})])
    joined=candidate.merge(deaths[['event_tick','attacker_key','victim_key']],on='event_tick',how='left',validate='many_to_many')
    joined=joined.loc[((joined.focal_player_key.eq(joined.attacker_key))&(joined.opponent_player_key.eq(joined.victim_key)))|((joined.focal_player_key.eq(joined.victim_key))&(joined.opponent_player_key.eq(joined.attacker_key)))].copy()
    if joined.duel_id.duplicated().any() or len(joined)!=len(candidate): raise RuntimeError('candidate_label_join_not_exact')
    joined['focal_wins']=joined.focal_player_key.eq(joined.attacker_key).astype('int8')
    return joined[['duel_id','focal_wins']]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--core-root',type=Path,default=CORE); p.add_argument('--scores-root',type=Path,default=SCORES); p.add_argument('--output-root',type=Path,default=OUT); p.add_argument('--max-maps',type=int,default=0); p.add_argument('--resume',action='store_true'); a=p.parse_args()
    if not all(x.is_file() for x in (a.core_root/'manifest.json',a.scores_root/'manifest.json',a.core_root/'access_control'/'r1_70_player_hmac_salt.bin')): raise SystemExit('locked_input_missing')
    if a.output_root.exists() and any(a.output_root.iterdir()) and not a.resume: raise SystemExit('nonempty_output_requires_resume')
    a.output_root.mkdir(parents=True,exist_ok=True); salt=(a.core_root/'access_control'/'r1_70_player_hmac_salt.bin').read_bytes(); receipt=a.output_root/'access_receipt.json'
    if not receipt.exists(): write(receipt,{'stage':'r1_70_label_release_receipt_v1','authorized_at_utc':now(),'outcome_access':'authorized_by_user','core_manifest_sha256':sha(a.core_root/'manifest.json'),'scores_manifest_sha256':sha(a.scores_root/'manifest.json'),'script_sha256':sha(Path(__file__).resolve())})
    progress=a.output_root/'progress.json'; prior={int(x['map_ordinal']):x for x in json.loads(progress.read_text()).get('runs',[])} if progress.exists() else {}; runs=[]; processed=0
    for core_map in sorted((a.core_root/'maps').iterdir()):
        ordinal=int(json.loads((core_map/'map_manifest.json').read_text())['selection']['map_ordinal']); target=a.output_root/'maps'/core_map.name
        if prior.get(ordinal,{}).get('status')=='completed': runs.append(prior[ordinal]); continue
        if a.max_maps and processed>=a.max_maps: continue
        try:
            lab=labels(core_map,a.scores_root/'maps'/core_map.name,salt,a.output_root/'staging'); partial=target.with_name(target.name+'.partial'); partial.mkdir(parents=True); lab.to_parquet(partial/'labels.parquet',index=False,compression='zstd'); write(partial/'map_manifest.json',{'map_ordinal':ordinal,'rows':len(lab),'labels_sha256':sha(partial/'labels.parquet'),'outcome_access':'released_once'}); partial.replace(target); run={'map_ordinal':ordinal,'status':'completed','rows':len(lab)}
        except Exception as e: run={'map_ordinal':ordinal,'status':'failed','error':str(e)[-1000:]}
        runs.append(run); processed+=1; write(progress,{'stage':'r1_70_label_release_v1','runs':runs,'updated_at_utc':now()},True)
        if run['status']=='failed': raise RuntimeError('map_failed:'+run['error'])
        print(f'[{processed}] map={ordinal} rows={run["rows"]}',flush=True)
    maps=sorted((a.core_root/'maps').iterdir())
    if len(runs)==len(maps) and all(x.get('status')=='completed' for x in runs):
        final=pd.concat([pd.read_parquet(x/'labels.parquet') for x in sorted((a.output_root/'maps').iterdir())],ignore_index=True)
        if final.duel_id.duplicated().any() or not set(final.focal_wins.unique()).issubset({0,1}): raise RuntimeError('final_labels_invalid')
        final.to_parquet(a.output_root/'labels.parquet',index=False,compression='zstd'); write(a.output_root/'manifest.json',{'stage':'r1_70_label_release_v1','status':'released_once','rows':len(final),'labels_sha256':sha(a.output_root/'labels.parquet'),'core_manifest_sha256':sha(a.core_root/'manifest.json'),'scores_manifest_sha256':sha(a.scores_root/'manifest.json'),'script_sha256':sha(Path(__file__).resolve()),'runs':runs,'completed_at_utc':now()})
if __name__=='__main__': main()
