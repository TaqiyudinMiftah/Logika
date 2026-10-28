"""Rank source-backed additions on one correction half, never outer labels."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='2'
from pathlib import Path
import collections,datetime,hashlib,json,time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from features import Features,formula_gate
from materialize import digest
R=Path(__file__).resolve().parents[1]
def unit(v):
    v=np.asarray(v,dtype=float);v=v-v.mean();norm=np.linalg.norm(v)
    return v/norm if norm>1e-12 else np.zeros_like(v)
def vectors(value,context,ranking):
    missing=~np.isfinite(value[ranking]);median=np.nanmedian(np.where(np.isfinite(value[context]),value[context],np.nan))
    filled=np.where(missing,median,value[ranking])
    return unit(rankdata(filled)),unit(missing.astype(float))
def main():
    start=time.time();plan=json.loads((R/'PLAN.json').read_text());assert time.time()<plan['search_deadline_epoch']
    (R/'selection').mkdir(exist_ok=False)
    cat=json.loads((R/'CATALOG.json').read_text());base=next(p['features'] for p in cat['packs'] if p['id']=='baseline90')
    assert hashlib.sha256((R/'CATALOG.json').read_bytes()).hexdigest()==plan['catalog_sha256']
    assert len(base)==90
    specs=[s for s in cat['feature_definitions'] if s['op']!='identity'];families=sorted({s['family'] for s in specs})
    checks=[]
    for fold in range(5):
        f=Features(fold);yc=f.ix['y_correction'];ranking,choice=train_test_split(np.arange(6000),test_size=.5,stratify=yc,random_state=261026)
        rows=f.ix['correction'][ranking];residual=unit(yc[ranking]-np.load(R/f'packets/fold{fold}/baseline.npz')['raw'][ranking])
        baseline_digests=set();baseline_vectors=[]
        for fid in base:
            v=f.compute(f.specs[fid]);baseline_digests.add(digest(v[f.ctx]))
            if pd.api.types.is_numeric_dtype(v):baseline_vectors.extend(vectors(v,f.ctx,rows))
        baseline_matrix=np.asarray(baseline_vectors)
        scores={};candidate_vectors={};rejections=collections.Counter();candidates=collections.defaultdict(list)
        for spec in specs:
            fid=spec['id'];value=f.compute(spec);ctx=value[f.ctx];query=value[rows]
            finite=np.isfinite(ctx)
            if finite.sum()<500 or np.isfinite(query).sum()<200:rejections['low_support']+=1;continue
            if np.nanmax(ctx)==np.nanmin(ctx) and finite.all():rejections['constant']+=1;continue
            h=digest(ctx)
            if h in baseline_digests:rejections['exact_baseline_duplicate']+=1;continue
            z,missing=vectors(value,f.ctx,rows)
            a,b=abs(float(z@residual)),abs(float(missing@residual))
            vector=z if a>=b else missing;score=max(a,b)
            if score<=1e-12:rejections['no_rank_signal']+=1;continue
            if np.max(np.abs(baseline_matrix@vector))>.995:rejections['baseline_rank_redundancy']+=1;continue
            scores[fid]=score;candidate_vectors[fid]=vector.astype(np.float32)
            candidates[spec['family']].append((score,fid,h))
        groups={};top=[]
        for family in families:
            chosen=[];chosen_vectors=[];seen=set(baseline_digests)
            for score,fid,h in sorted(candidates[family],key=lambda x:(-x[0],x[1])):
                v=candidate_vectors[fid]
                if h in seen or any(abs(float(v@previous))>.98 for previous in chosen_vectors):continue
                chosen.append(fid);chosen_vectors.append(v);seen.add(h)
                if len(chosen)==10:break
            groups[family]=chosen
            if chosen:top.append((scores[chosen[0]],chosen[0],family))
        diverse=[];diverse_vectors=[]
        for score,fid,family in sorted(top,key=lambda x:(-x[0],x[1])):
            v=candidate_vectors[fid]
            if all(abs(float(v@prev))<=.98 for prev in diverse_vectors):diverse.append(fid);diverse_vectors.append(v)
            if len(diverse)==10:break
        groups['diverse']=diverse
        record={'fold':fold,'baseline_features':base,'groups':groups,'families':families,
                'ranking_rows':rows.tolist(),'choice_rows':f.ix['correction'][choice].tolist(),
                'selected_scores':{fid:scores[fid] for ids in groups.values() for fid in ids},
                'rejections':dict(rejections),'outer_labels_loaded':False,'ranking_rows_count':len(ranking),
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'selected_feature_definitions':{fid:f.specs[fid] for ids in groups.values() for fid in ids}}
        (R/f'selection/fold{fold}.json').write_text(json.dumps(record,indent=2)+'\n')
        np.savez_compressed(R/f'selection/scores_fold{fold}.npz',ids=np.array(list(scores)),score=np.array(list(scores.values()),dtype=np.float32))
        check={'fold':fold,'groups':{k:len(v) for k,v in groups.items()},'baseline_columns':len(base),'max_columns':max(90+len(v) for v in groups.values()),'screened_definitions':len(specs),'retained_candidates':len(scores)}
        checks.append(check);print(json.dumps(check),flush=True)
    record={'status':'complete','folds':checks,'formula_checks':formula_gate(),'seconds':time.time()-start,
            'selection_fingerprints':{f'fold{f}':hashlib.sha256((R/f'selection/fold{f}.json').read_bytes()).hexdigest() for f in range(5)}}
    (R/'FEATURE_GATE.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'stage':'features_frozen','seconds':time.time()-start}),flush=True)
if __name__=='__main__':main()
