"""Recalculate every discoverable 56k screening prediction; no fitting or tuning."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,gzip,hashlib,time,concurrent.futures as cf
import numpy as np
import pandas as pd
from scipy.stats import rankdata,ks_2samp
ROOT=Path(__file__).resolve().parents[1]
Y=IDS=FOLDS=ROWS=None
def load(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as f:return json.load(f)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,allow_nan=False)+'\n');tmp.replace(path)
def init():
    global Y,IDS,FOLDS,ROWS
    b=ROOT/'bank-identities';Y=np.load(b/'development_y.npy');FOLDS=np.load(b/'folds.npy')
    ROWS=np.load(b/'development_rows.npy');IDS=pd.read_parquet(b/'metadata.parquet').application_id.iloc[ROWS].astype(str).to_numpy()
    assert len(Y)==56000 and set(np.unique(FOLDS))=={0,1,2}
def metric(y,p):
    order=np.argsort(p,kind='stable');a=p[order];yy=y[order]
    end=np.r_[np.flatnonzero(a[1:]!=a[:-1]),len(a)-1]
    positive=int(y.sum());negative=len(y)-positive
    cp=np.cumsum(yy)[end]/positive;cn=(end+1-np.cumsum(yy)[end])/negative
    d=cn-cp;ranks=rankdata(p,method='average')
    auc=(ranks[y==1].sum()-positive*(positive+1)/2)/(positive*negative)
    clipped=np.clip(p,1e-15,1-1e-15)
    return {'KS':float(np.max(np.abs(d))),'directional_KS':float(max(0,d.max())),
        'AUC':float(auc),'logloss':float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped)))}
def assess(item):
    out=dict(item)
    try:
        path=Path(item['prediction_path'])
        if not path.is_file():return dict(out,state='missing_saved_predictions')
        actual=sha(path);out['prediction_sha256']=actual
        if item.get('previous_prediction_sha256') and actual!=item['previous_prediction_sha256']:
            return dict(out,state='source_changed_since_previous_audit')
        with np.load(path,allow_pickle=False) as z:
            p=z['prediction'].astype(float)
            assert p.shape==Y.shape and np.isfinite(p).all(),'nonfinite/wrong shape'
            assert ((p>=0)&(p<=1)).all(),'not probabilities'
            assert 'fold' in z and np.array_equal(z['fold'],FOLDS),'fold identity differs'
            if 'rows'in z:assert np.array_equal(z['rows'],ROWS),'row identity differs'
            elif 'application_id'in z:
                assert np.array_equal(z['application_id'].astype(str),IDS),'application identity differs'
                assert 'target'in z and np.array_equal(z['target'],Y),'label identity differs'
            else:raise ValueError('No row identifiers')
        pooled=metric(Y,p)
        independent=float(ks_2samp(p[Y==0],p[Y==1],method='asymp').statistic)
        assert abs(independent-pooled['KS'])<1e-12,'independent KS mismatch'
        if item.get('reported_ks') is not None:assert abs(item['reported_ks']-pooled['KS'])<1e-10,'reported KS mismatch'
        folds=[metric(Y[FOLDS==f],p[FOLDS==f]) for f in range(3)]
        return dict(out,state='verified',pooled=pooled,folds=folds,rows=len(Y),
            score_verification_only=True,validation_integrity_not_implied=True)
    except Exception as e:return dict(out,state='invalid',error=repr(e))
def catalogue():
    items={}
    for mf in (ROOT/'snapshots').glob('*/manifest.json'):
        for s in load(mf)['sources']:
            name=Path(s['source']).name
            if name=='all_prediction_audits.json':
                for row in load(ROOT/s['snapshot']):
                    items[row['prediction_path']]=dict(id=row['id'],prediction_path=row['prediction_path'],
                        reported_ks=row.get('reported_ks'),previous_prediction_sha256=row.get('prediction_sha256'),
                        previous_quality_warnings=row.get('quality_warnings',[]),origin=row.get('origin'),
                        family=row.get('family'),model=row.get('model'))
            if s['category']=='legacy_report' and name in ['scoreboard_fixed_ensemble.json','scoreboard_nested_ensemble.json']:
                campaign=Path(s['source']).parents[1]
                for row in load(ROOT/s['snapshot']):
                    path=str(campaign/'runs'/row['id']/'oof.npz')
                    items[path]=dict(id=campaign.name+'/'+row['id'],prediction_path=path,
                        reported_ks=row.get('pooled_oof',{}).get('ks'),origin='historical',
                        kind=name.removeprefix('scoreboard_').removesuffix('.json'))
    return sorted(items.values(),key=lambda x:x['prediction_path'])
def main():
    (ROOT/'validation').mkdir(exist_ok=True);items=catalogue();start=time.time()
    write(ROOT/'validation/prediction_audit_plan.json',dict(items=len(items),workers=8,training=False,
        scope='All prior single-pipeline audit paths plus both complete legacy fixed/nested ensemble scoreboards.',
        other_protocols='Tracked separately; this verifier never relabels 56k3fold as 70k5fold.'))
    output=ROOT/'validation/prediction_audit.jsonl';done=set()
    if output.exists():
        for line in output.read_text().splitlines():
            try:done.add(json.loads(line)['prediction_path'])
            except ValueError:raise RuntimeError('Truncated previous audit output; retain and repair before resume')
    pending=[r for r in items if r['prediction_path'] not in done];counts={};n=len(done)
    with output.open('a',buffering=1) as file,cf.ProcessPoolExecutor(max_workers=8,initializer=init) as pool:
        for result in pool.map(assess,pending,chunksize=8):
            file.write(json.dumps(result,allow_nan=False)+'\n');n+=1;state=result['state'];counts[state]=counts.get(state,0)+1
            if n%100==0:
                write(ROOT/'validation/prediction_audit_status.json',dict(state='running',done=n,total=len(items),
                    seconds=time.time()-start,new_states=counts));print(json.dumps(dict(done=n,total=len(items))),flush=True)
        file.flush();os.fsync(file.fileno())
    write(ROOT/'validation/prediction_audit_status.json',dict(state='complete',done=n,total=len(items),
        seconds=time.time()-start,new_states=counts,training=False))
if __name__=='__main__':main()
