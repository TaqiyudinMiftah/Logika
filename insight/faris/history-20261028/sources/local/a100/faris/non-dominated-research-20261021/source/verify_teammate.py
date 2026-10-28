"""Readonly metric verification of every current teammate V7 saved OOF file."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import json,hashlib,time
import numpy as np
import pandas as pd
from scipy.stats import rankdata,ks_2samp
ROOT=Path(__file__).resolve().parents[1]
HISTORY=Path.home()/'adit/Logika/insight/goal054_offset_baseline_20261019/v7'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def metric(y,p):
    positive=int(y.sum());negative=len(y)-positive;ranks=rankdata(p,method='average')
    a=np.argsort(p,kind='stable');end=np.r_[np.flatnonzero(p[a][1:]!=p[a][:-1]),len(p)-1]
    cp=np.cumsum(y[a])[end];d=(end+1-cp)/negative-cp/positive
    ks=float(np.max(abs(d)));independent=float(ks_2samp(p[y==0],p[y==1],method='asymp').statistic)
    assert abs(ks-independent)<1e-12
    q=np.clip(p,1e-15,1-1e-15)
    return dict(KS=ks,AUC=float((ranks[y==1].sum()-positive*(positive+1)/2)/(positive*negative)),
        logloss=float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q))))
def main():
    start=time.time();dest=ROOT/'validation';dest.mkdir(exist_ok=True)
    master_path=HISTORY/'reports/EXPERIMENT_MASTER_LOG.json';master=json.loads(master_path.read_text())
    reference_path=HISTORY/'runs/V00/OOF.csv';reference=pd.read_csv(reference_path)
    assert len(reference)==70000 and reference.application_id.is_unique
    assert set(reference.fold.unique())==set(range(5))
    frozen=reference[['application_id','target','fold']]
    plan=dict(source=str(master_path),master_sha256=sha(master_path),reference_path=str(reference_path),
        reference_sha256=sha(reference_path),training=False,labels='Compared against frozen V00 OOF identity, not independently reconstructed from raw labels in this job.',
        score_recalculation_not_scientific_gate=True)
    (dest/'teammate_audit_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    output=dest/'teammate_audit.jsonl';assert not output.exists(),'Preserve existing audit'
    n=0;counts={}
    with output.open('w',buffering=1) as file:
        for row in master['active_experiments']:
            identifier=row['id'];result_path=HISTORY/'runs'/identifier/'RESULT.json';path=result_path.parent/'OOF.csv'
            out=dict(id=identifier,prediction_path=str(path))
            try:
                if not path.exists():out['state']='missing_saved_predictions'
                else:
                    result=json.loads(result_path.read_text());frame=pd.read_csv(path)
                    assert frame[['application_id','target','fold']].equals(frozen),'Frozen identity mismatch'
                    y=frame.target.to_numpy(int);p=frame.prediction.to_numpy(float)
                    assert set(np.unique(y))=={0,1} and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                    pooled=metric(y,p)
                    for k,v in pooled.items():
                        if k in result['OOF']:assert abs(v-result['OOF'][k])<1e-10,'Reported metric mismatch '+k
                    fold=frame.fold.to_numpy();folds=[metric(y[fold==f],p[fold==f]) for f in range(5)]
                    out.update(state='verified',pooled=pooled,folds=folds,prediction_sha256=sha(path),
                        result_sha256=sha(result_path),configuration=result.get('spec'),columns=result.get('columns'),
                        source_point_in_time_certified=result.get('point_in_time_certified'),
                        independent_holdout=False,scientific_rank_eligible=False)
            except Exception as e:out.update(state='invalid',error=repr(e))
            file.write(json.dumps(out,allow_nan=False)+'\n');n+=1;counts[out['state']]=counts.get(out['state'],0)+1
            (dest/'teammate_audit_status.json').write_text(json.dumps(dict(status='running',done=n,total=len(master['active_experiments']),states=counts,seconds=time.time()-start))+'\n')
    (dest/'teammate_audit_status.json').write_text(json.dumps(dict(status='complete',done=n,total=len(master['active_experiments']),states=counts,seconds=time.time()-start,training=False))+'\n')
if __name__=='__main__':main()
