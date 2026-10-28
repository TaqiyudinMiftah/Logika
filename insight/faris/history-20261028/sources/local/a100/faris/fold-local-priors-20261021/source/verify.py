"""Independent membership, output identity and fold-prior checks; no fitting."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    plan=json.loads((ROOT/'plan.json').read_text());inp=Path(plan['input']);raw=Path(plan['raw'])
    original=pd.read_parquet(inp/'data/train_features.parquet',columns=['application_id','customer_id'])
    apps=pd.read_csv(raw/'train_applications.csv',usecols=['customer_id','decision_datetime'],parse_dates=['decision_datetime'])
    rep=pd.read_csv(raw/'repayment_history.csv',usecols=['customer_id','due_date','max_days_past_due'],parse_dates=['due_date'])
    rep=rep.merge(apps,on='customer_id',validate='many_to_one')
    rep=rep[(rep.decision_datetime-rep.due_date).dt.total_seconds()>=86400]
    seen=np.zeros(len(original),int);checks=[]
    for f in range(5):
        root=ROOT/f'fold{f}';complete=json.loads((root/'complete.json').read_text())
        assert sha(root/'features.parquet')==complete['feature_table_sha256']
        frame=pd.read_parquet(root/'features.parquet');assert len(frame)==100000
        assert frame.application_id.is_unique and frame.customer_id.is_unique and 'target' not in frame
        assert frame.application_id.iloc[:70000].tolist()==original.application_id.tolist()
        ix=dict(np.load(inp/f'artifacts/fold{f}/indices.npz'))
        ci,hi,vi=ix['context'],ix['correction'],ix['validation']
        assert len(ci)==50000 and len(hi)==6000 and len(vi)==14000
        assert not set(ci)&set(hi) and not set(ix['train'])&set(vi)
        assert np.array_equal(np.sort(np.r_[ci,hi]),ix['train'])
        ids='\n'.join(original.application_id.iloc[ci]).encode();assert hashlib.sha256(ids).hexdigest()==complete['fit_ids_sha256']
        fit=set(original.customer_id.iloc[ci]);actual=float(rep[rep.customer_id.isin(fit)].max_days_past_due.ge(1).mean())
        reported=json.loads((root/'workspace/artifacts/optimization/v4/feature_audit.json').read_text())['mature_late_1_prior']
        assert abs(actual-reported)<1e-14
        seen[vi]+=1;checks.append(dict(fold=f,feature_sha256=complete['feature_table_sha256'],context_ids_sha256=complete['fit_ids_sha256'],independently_recomputed_mature_late_1_prior=actual,validation_rows=14000))
    assert np.all(seen==1)
    out=dict(status='passed',folds=5,unique_development_rows=70000,every_validation_row_once=True,
        feature_rows_each=100000,no_targets_in_feature_outputs=True,checks=checks,
        scope='Output hashes, frozen context/validation membership and independently recalculated mature repayment priors. Vendor source review supplies other fitting-scope evidence.',
        historical_raw_snapshot_provenance_certified=False)
    (ROOT/'INDEPENDENT_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
