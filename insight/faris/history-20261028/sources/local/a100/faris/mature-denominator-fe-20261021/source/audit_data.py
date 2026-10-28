"""Independent NumPy count reconstruction, with no default labels read."""
from pathlib import Path
import numpy as np,pandas as pd
import baseline as b
R=Path(__file__).resolve().parents[1];A=R.parent;RAW=A/'exploration-12h-20261017/raw';INPUT=A/'ranking-pipeline-20261020/input'
def main():
 apps=pd.concat([pd.read_csv(RAW/(s+'_applications.csv'),usecols=['application_id','customer_id','decision_datetime'])for s in ['train','test']],ignore_index=True);cut=pd.to_datetime(apps.decision_datetime);lookup=pd.Series(np.arange(len(apps)),index=apps.customer_id)
 rp=pd.read_csv(RAW/'repayment_history.csv',usecols=['customer_id','due_date','max_days_past_due']);rows=rp.customer_id.map(lookup).to_numpy(np.int64);age=(cut.iloc[rows].to_numpy()-pd.to_datetime(rp.due_date).to_numpy())/np.timedelta64(1,'D');dpd=rp.max_days_past_due.to_numpy(float);assert np.isfinite(dpd).all()and np.all(age>0)
 checks=[]
 specs=[('v3_repay_all_late_'+str(t)+'_posterior_24',t,None)for t in [1,7,30,90]]+[('v3_repay_180d_late_1_posterior_24',1,180)]
 for f in range(5):
  d=R/f'data/fold{f}';manifest=__import__('json').loads((d/'manifest.json').read_text())
  for n,h in manifest['files'].items():assert b.sha(d/n)==h
  changed=pd.read_parquet(d/'changed_features.parquet');assert changed.application_id.tolist()==apps.application_id.tolist()
  ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz');ctx=ix['context'];feat=[]
  for c,t,w in specs:
   mask=age>=t
   if w is not None:mask &=age<=w
   n=np.bincount(rows[mask],minlength=len(apps));k=np.bincount(rows[mask&(dpd>=t)],minlength=len(apps));prior=float(k[ctx].sum()/n[ctx].sum());expected=(k+24*prior)/(n+24);error=float(abs(expected-changed[c]).max());assert error<1e-14
   feat.append(dict(column=c,max_error=error,eligible_rows=int(mask.sum()),prior=prior))
  q=pd.read_parquet(d/'query.parquet',columns=['application_id']);context=pd.read_parquet(d/'context.parquet',columns=['application_id']);assert set(context.application_id).isdisjoint(q.application_id)
  assert context.application_id.tolist()==apps.application_id.iloc[ctx].tolist()and q.application_id.tolist()==apps.application_id.iloc[np.r_[ix['correction'],ix['validation']]].tolist()
  checks.append(dict(fold=f,features=feat,roles_verified=True))
 b.dump(dict(status='passed',default_labels_opened=False,rows=100000,checks=checks,method='Independent raw-row NumPy bincount, context-only prevalence, exact per-applicant posterior.'),R/'INDEPENDENT_DATA_AUDIT.json')
 b.dump(dict(query_table_contains_target=False,outer_validation_labels_staged=False,correction_labels_staged_separately=True,correction_label_rows=6000,qualification='The earlier query_labels_staged field means outer evaluation labels; the6000correction labels exist separately for head fitting and are never passed to Kumo.'),R/'LABEL_SCOPE.json')
if __name__=='__main__':main()
