"""Actual-data parity and information-boundary checks for the handoff factory."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import hashlib,json,time
import numpy as np,pandas as pd
from maturity_features import MaturityFeatures,ALIASES,MEANS

R=Path(__file__).resolve().parent;A=R.parents[1];RAW=A/'exploration-12h-20261017/raw';INPUT=A/'ranking-pipeline-20261020/input'
def main():
 start=time.time();assert not(R/'FACTORY_AUDIT.json').exists()
 apps=pd.concat([pd.read_csv(RAW/(s+'_applications.csv'),usecols=['application_id','customer_id','decision_datetime'])for s in ['train','test']],ignore_index=True)
 rp=pd.read_csv(RAW/'repayment_history.csv',usecols=['customer_id','installment_id','due_date','max_days_past_due'],parse_dates=['due_date'])
 original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','customer_id']);assert original.application_id.tolist()==apps.application_id.iloc[:70000].tolist()
 checks=[]
 for fold in range(5):
  ix=np.load(INPUT/f'artifacts/fold{fold}/indices.npz');ids=original.application_id.iloc[ix['context']].tolist()
  model=MaturityFeatures('eligible_denominators').fit(apps,rp,ids);pred=model.transform(apps,rp).set_index('application_id')
  before=pd.read_parquet(A/f'repayment-severity-fe-20261021/fold{fold}/features.parquet').set_index('application_id')
  new=pd.read_parquet(A/f'mature-denominator-fe-20261021/data/fold{fold}/changed_features.parquet').set_index('application_id')
  errors={}
  for t in [1,30]:
   c=f're_late_{t}_posterior';errors[c]=float(abs(pred[c]-before[f'matched_beta_late_{t}']).max());assert errors[c]<1e-12
  for c,_,_ in ALIASES:errors[c]=float(abs(pred[c]-new[c]).max());assert errors[c]<1e-14
  directory=R/'fitted/eligible_denominators';directory.mkdir(parents=True,exist_ok=True);path=directory/f'fold{fold}.json';model.save(path)
  repeated=MaturityFeatures.load(path).transform(apps,rp).set_index('application_id');assert np.array_equal(pred.to_numpy(),repeated.to_numpy())
  small=MaturityFeatures('posterior_means_only').fit(apps,rp,ids);small_directory=R/'fitted/posterior_means_only';small_directory.mkdir(parents=True,exist_ok=True);small.save(small_directory/f'fold{fold}.json')
  sp=small.transform(apps,rp).set_index('application_id');assert np.array_equal(sp[MEANS].to_numpy(),pred[MEANS].to_numpy())
  checks.append(dict(fold=fold,rows=100000,fit_rows=50000,feature_errors=errors,serialized_transformer_replay_exact=True,posterior_only_matches_extended=True,fit_ids_sha256=model.parameters['fit_application_ids_sha256']))
  print(json.dumps(dict(stage='factory_parity_complete',fold=fold,max_error=max(errors.values()))),flush=True)
 # Target poisoning must be rejected before the factory estimates anything.
 try:model.fit(apps.assign(target=999),rp,ids)
 except ValueError:rejected=True
 else:raise AssertionError('Factory accepted an outcome column')
 # Hold context IDs fixed and corrupt every other customer's historical outcomes.
 fit_ids=original.application_id.iloc[np.load(INPUT/'artifacts/fold0/indices.npz')['context']].tolist()
 original_model=MaturityFeatures('eligible_denominators').fit(apps,rp,fit_ids)
 fit_customers=set(apps.set_index('application_id').loc[fit_ids,'customer_id']);poisoned=rp.copy();poisoned.loc[~poisoned.customer_id.isin(fit_customers),'max_days_past_due']=0
 poison_model=MaturityFeatures('eligible_denominators').fit(apps,poisoned,fit_ids);assert poison_model.parameters==original_model.parameters
 # Future records, application IDs and order are not predictive inputs.
 reference=original_model.transform(apps,rp).set_index('application_id')
 extra=rp.iloc[[0]].copy();extra['installment_id']='FEATURE_FACTORY_FUTURE_AUDIT';extra['due_date']=pd.Timestamp(apps.set_index('customer_id').loc[extra.customer_id.iloc[0],'decision_datetime'])+pd.Timedelta(days=30)
 future=original_model.transform(apps,pd.concat([rp,extra],ignore_index=True)).set_index('application_id');assert np.array_equal(reference.to_numpy(),future.to_numpy())
 shuffled=original_model.transform(apps.sample(frac=1,random_state=9173),rp.sample(frac=1,random_state=9173)).set_index('application_id').loc[reference.index];assert np.array_equal(reference.to_numpy(),shuffled.to_numpy())
 changed=apps.copy();changed.application_id='POISON_'+np.arange(len(apps)).astype(str);identifier_test=original_model.transform(changed,rp).drop(columns='application_id');assert np.array_equal(reference.to_numpy(),identifier_test.to_numpy())
 result=dict(status='passed',folds=checks,default_labels_opened=False,target_column_rejected=rejected,noncontext_history_changes_leave_priors_identical=True,future_event_invariance=True,row_order_invariance=True,application_identifier_invariance=True,source_sha256=hashlib.sha256((R/'maturity_features.py').read_bytes()).hexdigest(),seconds=time.time()-start,scope='Actual100kapplicant parity for all7fields/all5folds plus metamorphic boundary checks; verifies feature factory, not untouched predictive generalization.')
 (R/'FACTORY_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='passed',seconds=time.time()-start)),flush=True)
if __name__=='__main__':main()
