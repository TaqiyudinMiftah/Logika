"""Verify deterministic representation overlap, without applicant default labels."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time
import numpy as np,pandas as pd,torch,sdm
import baseline as b
R=Path(__file__).resolve().parents[1];A=R.parent;DATA=A/'kumo-temperature-adaptation-20261021/full_data';INPUT=A/'ranking-pipeline-20261020/input'
def main():
 assert not(R/'SOURCE_REDUNDANCY.json').exists();start=time.time();original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id']);checks=[]
 columns=['application_id','has_account','aux_account_months','v2_ecom_observed_days','aux_ecommerce_observed_days']
 for f in range(5):
  d=DATA/f'fold{f}';m=json.loads((d/'manifest.json').read_text());assert b.sha(d/'context.parquet')==m['files']['context.parquet'];x=pd.read_parquet(d/'context.parquet',columns=columns);ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz');assert len(x)==50000 and x.application_id.tolist()==original.application_id.iloc[ix['context']].tolist()
  account_exact=np.array_equal(x.aux_account_months.to_numpy(),3*x.has_account.to_numpy(),equal_nan=True)
  frame=x[['v2_ecom_observed_days','aux_ecommerce_observed_days']];table=sdm.TableTensor.from_pandas(frame,stypes={c:'numerical'for c in frame},device='cpu');numeric=table.numerical
  assert numeric.shape==(50000,2)
  finite=torch.isfinite(numeric[:,0])&torch.isfinite(numeric[:,1]);same_missing=torch.equal(torch.isfinite(numeric[:,0]),torch.isfinite(numeric[:,1]));span_exact=bool(torch.equal(numeric[finite,0],numeric[finite,1]))and same_missing
  checks.append(dict(fold=f,fit_rows=50000,context_ids_sha256=b.ids_hash(original,ix['context']),account_months_exactly_three_times_existing_flag=bool(account_exact),span_identical_at_actual_TableTensor_precision=span_exact,actual_numerical_dtype=str(numeric.dtype),span_raw_max_abs_difference=float(np.nanmax(abs(x.v2_ecom_observed_days-x.aux_ecommerce_observed_days))),span_missing_masks_equal=same_missing))
 assert all(v['account_months_exactly_three_times_existing_flag']and v['span_identical_at_actual_TableTensor_precision']for v in checks)
 b.dump(dict(status='passed',default_labels_opened=False,checks=checks,distinct_candidate_coverage_fields=['risk_conditional_negative_months','risk_conditional_nonnegative_months','aux_orders90','aux_allocation_count'],classification_training_admitted=False,source_sha256=b.sha(Path(__file__)),seconds=time.time()-start,qualification='Actual supplied50kcontext data in eachfold. This verifies representation overlap, not predictive usefulness or future-data equivalence. A future selector must fit inside eachcontext; high correlation alone is not proof of dispensability.'),R/'SOURCE_REDUNDANCY.json')
 print(json.dumps(dict(status='passed',contexts=5,input_dtype=str(numeric.dtype))),flush=True)
if __name__=='__main__':main()
