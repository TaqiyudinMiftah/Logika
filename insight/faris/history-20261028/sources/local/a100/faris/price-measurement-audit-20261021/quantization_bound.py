"""Descriptive error envelope conditional on the observed nearest1000 grid."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'));import baseline as b
A=R.parent;RAW=A/'exploration-12h-20261017/raw';INPUT=A/'ranking-pipeline-20261020/input'
def main():
 assert not(R/'ROUNDING_SCALE.json').exists();audit=json.loads((R/'RESULT.json').read_text());assert audit['default_labels_opened']is False
 apps=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','customer_id']);ix=np.load(INPUT/'artifacts/fold0/indices.npz');ids=set(apps.customer_id.iloc[ix['context']]);raw=RAW/'ecommerce_transactions.csv';assert b.sha(raw)==audit['source_raw_sha256']
 d=pd.read_csv(raw,usecols=['customer_id','quantity','item_total']);d=d[d.customer_id.isin(ids)];u=(d.item_total/d.quantity).to_numpy(float);assert len(u)==audit['summary']['item_rows']and np.all(u>=5000)
 nonfloor=u>5000;v=u[nonfloor];lo=np.log(v-500);hi=np.log(v+500);observed=np.log(v);envelope=np.maximum(abs(lo-observed),abs(hi-observed));uniform_variance=(hi-lo)**2/12
 prior=json.loads((A/'fold-local-priors-20261021/fold0/prior_audit.json').read_text())['price'];sigma=prior['estimated_item_log_noise_sd']
 result=dict(status='complete',default_labels_opened=False,context_rows=50000,nonfloor_item_rows=len(v),floor_rows_excluded=int((~nonfloor).sum()),fraction_units_le_10000=float(np.mean(u<=10000)),mean_squared_nonfloor_error_envelope=float(np.mean(envelope**2)),envelope_relative_to_fitted_item_noise_variance=float(np.mean(envelope**2)/sigma**2),mean_uniform_loginterval_variance=float(np.mean(uniform_variance)),uniform_interval_relative_to_item_noise_variance=float(np.mean(uniform_variance)/sigma**2),nonfloor_error_envelope_quantiles=np.quantile(envelope,[0,.5,.9,.99,1]).tolist(),source_sha256=b.sha(Path(__file__)),qualification='Conditional numericalenvelope only: assumes latent unroundedunit lies in observedunit+-500 as impliedby nearest1000interpretation. Originalgenerator rule isnotproven; floor observations areexcluded because their lowerlatentbound isunknown. Uniformwithinintervalvariance isillustrative,notestimatedtruth. This isnotdefaultscoring evidence or a boundonKS.')
 b.dump(result,R/'ROUNDING_SCALE.json');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
