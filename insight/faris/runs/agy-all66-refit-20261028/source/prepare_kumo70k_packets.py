"""Emit all70k-context Kumo56 analogue packets and a separate70k head table."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import hashlib,json,sys
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];A=Path('/mnt/home/faris/logika-faris-exploration');F=A/'submission-full-periodicity-20261021';sys.path.insert(0,str(F/'source'))
from maturity_features import MaturityFeatures

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main():
 prior=R/'prior70k/fold0';assert json.loads((R/'prior70k/COMPLETE.json').read_text())['status']=='complete';frame=pd.read_parquet(prior/'features.parquet');apps=pd.read_parquet(F/'applications.parquet').drop(columns=['row'],errors='ignore');assert frame.application_id.tolist()==apps.application_id.tolist()
 ctx=np.arange(70000);query_rows=np.arange(70000,100000);config=json.loads((R/'ENGINE_CONFIGS.json').read_text())['models']['kumo56'];columns=config['columns'];assert len(columns)==90
 y=pd.read_parquet(A/'ranking-pipeline-20261020/input/data/train_features.parquet',columns=['target']).target.to_numpy(int)
 shared=R/'packets/kumo56/shared70k';shared.mkdir(parents=True,exist_ok=True);context=frame.iloc[ctx][columns].copy();context['target']=y;query=frame.iloc[query_rows][columns].copy();context.to_parquet(shared/'context.parquet',index=False);query.to_parquet(shared/'query.parquet',index=False);np.save(shared/'context_rows.npy',ctx);np.save(shared/'query_rows.npy',query_rows)
 for fold in range(5):
  packet=R/f'packets/kumo56/fold{fold}';packet.mkdir(parents=True,exist_ok=True)
  for name in ['context.parquet','query.parquet','context_rows.npy','query_rows.npy']:
   p=packet/name
   if not p.exists():os.link(shared/name,p)
  dump(packet/'meta.json',{'model':'kumo56','fold':fold,'engine':config['engine'],'seed':config['seed_base']+fold*config['fold_seed_increment'],'num_estimators':8,'columns':columns,'n_context':70000,'n_query':30000,'context_sha256':sha(shared/'context.parquet'),'query_sha256':sha(shared/'query.parquet'),'query_rows_sha256':sha(shared/'query_rows.npy'),'context_rows_sha256':sha(shared/'context_rows.npy'),'prior70k_manifest_sha256':sha(prior/'complete.json'),'correction_prediction_policy':'Keep exact original nestedinnerOOF rawcorrection predictions; productiontest nowall70kcontext; no querylabels','shared_feature_table_reason':'All70kcontext membership identical acrossfoldseeds; deterministic70kpriorbuilder shared','fitted_weights_saved':False})
 # Correctedmat18_kumo56 uses maturemeans/coverage for test headfeatures;
 # retain oldcachedcorrectionheadfeatures separately in assembly.
 cov=pd.read_parquet(F/'coverage.parquet').set_index('application_id').loc[apps.application_id];spectral=pd.read_parquet(A/'shopping-periodicity-fe-20261021/true_features.parquet').set_index('application_id').loc[apps.application_id]
 for table in [cov,spectral]:
  for c in table:
   if c!='customer_id':frame[c]=table[c].to_numpy()
 history=pd.read_csv(A/'exploration-12h-20261017/raw/repayment_history.csv',usecols=['customer_id','installment_id','due_date','max_days_past_due']);factory=MaturityFeatures().fit(apps,history,apps.application_id.iloc[ctx]);mature=factory.transform(apps,history)
 for c in ['re_late_1_posterior','re_late_30_posterior']:frame[c]=mature[c].to_numpy()
 factory.save(R/'prior70k/MATURITY_PARAMETERS.json');frame.to_parquet(R/'prior70k/head_features.parquet',index=False)
 dump(R/'prior70k/PACKETS_COMPLETE.json',{'status':'complete','context_rows':70000,'test_rows':30000,'folds':5,'head_features_sha256':sha(R/'prior70k/head_features.parquet'),'original_kumo_prior_table_sha256':sha(prior/'features.parquet'),'default_labels_fit_feature_priors':False,'fitted_weights_saved':False,'warning':'Correctionheads use historicalnestedOOF56k raw and originalcorrectionfeaturevalues; test nowlarger70kcontext. This is a documentedrefitpolicy, not a newvalidationmeasurement.'})
 print('All70kKumo56packets and70ktest-head features ready',flush=True)
if __name__=='__main__':main()
