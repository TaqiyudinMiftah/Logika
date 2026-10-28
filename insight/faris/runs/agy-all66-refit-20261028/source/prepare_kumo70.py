"""Build the distinct all70k Kumo56 production packets and head query frame.

Kumo inputs retain original nested-correction priors; matched maturity is applied
only to the head frame, matching the selected recipe's corrected-head source.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];A=Path('/mnt/home/faris/logika-faris-exploration');F=A/'submission-full-periodicity-20261021';FRZ=Path('/mnt/home/faris/logika-frozen-shadow-20261026')
sys.path.insert(0,str(F/'source'));from maturity_features import MaturityFeatures

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
apps=pd.read_parquet(F/'applications.parquet').drop(columns=['row'],errors='ignore');assert len(apps)==100000 and apps.application_id.is_unique
path=R/'prior70k/fold0/features.parquet';receipt=json.loads((R/'prior70k/fold0/complete.json').read_text());assert sha(path)==receipt['feature_table_sha256']
frame=pd.read_parquet(path);assert frame.application_id.tolist()==apps.application_id.tolist()
coverage=pd.read_parquet(F/'coverage.parquet').set_index('application_id').loc[apps.application_id].reset_index(drop=True)
for c in coverage:frame[c]=coverage[c].to_numpy()
head=json.loads((FRZ/'frozen/fold0/MANIFEST.json').read_text())['head_columns'];spec=json.loads((R/'ENGINE_CONFIGS.json').read_text())['models']['kumo56'];columns=spec['columns']
y=pd.read_parquet(A/'ranking-pipeline-20261020/input/data/train_features.parquet',columns=['target']).target.to_numpy(int)
assert not {'target'}&set(frame)
ctx=np.arange(70000);queryrows=np.arange(70000,100000)
for fold in range(5):
 p=R/f'packets/kumo56/fold{fold}';assert not p.exists();p.mkdir(parents=True)
 context=frame.iloc[ctx][columns].reset_index(drop=True).copy();context['target']=y
 query=frame.iloc[queryrows][columns].reset_index(drop=True).copy();context.to_parquet(p/'context.parquet',index=False);query.to_parquet(p/'query.parquet',index=False);np.save(p/'context_rows.npy',ctx);np.save(p/'query_rows.npy',queryrows)
 dump(p/'meta.json',{'model':'kumo56','fold':fold,'engine':spec['engine'],'seed':spec['seed_base']+fold*spec['fold_seed_increment'],'columns':columns,'n_context':70000,'n_query':30000,'num_estimators':8,'source_sha256':sha(__file__),'context_sha256':sha(p/'context.parquet'),'query_sha256':sha(p/'query.parquet'),'context_rows_sha256':sha(p/'context_rows.npy'),'query_rows_sha256':sha(p/'query_rows.npy'),'policy':'All70kcontext; exact historicalnestedOOFcorrectionraw remains separately cached; noquerylabels orfittedweights serialized'})
history=pd.read_csv(A/'exploration-12h-20261017/raw/repayment_history.csv',usecols=['customer_id','installment_id','due_date','max_days_past_due'])
mature=MaturityFeatures('eligible_denominators').fit(apps,history,apps.application_id.iloc[ctx]).transform(apps,history)
headframe=frame[['application_id','customer_id']+head].copy()
for c in ['re_late_1_posterior','re_late_30_posterior']:headframe[c]=mature[c].to_numpy()
out=R/'features70';out.mkdir(exist_ok=True);headframe.to_parquet(out/'expanded.parquet',index=False)
dump(out/'COMPLETE.json',{'status':'complete','source_sha256':sha(__file__),'prior_table_sha256':sha(path),'head_frame_sha256':sha(out/'expanded.parquet'),'rows':100000,'Kumo_context_rows':70000,'query_labels_used':False,'maturity_policy':'Matchedmaturity in headfeatures; original nestedKumofeaturepriors preserved','original_correction_features_policy':'Assembler usesexactcachedoriginalfoldheadfeatures andnestedOOFraw'})
print(json.dumps({'stage':'kumo70_packets_ready','folds':5,'head_frame':str(out/'expanded.parquet')}),flush=True)
