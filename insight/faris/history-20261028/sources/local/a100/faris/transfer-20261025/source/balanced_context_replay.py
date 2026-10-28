"""Fresh reconstruction of a retained H015 context and downstream prediction."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit,logit
import baseline as b
from gpu_backend import Engine
from robustness import R,inputs,stages,read

p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),default=0)
p.add_argument('--kind',choices=['balanced','proportional'],default='balanced')
p.add_argument('--mode',choices=['original','fresh_kumo_seed'],default='original')
p.add_argument('--output',required=True);args=p.parse_args()
output=Path(args.output);assert output.suffix=='.npz'and not output.exists()and not output.with_suffix('.audit.json').exists()
root=R/f'balanced_context/fold{args.fold}/{args.kind}__{args.mode}'
complete=read(root/'complete.json');assert complete['status']=='complete'
for n,h in complete['files'].items():assert b.sha(root/n)==h,n
model=joblib.load(root/'kumo_model.joblib');source=Path(model['context_source'])
assert b.sha(source)==model['context_source_sha256']
assert b.sha(model['checkpoint'])==model['checkpoint_sha256']
context=pd.read_parquet(source)
assert b.ids_hash(context,model['context_local_rows'])==model['selected_ID_hash']
original,ix,q,_,_,_,_,frame,mature,models=inputs(args.fold)
query,table=stages(q,frame,mature,'maturity')
engine=Engine(model['checkpoint'],model['columns'],context)
raw=engine.fit_predict(model['context_local_rows'],query,model['backbone_seed'],cache_path=None)
saved=np.load(root/'raw_predictions.npz')
assert np.array_equal(saved['rows'],np.r_[ix['correction'],ix['validation']])
assert np.array_equal(raw,saved['model_probability'])
adjusted=expit(logit(np.clip(raw.astype(float),1e-6,1-1e-6))+logit(model['full_context_prevalence'])-logit(model['selected_prevalence']))
assert np.max(abs(adjusted-saved['prior_adjusted_probability']))<1e-15
head=joblib.load(root/'head.joblib');sp=joblib.load(complete['spline_source'])
assert b.sha(complete['spline_source'])==complete['spline_source_sha256']
pred=.75*b.corrected_probability(head,query.iloc[6000:],adjusted[6000:])[0]+.25*b.predict_spline(sp,table.iloc[ix['validation']])
stored=np.load(root/'validation.npz');error=float(np.max(abs(pred-stored['prediction'])))
assert error<2e-14
np.savez_compressed(output,query_rows=saved['rows'],raw_probability=raw,prior_adjusted_probability=adjusted,
                    validation_rows=ix['validation'],validation_prediction=pred)
b.dump(dict(status='passed',fold=args.fold,kind=args.kind,mode=args.mode,query_rows=20000,validation_rows=14000,
            raw_bitwise_equal=True,validation_max_error=error,canonical_source_sha256=b.sha(root/'kumo_model.joblib'),
            artifact_sha256=b.sha(output),source_sha256=b.sha(__file__)),output.with_suffix('.audit.json'))
