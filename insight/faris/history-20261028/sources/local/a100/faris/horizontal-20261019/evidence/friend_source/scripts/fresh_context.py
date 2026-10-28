"""Optional fresh GPU execution of this one fixed baseline; never auto-queued.

Loads the owned local backbone with SDM0.2.1's unchanged checkpoint loader.
Fresh FP16 context scores may differ numerically from the locked source cache.
"""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
from pathlib import Path
import argparse,importlib,json,gc
from unittest.mock import patch
import numpy as np
import pandas as pd
import joblib
from baseline import ROOT,sha,dump,scores,refit_correction,corrected_probability,predict_spline


def load_local_model(device):
 import sdm
 lock=json.loads((ROOT/'BASELINE_LOCK.json').read_text());path=ROOT/'checkpoints/kumo_large/classifier.pt'
 assert sha(path)==lock['Kumo']['checkpoint_sha256']
 module=importlib.import_module(sdm.models.KumoTabular.__module__)
 def resolve(**kwargs):
  assert kwargs['repo_id']=='nvidia/Kumo-Tabular' and kwargs['filename']=='large/classifier.pt' and kwargs['revision']=='v1.0.1'
  return str(path)
 # Only replace the path resolver during construction. The architecture,
 # torch.load(weights_only=True),and state-dict loader are the SDK's originals.
 with patch.object(module,'download_checkpoint',resolve):model=sdm.models.KumoTabular(task='classification',size='large',device=device)
 for p in model.parameters():p.requires_grad_(False)
 model.eval();return model


def tables(frame,columns):
 import sdm
 x=frame[columns].copy();types={}
 for c in columns:
  if pd.api.types.is_numeric_dtype(x[c]):x[c]=x[c].replace([np.inf,-np.inf],np.nan);types[c]='numerical'
  else:x[c]=x[c].astype('string');types[c]='categorical'
 return sdm.TableTensor.from_pandas(x,stypes=types,device='cuda')


def predict(model,x):
 import torch
 values=[]
 with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):
  for start in range(0,len(x),8192):values.append(model.predict(x[start:start+8192])[:,'1'].numerical.reshape(-1).float().cpu().numpy())
 return np.concatenate(values)


def run(fold_numbers,check_loader=False):
 import torch,sdm
 torch.set_num_threads(4)
 if check_loader:
  model=load_local_model('cpu');result=dict(status='PASS',checkpoint_loads_locally=True,all_parameters_frozen=all(not p.requires_grad for p in model.parameters()),parameters=sum(p.numel() for p in model.parameters()),GPU_used=False)
  dump(result,ROOT/'reports/LOCAL_CHECKPOINT_LOADER.json');print(json.dumps(result),flush=True);return
 free,total=torch.cuda.mem_get_info();assert free>=.35*total,'Insufficient free GPU memory for the fixed30% allocation cap'
 torch.cuda.set_per_process_memory_fraction(.30);os.environ['SDM_CHUNK_MEMORY_FRACTION']='0.05'
 train=pd.read_parquet(ROOT/'data/train_features.parquet');test=pd.read_parquet(ROOT/'data/test_features.parquet');definition=json.loads((ROOT/'provenance/definition.json').read_text())
 columns=definition['context_columns'];x=tables(train,columns);xt=tables(test,columns)
 yt=sdm.TableTensor.from_pandas(pd.DataFrame(dict(target=train.target.astype(int))),stypes={'target':'categorical'},device='cuda');model=load_local_model('cuda')
 for fold in fold_numbers:
  saved=ROOT/'artifacts'/f'fold{fold}';dest=ROOT/'fresh'/f'fold{fold}';dest.mkdir(parents=True,exist_ok=True)
  if (dest/'RESULT.json').exists():raise RuntimeError('Fresh fold already exists; preserve prior result before another run')
  ids=np.load(saved/'indices.npz');ci,hi,vi=[ids[k] for k in ['context','correction','validation']]
  before={n:p.detach().clone() for n,p in model.named_parameters()};torch.manual_seed(2026+fold)
  with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):model.fit(x[ci],yt[ci],num_estimators=8,estimator_batch_size=4,generator=torch.Generator(device='cuda').manual_seed(2026+fold))
  raw={kind:predict(model,q) for kind,q in [('holdout',x[hi]),('oof',x[vi]),('test',xt)]}
  assert all(torch.equal(p.detach(),before[n]) for n,p in model.named_parameters());del before
  corrector=refit_correction(definition,train.iloc[hi],raw['holdout']);joblib.dump(corrector,dest/'correction.joblib',compress=3)
  spline=joblib.load(saved/'spline.joblib');checks={}
  for kind,query in [('holdout',train.iloc[hi]),('oof',train.iloc[vi]),('test',test)]:
   pd.DataFrame(dict(application_id=query.application_id.to_numpy(),prediction=raw[kind])).to_parquet(dest/f'kumo_{kind}.parquet',index=False)
   reference=pd.read_parquet(saved/f'kumo_{kind}.parquet');checks[kind+'_max_abs_change']=float(np.max(abs(raw[kind]-reference.prediction)))
   if kind!='holdout':
    pc,delta=corrected_probability(corrector,query,raw[kind]);p=.75*pc+.25*predict_spline(spline,query)
    pd.DataFrame(dict(application_id=query.application_id.to_numpy(),prediction=p)).to_parquet(dest/f'combined_{kind}.parquet',index=False)
    if kind=='oof':metrics=scores(query.target.astype(int),p)
  result=dict(fold=fold,metrics=metrics,backbone_frozen_fraction=1.,weights_unchanged=True,source_comparison=checks,public_score=None,
   context_rows=len(ci),correction_rows=len(hi),validation_rows=len(vi),seed=2026+fold,physical_allocation_fraction=.30,SDM_CHUNK_MEMORY_FRACTION='0.05',
   artifacts={p.name:sha(p) for p in dest.iterdir() if p.is_file()})
  dump(result,dest/'RESULT.json');print(json.dumps(result),flush=True);model.clear();gc.collect();torch.cuda.empty_cache()
 if all((ROOT/'fresh'/f'fold{f}/RESULT.json').exists() for f in range(5)):
  sample=pd.read_csv(ROOT/'data/sample_submission.csv');pt=np.zeros(len(sample));oof=np.full(len(train),np.nan)
  for fold in range(5):
   dest=ROOT/'fresh'/f'fold{fold}';part=pd.read_parquet(dest/'combined_test.parquet');pt+=pd.Series(part.prediction.to_numpy(),index=part.application_id).reindex(sample.application_id).to_numpy()/5
   vi=np.load(ROOT/'artifacts'/f'fold{fold}/indices.npz')['validation'];oof[vi]=pd.read_parquet(dest/'combined_oof.parquet').prediction
  out=ROOT/'fresh/submission_fresh_context.csv';pd.DataFrame(dict(application_id=sample.application_id,default_probability=pt)).to_csv(out,index=False)
  dump(dict(metrics=scores(train.target.astype(int),oof),sha256=sha(out),public_score=None,locked_submission_overwritten=False),ROOT/'fresh/COMPLETE.json')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fold',action='append',type=int,choices=range(5));p.add_argument('--check-local-loader',action='store_true');a=p.parse_args();run(a.fold if a.fold is not None else list(range(5)),a.check_local_loader)
