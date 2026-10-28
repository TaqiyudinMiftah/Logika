"""Refit the inherited fixed shadow/latent feature models on the new64kcontext.

No feature selection. Historical borrower3fold cross-fitting, 35epoch Weibull,
400tree CatBoost and 40epoch variational teacher preserve their prior recipes.
Only2CPUthreads replace3 in the oldshadowhelper as an execution resource limit.
No fitted weights or caches are serialized.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,gc,hashlib,importlib.util,json,sys,time,types
from pathlib import Path
import numpy as np,pandas as pd
import torch
from torch import nn
from sklearn.model_selection import KFold
R=Path(__file__).resolve().parents[1];FRZ=Path('/mnt/home/faris/logika-frozen-shadow-20261026')
sys.path.insert(0,str(FRZ/'source'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def inherited(name):
 path=FRZ/'source'/name;code=path.read_text().replace('torch.set_num_threads(3)','torch.set_num_threads(2)').replace('thread_count=3','thread_count=2');module=types.ModuleType('inherited_'+path.stem);module.__file__=str(path);exec(compile(code,str(path),'exec'),module.__dict__);return module

def shadow(fold):
 root=R/f'features/fold{fold}';dest=root/'teachers';dest.mkdir(parents=True,exist_ok=True)
 if(dest/'SHADOW.json').exists():return
 old=inherited('shadow_features.py');h=pd.read_parquet(FRZ/'prepared/historical.parquet');current=pd.read_parquet(FRZ/'prepared/current.parquet');roles=np.load(root/'roles.npz');context=roles['context'];query=roles['query'];customers=current.customer_id.iloc[context].to_numpy();features=json.loads((FRZ/'SHADOW_DATA_GATE.json').read_text())['teacher_features']
 allowed=h.schedule_complete|h.event;out=np.full((100000,2),np.nan);recipes=[];start=time.time()
 for split,(ti,vi)in enumerate(KFold(3,shuffle=True,random_state=261026).split(context)):
  fitcustomers=set(customers[ti]);validcustomers=set(customers[vi]);assert not fitcustomers&validcustomers
  fitrows=np.flatnonzero(h.customer_id.isin(fitcustomers)&allowed);predictrows=context[vi];seed=261026+fold*10+split
  dump(dest/f'SHADOW_FIT{split}_PLAN.json',{'fold':fold,'split':split,'seed':seed,'historical_fit_rows':len(fitrows),'fit_customers':len(fitcustomers),'query_customers':len(validcustomers),'fit_borrower_IDs_sha256':hashlib.sha256('\n'.join(sorted(fitcustomers)).encode()).hexdigest(),'original_source_sha256':sha(FRZ/'source/shadow_features.py'),'wrapper_sha256':sha(__file__),'threads':2,'fitted_weights_saved':False})
  p,c,recipe=old.fit_predict(h,current,fitrows,predictrows,features,seed);out[predictrows]=np.column_stack([p,c]);recipes.append({**recipe,'split':split,'excluded_query_customers':len(validcustomers)});print(json.dumps({'stage':'shadow_inner_fit_done','fold':fold,'split':split}),flush=True);gc.collect()
 fitrows=np.flatnonzero(h.customer_id.isin(set(customers))&allowed);assert not set(current.customer_id.iloc[query])&set(h.customer_id.iloc[fitrows]);seed=261026+fold*10+3
 dump(dest/'SHADOW_FULL_PLAN.json',{'fold':fold,'seed':seed,'context_rows':len(context),'historical_fit_rows':len(fitrows),'query_rows':len(query),'roles_sha256':sha(root/'roles.npz'),'original_source_sha256':sha(FRZ/'source/shadow_features.py'),'wrapper_sha256':sha(__file__),'threads':2,'fitted_weights_saved':False})
 p,c,recipe=old.fit_predict(h,current,fitrows,query,features,seed);out[query]=np.column_stack([p,c]);recipes.append({**recipe,'split':'full_context'})
 assert np.isfinite(out).all()and out.min()>=0 and out.max()<=1
 pd.DataFrame({'shadow_interval12':out[:,0],'shadow_known12':out[:,1]}).to_parquet(dest/'shadow.parquet',index=False)
 dump(dest/'SHADOW.json',{'status':'complete','fold':fold,'context_rows':64000,'query_rows':36000,'historical_label_fit_boundary':'Contextborrowers only,3borrowerfoldcrossfitwithincontext; fullcontext teacher forcorrection/test','selected_target_labels_used':False,'original_source_sha256':sha(FRZ/'source/shadow_features.py'),'wrapper_sha256':sha(__file__),'execution_change':'CPUthreads3->2; immutable scientificparameters preserved','recipes':recipes,'prediction_sha256':sha(dest/'shadow.parquet'),'fitted_weights_saved':False,'seconds':time.time()-start});print(json.dumps({'stage':'shadow_done','fold':fold}),flush=True)

def latent(fold):
 root=R/f'features/fold{fold}';dest=root/'teachers';dest.mkdir(parents=True,exist_ok=True)
 if(dest/'LATENT.json').exists():return
 start=time.time();old=inherited('latent_features.py');seed=262026+fold;torch.manual_seed(seed);torch.set_num_threads(2)
 frame=pd.read_parquet(root/'champion_current.parquet');roles=np.load(root/'roles.npz');ctx=roles['context'];columns=json.loads((FRZ/f'learned/fold{fold}/LATENT.json').read_text())['columns'];raw=frame[columns].to_numpy(float);raw=np.sign(raw)*np.log1p(abs(raw));mask=np.isfinite(raw)
 mean=np.nan_to_num(np.nanmean(raw[ctx],0));sd=np.maximum(np.nan_to_num(np.nanstd(raw[ctx],0),nan=1),.05);x=np.clip((np.where(mask,raw,mean)-mean)/sd,-6,6).astype('float32');mask=mask.astype('float32')
 def view(c):
  if'bureau'in c:return 0
  if any(s in c for s in ['repay','late_','allocation','loan_count','dpd_']):return 1
  if any(s in c for s in ['ecom','price','orders']):return 2
  return 3
 views=[[j for j,c in enumerate(columns)if view(c)==k]for k in range(4)];assert all(views)
 labels=pd.read_parquet(FRZ/'prepared/current.parquet',columns=['customer_id']);historical=pd.read_parquet(FRZ/'prepared/historical.parquet',columns=['customer_id','target12']);known=historical[historical.target12>=0].groupby('customer_id').target12.max();aux=labels.customer_id.map(known).to_numpy(float)
 A=Path('/mnt/home/faris/logika-faris-exploration');y=pd.read_parquet(A/'ranking-pipeline-20261020/input/data/train_features.parquet',columns=['target']).target.to_numpy(int)
 dump(dest/'LATENT_PLAN.json',{'seed':seed,'context_rows':len(ctx),'roles_sha256':sha(root/'roles.npz'),'features_sha256':sha(root/'champion_current.parquet'),'original_source_sha256':sha(FRZ/'source/latent_features.py'),'wrapper_sha256':sha(__file__),'epochs':40,'columns':columns,'device':'cpu','threads':2,'labels_used':'Only64k context selecteddefault labels and historical target12 fromsamecontextborrowers','correction_test_labels_used':False,'fitted_weights_saved':False})
 model=old.Joint(views);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);tx=torch.tensor(x[ctx]);tm=torch.tensor(mask[ctx]);ty=torch.tensor(y[ctx],dtype=torch.float32);th=torch.tensor(np.nan_to_num(aux[ctx]),dtype=torch.float32);available=torch.tensor(np.isfinite(aux[ctx]));rng=np.random.default_rng(seed);trace=[]
 for epoch in range(40):
  model.train();total=0;order=rng.permutation(len(ctx))
  for first in range(0,len(ctx),1024):
   ids=order[first:first+1024];mu,var,rec,p,h=model(tx[ids],tm[ids]);a=available[ids];history_loss=nn.functional.binary_cross_entropy_with_logits(h[a],th[ids][a])if a.any()else h.sum()*0
   loss=rec+.25*nn.functional.binary_cross_entropy_with_logits(p,ty[ids])+.25*history_loss;assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();total+=float(loss.detach())
  trace.append(total/int(np.ceil(len(ctx)/1024)))
 model.eval();representations=[]
 with torch.inference_mode():
  for first in range(0,100000,2048):
   mu,var=model.encode(torch.tensor(x[first:first+2048]),torch.tensor(mask[first:first+2048]));representations.append(np.column_stack([mu.numpy(),var.mean(1).numpy()]))
 features=np.concatenate(representations);assert features.shape==(100000,9)and np.isfinite(features).all()
 pd.DataFrame(features,columns=[f'joint_z{j}'for j in range(8)]+['joint_posterior_variance']).to_parquet(dest/'latent.parquet',index=False)
 dump(dest/'LATENT.json',{'status':'complete','fold':fold,'context_rows':64000,'seed':seed,'epochs':40,'device':'cpu','original_source_sha256':sha(FRZ/'source/latent_features.py'),'wrapper_sha256':sha(__file__),'loss_first_last':[trace[0],trace[-1]],'prediction_sha256':sha(dest/'latent.parquet'),'fitted_weights_saved':False,'seconds':time.time()-start});print(json.dumps({'stage':'latent_done','fold':fold}),flush=True)

def finish(fold):
 root=R/f'features/fold{fold}';frame=pd.read_parquet(root/'expanded_base.parquet')
 for name in ['shadow','latent']:
  d=pd.read_parquet(root/f'teachers/{name}.parquet');assert len(d)==100000
  for c in d:frame[c]=d[c].to_numpy()
 import pyarrow.parquet as pq
 old_columns=pq.read_schema('/mnt/home/faris/logika-joint-20261028/fold'+str(fold)+'/expanded.parquet').names
 spectral_columns=[c for c in frame if c.startswith('beh_spectral_')]
 wanted=['application_id','customer_id']+old_columns+[c for c in spectral_columns if c not in old_columns]
 assert set(frame.columns)==set(wanted),(set(frame)-set(wanted),set(wanted)-set(frame))
 frame=frame[wanted]
 assert 'target'not in frame and frame.application_id.is_unique
 frame.to_parquet(root/'expanded.tmp.parquet',index=False);(root/'expanded.tmp.parquet').replace(root/'expanded.parquet')
 # Shadow91requires only this inheritedhistoricalteacher, not any newviewselection.
 config=json.loads((R/'ENGINE_CONFIGS.json').read_text())['models']['shadow91'];roles=np.load(root/'roles.npz');ctx=roles['context'];query_rows=roles['query'];columns=config['columns'];packet=R/f'packets/shadow91/fold{fold}';packet.mkdir(parents=True,exist_ok=True)
 A=Path('/mnt/home/faris/logika-faris-exploration');y=pd.read_parquet(A/'ranking-pipeline-20261020/input/data/train_features.parquet',columns=['target']).target.to_numpy(int)
 context=frame.iloc[ctx][columns].reset_index(drop=True).copy();context['target']=y[ctx];query=frame.iloc[query_rows][columns].reset_index(drop=True).copy();context.to_parquet(packet/'context.parquet',index=False);query.to_parquet(packet/'query.parquet',index=False);np.save(packet/'context_rows.npy',ctx);np.save(packet/'query_rows.npy',query_rows)
 dump(packet/'meta.json',{'model':'shadow91','fold':fold,'engine':config['engine'],'seed':config['seed_base']+fold*config['fold_seed_increment'],'num_estimators':8,'columns':columns,'n_context':len(ctx),'n_query':len(query_rows),'context_sha256':sha(packet/'context.parquet'),'query_sha256':sha(packet/'query.parquet'),'context_rows_sha256':sha(packet/'context_rows.npy'),'query_rows_sha256':sha(packet/'query_rows.npy'),'fitted_weights_saved':False})
 dump(root/'COMPLETE.json',{'status':'complete','fold':fold,'rows':100000,'columns':list(frame),'expanded_sha256':sha(root/'expanded.parquet'),'base_manifest_sha256':sha(root/'BASE_MANIFEST.json'),'shadow_sha256':sha(root/'teachers/SHADOW.json'),'latent_sha256':sha(root/'teachers/LATENT.json'),'fitted_weights_saved':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);p.add_argument('--kind',choices=['shadow','latent','all'],default='all');a=p.parse_args()
 if a.kind in['shadow','all']:shadow(a.fold)
 if a.kind in['latent','all']:latent(a.fold)
 if a.kind=='all':finish(a.fold)
