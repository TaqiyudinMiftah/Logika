"""Exact historical Kumo inference patterns, without fitted weight/cache serialization.

legacy_separate: Engine in coverage/mature/nested/transfer gpu_backend.py.
concat_context_vocab: frozen-shadow model_worker and new specialist helper.
Their categorical encoding and infinity policies are intentionally distinct.
"""
import os
os.environ.setdefault('SDM_CHUNK_MEMORY_FRACTION','0.02')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import gc,hashlib,importlib,json,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

LARGE_SHA='cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def make_model(checkpoint,device='cuda'):
    import sdm
    assert sha(checkpoint)==LARGE_SHA,'Wrong pretrained checkpoint'
    module=importlib.import_module(sdm.models.KumoTabular.__module__)
    def resolve(**kw):
        assert kw['repo_id']=='nvidia/Kumo-Tabular'and kw['filename']=='large/classifier.pt'and kw['revision']=='v1.0.1'
        return str(Path(checkpoint).resolve())
    with patch.object(module,'download_checkpoint',resolve):
        return sdm.models.KumoTabular(task='classification',size='large',device=device)

def predict(context,query,columns,y_context,checkpoint,seed,engine='legacy_separate',estimators=8,pred_batch=512,mem_fraction=None):
    import torch,sdm
    assert torch.cuda.is_available(),'CPU Kumo is not admitted'
    assert engine in {'legacy_separate','concat_context_vocab'}
    assert len(context)==len(y_context)and len(query)>0 and len(columns)<=99
    assert not {'target','application_id','customer_id','split'}&set(columns)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(mem_fraction or(.90 if engine=='legacy_separate'else .85))
    device='cuda';n=len(context)
    if engine=='legacy_separate':
        def table(frame):
            x=frame[columns].copy();types={}
            for c in columns:
                if pd.api.types.is_numeric_dtype(x[c]):x[c]=x[c].replace([np.inf,-np.inf],np.nan);types[c]='numerical'
                else:x[c]=x[c].astype('string');types[c]='categorical'
            return sdm.TableTensor.from_pandas(x,stypes=types,device=device)
        xt=table(context);xq=table(query)
    else:
        ordered=pd.concat([context[columns],query[columns]],ignore_index=True).copy();types={}
        for c in columns:
            if pd.api.types.is_numeric_dtype(ordered[c]):types[c]='numerical'
            else:
                vocabulary=set(ordered[c].iloc[:n].dropna().astype(str))
                ordered[c]=ordered[c].astype('string').where(ordered[c].astype('string').isin(vocabulary));types[c]='categorical'
        allx=sdm.TableTensor.from_pandas(ordered,stypes=types,device=device);xt=allx[:n];xq=allx[n:]
    yt=sdm.TableTensor.from_pandas(pd.DataFrame({'target':np.asarray(y_context).astype(int)}),stypes={'target':'categorical'},device=device)
    model=make_model(checkpoint);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    pred=[];start=time.time()
    with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):
        model.fit(xt,yt,num_estimators=int(estimators),estimator_batch_size=1,generator=torch.Generator(device='cuda').manual_seed(int(seed)))
        print(json.dumps({'stage':'context_fitted','rows':n,'seconds':time.time()-start}),flush=True)
        for s in range(0,len(query),pred_batch):
            pred.append(model.predict(xq[s:s+pred_batch])[:,'1'].numerical.reshape(-1).float().cpu().numpy())
            if len(pred)==1 or len(pred)%10==0:print(json.dumps({'stage':'predicting','query_done':min(s+pred_batch,len(query)),'seconds':time.time()-start}),flush=True)
        replay=model.predict(xq[:min(pred_batch,len(query))])[:,'1'].numerical.reshape(-1).float().cpu().numpy()
    raw=np.concatenate(pred).astype(np.float32);assert np.isfinite(raw).all()and len(raw)==len(query)
    error=float(np.max(np.abs(raw[:len(replay)]-replay)));assert error<1e-7,('same-cache prediction nondeterminism',error)
    model.clear();del model,xt,xq,yt
    if engine!='legacy_separate':del allx
    gc.collect();torch.cuda.empty_cache()
    return raw,{'engine':engine,'seed':int(seed),'num_estimators':int(estimators),'estimator_batch_size':1,'prediction_batch':pred_batch,'precision':'float16 AMP','generator':'CUDA','first_batch_replay_error':error,'seconds':time.time()-start,'torch':torch.__version__,'sdm':getattr(sdm,'__version__','unknown'),'pretrained_checkpoint_sha256':LARGE_SHA}
