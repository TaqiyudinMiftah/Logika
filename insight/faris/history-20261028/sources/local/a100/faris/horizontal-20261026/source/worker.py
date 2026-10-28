"""One Kumo subspace/fold. Retain recipe and predictions, never fitted weights."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['SDM_CHUNK_MEMORY_FRACTION']='0.02'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
os.environ.setdefault('HF_HOME','/mnt/home/faris/logika-model-arena-20261025/storage/huggingface')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
from pathlib import Path
import argparse, datetime, gc, hashlib, importlib, json, time, traceback
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch,sdm
from features import Features

R=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def dump(x,p):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,default=str,allow_nan=False)+'\n');tmp.replace(p)

def run(a):
    start=time.time();root=R/f'runs/{a.size}{a.estimators}/{a.pack}/fold{a.fold}'
    root.mkdir(parents=True,exist_ok=False)
    try:
        plan=json.loads((R/'PLAN.json').read_text())
        assert R.is_relative_to(Path.home()),'Use only the signed-in user home directory'
        assert not (R/'STOP').exists()
        assert time.time()<plan['deadline_epoch']
        catalog=json.loads((R/'CATALOG.json').read_text())
        assert sha(R/'CATALOG.json')==plan['catalog_sha256']
        f=Features(a.fold);pack=next(p for p in catalog['packs'] if p['id']==a.pack)
        frame,dropped=f.pack(pack)
        rows=np.r_[f.ix['context'],f.ix['correction'],f.ix['validation']]
        ordered=frame.iloc[rows].reset_index(drop=True)
        types={}
        for c in ordered:
            if pd.api.types.is_numeric_dtype(ordered[c]):types[c]='numerical'
            else:
                vocabulary=set(ordered[c].iloc[:50000].dropna().astype(str))
                ordered[c]=ordered[c].astype('string').where(ordered[c].astype('string').isin(vocabulary))
                types[c]='categorical'
        torch.set_num_threads(2)
        assert torch.cuda.is_available()
        torch.cuda.set_per_process_memory_fraction(.90)
        torch.set_grad_enabled(False)
        checkpoint=Path(os.environ['HF_HOME'])/f'hub/models--nvidia--Kumo-Tabular/snapshots/b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f/{a.size}/classifier.pt'
        assert checkpoint.is_file(),checkpoint
        module=importlib.import_module(sdm.models.KumoTabular.__module__)
        def resolve(**kw):
            assert kw['repo_id']=='nvidia/Kumo-Tabular' and kw['filename']==f'{a.size}/classifier.pt'
            assert kw['revision']=='v1.0.1'
            return str(checkpoint)
        with patch.object(module,'download_checkpoint',resolve):
            model=sdm.models.KumoTabular(task='classification',size=a.size,device='cuda')
        table=sdm.TableTensor.from_pandas(ordered,stypes=types,device='cuda')
        target=sdm.TableTensor.from_pandas(pd.DataFrame({'target':f.ix['y_context'].astype(int)}),stypes={'target':'categorical'},device='cuda')
        dump({'status':'running','pack':a.pack,'fold':a.fold,'size':a.size,'num_estimators':a.estimators,
              'seed':2026+a.fold,'columns':list(ordered.columns),'dropped':dropped,
              'context_rows':50000,'correction_rows':6000,'validation_rows':14000,'outer_labels_loaded':False,
              'source_sha256':sha(__file__),'feature_source_sha256':sha(R/'source/features.py'),
              'catalog_sha256':sha(R/'CATALOG.json'),'plan_sha256':sha(R/'PLAN.json'),
              'packet_manifest_sha256':sha(R/f'packets/fold{a.fold}/manifest.json'),
              'checkpoint_sha256':sha(checkpoint),'torch_version':torch.__version__,
              'model_source_sha256':sha(module.__file__),'GPU':torch.cuda.get_device_name(0),
              'execution_host':__import__('socket').gethostname()},root/'START.json')
        predictions=[]
        with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):
            model.fit(table[:50000],target,num_estimators=a.estimators,estimator_batch_size=1,
                      generator=torch.Generator(device='cuda').manual_seed(2026+a.fold))
            for offset in range(50000,70000,512):
                pred=model.predict(table[offset:offset+512])[:,'1'].numerical.reshape(-1).float().cpu().numpy()
                predictions.append(pred)
            repeated=model.predict(table[50000:50512])[:,'1'].numerical.reshape(-1).float().cpu().numpy()
        raw=np.concatenate(predictions);error=float(np.max(abs(repeated-raw[:512])))
        assert error<1e-7 and len(raw)==20000 and np.isfinite(raw).all()
        assert raw.min()>=0 and raw.max()<=1
        np.savez_compressed(root/'prediction.npz',rows=rows[50000:],raw=raw)
        model.clear();del model,table,target;gc.collect();torch.cuda.empty_cache()
        dump({'status':'complete','pack':a.pack,'fold':a.fold,'size':a.size,'num_estimators':a.estimators,
              'seconds':time.time()-start,'replay_max_abs':error,'prediction_sha256':sha(root/'prediction.npz'),
              'fitted_weights_saved':False,'completed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()},root/'RESULT.json')
        print(json.dumps({'pack':a.pack,'fold':a.fold,'seconds':time.time()-start,'status':'complete'}),flush=True)
    except BaseException:
        dump({'error':traceback.format_exc(),'seconds':time.time()-start},root/'FAILURE.json');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pack',required=True);p.add_argument('--fold',type=int,required=True)
    p.add_argument('--size',choices=['small','medium','large'],default='small');p.add_argument('--estimators',type=int,default=2)
    run(p.parse_args())
