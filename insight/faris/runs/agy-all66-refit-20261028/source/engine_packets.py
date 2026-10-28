"""Portable production runner preserving declared historical Engine semantics.

Each packet: context.parquet (features + target), query.parquet (features, no
labels), query_rows.npy, meta.json (columns, engine, seed, model, fold).
Only compact predictions and source/parameter evidence are serialized.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse,datetime,hashlib,json,time,traceback,sys,platform,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from engine_runtime import predict,sha,LARGE_SHA

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--packets',required=True);ap.add_argument('--out',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--only',nargs='*');ap.add_argument('--mem-fraction',type=float,default=None);a=ap.parse_args()
    root=Path(a.packets);out=Path(a.out);assert sha(a.checkpoint)==LARGE_SHA
    paths=[root/p for p in a.only]if a.only else sorted(p.parent for p in root.rglob('meta.json'))
    for p in paths:
        rel=p.relative_to(root);dest=out/rel
        if(dest/'COMPLETE.json').exists():
            r=json.loads((dest/'COMPLETE.json').read_text());assert sha(dest/'prediction.npz')==r['prediction_sha256'];continue
        assert not dest.exists(),f'Preserve incomplete attempt evidence; use a new output root: {dest}'
        dest.mkdir(parents=True);start=time.time()
        try:
            meta=json.loads((p/'meta.json').read_text())
            for field,name in [('context_sha256','context.parquet'),('query_sha256','query.parquet'),('query_rows_sha256','query_rows.npy')]:
                if field in meta:assert sha(p/name)==meta[field],('Changed declared packet',field,str(p))
            context=pd.read_parquet(p/'context.parquet');query=pd.read_parquet(p/'query.parquet');rows=np.load(p/'query_rows.npy').astype(np.int64)
            assert 'target'in context and'target'not in query and len(query)==len(rows)
            assert len(np.unique(rows))==len(rows)
            assert len(context)==meta['n_context']
            assert not{'target','application_id','customer_id','split'}&set(meta['columns'])
            if 'application_id'in context and'application_id'in query:assert set(context.application_id).isdisjoint(query.application_id)
            evidence={'packet':str(rel),'metadata':meta,'started_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'files_sha256':{name:sha(p/name)for name in ['meta.json','context.parquet','query.parquet','query_rows.npy']},
                      'source_sha256':{x.name:sha(x)for x in [Path(__file__),Path(__file__).with_name('engine_runtime.py')]},
                      'python':sys.version,'platform':platform.platform(),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
                      'fitted_weights_serialized':False,'KV_caches_serialized':False}
            (dest/'START.json').write_text(json.dumps(evidence,indent=2)+'\n')
            raw,run=predict(context,query,meta['columns'],context.target.astype(int).to_numpy(),a.checkpoint,meta['seed'],engine=meta['engine'],estimators=meta.get('num_estimators',8),mem_fraction=a.mem_fraction)
            np.savez_compressed(dest/'prediction.partial.npz',rows=rows,prediction=raw);(dest/'prediction.partial.npz').replace(dest/'prediction.npz')
            import torch
            evidence.update(run);evidence.update(status='complete',seconds=time.time()-start,query_rows=len(rows),mean_probability=float(raw.mean()),prediction_sha256=sha(dest/'prediction.npz'),cuda_runtime=torch.version.cuda,GPU=torch.cuda.get_device_name(0),finished_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
            (dest/'COMPLETE.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps({'packet':str(rel),'status':'complete','seconds':time.time()-start,'mean_probability':float(raw.mean())}),flush=True)
        except Exception as e:
            (dest/'FAILURE.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc(),'seconds':time.time()-start,'resource_failure_is_not_predictive_evidence':True},indent=2)+'\n');raise
if __name__=='__main__':main()
