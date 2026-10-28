"""Bounded exact fold replay using existing frozen feature and row artifacts."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['HF_HUB_OFFLINE']='1'
import argparse,datetime,json,time,traceback,sys,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from engine_runtime import predict,sha

EXP=Path('/mnt/home/faris/logika-faris-exploration');FRZ=Path('/mnt/home/faris/logika-frozen-shadow-20261026')

def ks(y,p):
    order=np.argsort(p,kind='mergesort');ys=y[order];ps=p[order]
    d=np.abs(np.cumsum(ys==1)/np.sum(ys==1)-np.cumsum(ys==0)/np.sum(ys==0));return float(np.max(d[np.r_[ps[:-1]!=ps[1:],True]]))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['fe_cov','fe_mat','champion90_s7731','shadow91','kumo56'],required=True);ap.add_argument('--fold',type=int,default=0);ap.add_argument('--out',required=True);a=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1','This pilot is authorized on physical GPU1 only'
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    line=[x for x in gpu.splitlines()if x.split(',')[0].strip()=='1'][0];_,mem,util=[int(x.strip())for x in line.split(',')];assert mem<256 and util<5,('GPU1 occupied',line)
    dest=Path(a.out);dest.mkdir(parents=True,exist_ok=False);f=a.fold;rows=roles=None
    try:
        if a.model in {'fe_cov','fe_mat'}:
            b=EXP/('coverage-information-fe-20261021'if a.model=='fe_cov'else'mature-mask-preserved-fe-20261021');d=b/f'data/fold{f}'
            c=pd.read_parquet(d/'context.parquet');q=pd.read_parquet(d/'query.parquet');ix=np.load(d/'indices.npz');rows=np.r_[ix['correction'],ix['validation']];cols=json.loads((b/'columns.json').read_text());seed=2026+f;engine='legacy_separate';refpath=b/f'runs/fold{f}/raw.npz';z=np.load(refpath);ref=z['prediction'];assert np.array_equal(rows,z['rows']);y=c.target.astype(int).to_numpy()
            assert q.application_id.to_numpy(str).tolist()==z['application_id'].tolist()
        elif a.model=='kumo56':
            j=f*6+5;d=EXP/f'nested-correction-20261021/data/job{j}';c=pd.read_parquet(d/'context.parquet');q=pd.read_parquet(d/'query.parquet');ix=np.load(d/'indices.npz');rows=ix['validation'];cols=json.loads((EXP/'nested-correction-20261021/columns.json').read_text());seed=2026+f;engine='legacy_separate';refpath=EXP/f'nested-correction-20261021/runs/job{j}/raw.npz';z=np.load(refpath);ref=z['prediction'];assert np.array_equal(rows,z['rows']);y=c.target.astype(int).to_numpy()
        else:
            d=FRZ/f'frozen/fold{f}';x=pd.read_parquet(d/'features.parquet');roles=np.load(d/'roles.npz');rows=np.r_[roles['correction'],roles['validation']];view='champion90'if a.model=='champion90_s7731'else'shadow91';cols=json.loads((d/'MANIFEST.json').read_text())['views'][view];c=x.iloc[roles['context']];q=x.iloc[rows];y=roles['y_context'];seed=(7731 if a.model=='champion90_s7731'else 2026)+f;engine='concat_context_vocab';refpath=FRZ/f'runs/kumo_large_{a.model}/fold{f}/prediction.npz';z=np.load(refpath);ref=z['raw'];assert np.array_equal(rows,z['rows'])if'rows'in z else True
        ck=Path('/mnt/home/faris/logika-model-arena-20261025/storage/huggingface/hub/models--nvidia--Kumo-Tabular/blobs/cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613');assert sha(ck)=='cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613'
        meta={'model':a.model,'fold':f,'engine':engine,'seed':seed,'context_rows':len(c),'query_rows':len(q),'columns':cols,'source_sha256':{p.name:sha(p)for p in [Path(__file__),Path(__file__).with_name('engine_runtime.py')]},'reference_sha256':sha(refpath),'rows_sha256':__import__('hashlib').sha256(rows.tobytes()).hexdigest(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fitted_weights_serialized':False,'KV_cache_serialized':False}
        (dest/'START.json').write_text(json.dumps(meta,indent=2)+'\n')
        raw,run=predict(c,q,cols,y,ck,seed,engine=engine);np.savez_compressed(dest/'prediction.npz',rows=rows,prediction=raw)
        err=float(np.max(np.abs(raw-ref)));result={**meta,**run,'max_abs_prediction_error':err,'mean_abs_prediction_error':float(np.mean(np.abs(raw-ref))),'same_backend_max_abs_gate':1e-5,'prediction_gate_passed':err<=1e-5,'prediction_sha256':sha(dest/'prediction.npz')}
        # Existing globally exposed labels are opened only after inference for replay metrics.
        allref=np.load(EXP/'transfer-20261025/ROBUSTNESS_OOF.npz');labels=allref['target'][rows];result.update(reference_KS=ks(labels,ref),replay_KS=ks(labels,raw));result['abs_KS_error']=abs(result['reference_KS']-result['replay_KS']);result['passed']=result['prediction_gate_passed']and result['abs_KS_error']<=1e-5
        (dest/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);assert result['passed'],'Replay gate failed; do not admit dependent production'
    except Exception as e:
        (dest/'FAILURE.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc(),'resource_failure_is_not_predictive_evidence':True},indent=2)+'\n');raise
if __name__=='__main__':main()
