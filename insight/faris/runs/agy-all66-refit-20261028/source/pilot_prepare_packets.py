"""Build two original fold0 portability packets without query labels or fitting."""
from pathlib import Path
import hashlib,json,tarfile
import numpy as np
import pandas as pd
from engine_runtime import sha
R=Path('/mnt/home/faris/logika-antigravity-refit-20261028');EXP=Path('/mnt/home/faris/logika-faris-exploration');FRZ=Path('/mnt/home/faris/logika-frozen-shadow-20261026')
configs=json.loads((R/'ENGINE_CONFIGS.json').read_text())['models'];root=R/'pilot_packets';assert not root.exists();root.mkdir()
checks=[]
for model in ['fe_cov','champ_s7731']:
    spec=configs[model];cols=spec['columns'];dest=root/model/'fold0';dest.mkdir(parents=True)
    if model=='fe_cov':
        data=EXP/'coverage-information-fe-20261021/data/fold0';cpath=data/'context.parquet';qpath=data/'query.parquet';c0=pd.read_parquet(cpath);q0=pd.read_parquet(qpath);ix=np.load(data/'indices.npz');ctxrows=ix['context'];queryrows=np.r_[ix['correction'],ix['validation']];y=c0.target.astype(int).to_numpy();refpath=EXP/'coverage-information-fe-20261021/runs/fold0/raw.npz';ref=np.load(refpath);pred=ref['prediction'];assert np.array_equal(queryrows,ref['rows']);paths=[cpath,qpath,data/'indices.npz']
    else:
        data=FRZ/'frozen/fold0';p=data/'features.parquet';x=pd.read_parquet(p);ix=np.load(data/'roles.npz');ctxrows=ix['context'];queryrows=np.r_[ix['correction'],ix['validation']];c0=x.iloc[ctxrows];q0=x.iloc[queryrows];y=ix['y_context'];refpath=FRZ/'runs/kumo_large_champion90_s7731/fold0/prediction.npz';ref=np.load(refpath);pred=ref['raw'];paths=[p,data/'roles.npz',data/'MANIFEST.json']
    context=c0[cols].reset_index(drop=True).copy();query=q0[cols].reset_index(drop=True).copy();context['target']=y
    assert not np.intersect1d(ctxrows,queryrows).size and len(ctxrows)==50000 and len(queryrows)==20000
    context.to_parquet(dest/'context.parquet',index=False);query.to_parquet(dest/'query.parquet',index=False)
    pd.testing.assert_frame_equal(context,pd.read_parquet(dest/'context.parquet'),check_exact=True);pd.testing.assert_frame_equal(query,pd.read_parquet(dest/'query.parquet'),check_exact=True)
    np.save(dest/'context_rows.npy',ctxrows);np.save(dest/'query_rows.npy',queryrows);np.savez_compressed(dest/'reference.npz',rows=queryrows,prediction=pred)
    meta={'model':model,'fold':0,'engine':spec['engine'],'seed':spec['seed_base'],'columns':cols,'n_context':50000,'num_estimators':8,'purpose':'Predeclared A100 crossbackend portability replay, no search','query_labels_staged':False,'historical_source_files_sha256':{str(p):sha(p)for p in paths},'historical_prediction_source':str(refpath),'historical_prediction_sha256':sha(refpath),'feature_rows_verified_equal_before_and_after_parquet':True,'cross_backend_tolerance':{'max_abs_prediction_error':0.002,'mean_abs_prediction_error':0.0002,'abs_KS_error':0.001},'source_sha256':sha(__file__)}
    (dest/'meta.json').write_text(json.dumps(meta,indent=2)+'\n');checks.append({'packet':str(dest.relative_to(root)),'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)}for p in dest.iterdir()if p.is_file()}})
(root/'MANIFEST.json').write_text(json.dumps({'packets':checks,'query_labels_included':False,'fitted_weights_or_KV_included':False},indent=2)+'\n')
archive=R/'pilot_packets.tar.gz'
with tarfile.open(archive,'w:gz')as tar:tar.add(root,arcname='pilot_packets')
receipt={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'packets':2,'query_labels_included':False}
(R/'PILOT_PACKET_BUNDLE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
