from pathlib import Path
import shutil,os,json,hashlib,subprocess,concurrent.futures,time
import numpy as np,pandas as pd
R=Path('/mnt/home/faris/logika-final-three-20261030');OLD=Path('/mnt/home/faris/logika-midnight-multiseed-20261030');E=R/'ecommerce';E.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=np.load(R/'ECOMMERCE_FEATURES.npz');plan=json.loads((R/'ECOMMERCE_PLAN.json').read_text());assert sha(R/'ECOMMERCE_FEATURES.npz')==plan['features_npz_sha256'];assert q['values'].shape==(100000,8)
tr=pd.read_csv('/mnt/home/faris/logika-twelvehour-next-20261029/supplied_data/train_applications.csv',usecols=['application_id']);ids=np.r_[tr.application_id.astype(str).to_numpy(),np.load(OLD/'packets/fold0/test_ids.npy').astype(str)];assert len(set(ids))==100000
features=pd.DataFrame(q['values'],index=q['application_id'].astype(str),columns=q['columns'].astype(str)).reindex(ids).reset_index(drop=True);assert not features.isna().all(axis=1).any()
for n in ['CONFIG.json','CLOCK.json','PREPUBLICATION_SOURCE.json','compose_final.py','sample_submission.csv']:shutil.copy2(R/n,E/n)
shutil.copytree(OLD/'source',E/'source');worker=(OLD/'worker.py').read_text();worker=worker.replace("'global99'if kind=='kumo70'else'actuarial99'","'global99'if kind=='kumo70'else'actuarial_ecommerce'").replace('len(cols)==99','len(cols)==107').replace('100000),99)','100000),107)');assert 'len(cols)==107'in worker and '100000),107)'in worker
(E/'worker.py').write_text(worker);c=json.loads((E/'CONFIG.json').read_text());c.update(id='final-three-ecommerce-seed2026',new_seeds=[2026],kinds=['localglm'],features=plan);(E/'CONFIG.json').write_text(json.dumps(c,indent=2)+'\n')
for fold in range(5):
    p=E/f'packets/fold{fold}';p.mkdir(parents=True);base=OLD/f'packets/fold{fold}';m=json.loads((base/'MANIFEST.json').read_text());cols=m['views']['actuarial99'];newcols=cols+list(features)
    for phase in ['validation','production']:
        name='localglm_validation.parquet' if phase=='validation' else'features.parquet';f=pd.read_parquet(base/name,columns=cols);n=len(f)
        for col in features:f[col]=features[col].to_numpy()[:n]
        assert f.shape==(n,107);f.to_parquet(p/name,index=False)
    for n in ['roles.npz','y.npy','test_ids.npy','anchor.npz','anchor_test.npy']:os.link(base/n,p/n)
    m['views']['actuarial_ecommerce']=newcols;m['files']={n:sha(p/n)for n in ['localglm_validation.parquet','features.parquet','roles.npz','y.npy','test_ids.npy','anchor.npz','anchor_test.npy']};m['ecommerce_plan_sha256']=sha(R/'ECOMMERCE_PLAN.json');(p/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
seal={str(p.relative_to(E)):sha(p)for p in [E/'worker.py',E/'compose_final.py',*(E/'source').rglob('*.py')]};(E/'SOURCE_SEAL.json').write_text(json.dumps(seal,indent=2)+'\n');(E/'FEATURE_READY.json').write_text(json.dumps(dict(epoch=time.time(),new_fields=list(features),source_seal_sha256=sha(E/'SOURCE_SEAL.json'),row_alignment_by_application_ID=True,no_targets_in_new_features=True))+'\n')
PY='/mnt/home/faris/logika-model-arena-20261025/env311/bin/python'
def pair(fold):
    for phase in ['validation','production']:
        if time.time()>=json.loads((E/'CLOCK.json').read_text())['admission_end_epoch']:return
        log=E/f'fold{fold}_{phase}.log'
        with log.open('ab')as f:q=subprocess.run([PY,'-u',str(E/'worker.py'),'--kind','localglm','--seed','2026','--fold',str(fold),'--phase',phase],stdout=f,stderr=subprocess.STDOUT,timeout=max(1,json.loads((E/'CLOCK.json').read_text())['model_deadline_epoch']-time.time()))
        if q.returncode:return
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as ex:list(ex.map(pair,range(5)))
(E/'FINAL.json').write_text(json.dumps(dict(epoch=time.time(),no_more_experiments=True))+'\n');print('Ecommerce five-fold validation/production complete',flush=True)
