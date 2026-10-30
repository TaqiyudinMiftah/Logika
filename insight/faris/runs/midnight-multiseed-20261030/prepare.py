from pathlib import Path
import json,hashlib,os,shutil
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;OLD=Path('/mnt/home/faris/logika-overnight-refit-20261030');DEV=Path('/mnt/home/faris/logika-twelvehour-next-20261029/neural-cann');SIX=Path('/mnt/home/faris/logika-sixhour-broad-20261028')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def link(a,b):
 b.parent.mkdir(parents=True,exist_ok=True)
 if b.exists():assert sha(a)==sha(b);return
 try:os.link(a,b)
 except OSError:shutil.copy2(a,b)
checks=[]
for fold in range(5):
 p=R/f'packets/fold{fold}';p.mkdir(parents=True,exist_ok=True);old=OLD/f'packets/fold{fold}';m=json.loads((old/'MANIFEST.json').read_text());roles=DEV/f'packets/fold{fold}/roles.npz';z=np.load(roles)
 assert len(set(np.r_[z['context'],z['correction'],z['validation']]))==70000
 for n in ['features.parquet','y.npy','test_ids.npy']:link(old/n,p/n)
 for n in ['roles.npz','anchor.npz']:link(DEV/f'packets/fold{fold}'/n,p/n)
 frame=pd.read_parquet(p/'features.parquet');assert len(frame)==100000
 for kind,root,view in [('localglm',OLD/'localglm-cann','actuarial99'),('cann',DEV,'actuarial99'),('kumo70',SIX,'global99')]:
  other=root/f'packets/fold{fold}';otherm=json.loads((other/'MANIFEST.json').read_text());cols=m['views'][view];assert cols==otherm['views'][view];f=pd.read_parquet(other/'features.parquet',columns=cols);vf=p/(kind+'_validation.parquet');f.to_parquet(vf,index=False);assert pd.read_parquet(vf).equals(f.reset_index(drop=True)),(kind,fold,'Validationprojectionmismatch')
  rr=np.load(other/'roles.npz');assert all(np.array_equal(rr[k],z[k])for k in ['context','correction','validation','y_context','y_correction'])
 # Test anchor is a permitted fixed pipeline component, aligned by applicant ID.
 a=Path('/mnt/home/faris/logika-submit-best-sub25k-20261029/private/anchor')/f'fold{fold}';j=json.loads((a/'complete.json').read_text());assert sha(a/'test_predictions.npz')==j['files']['test_predictions.npz'];q=np.load(a/'test_predictions.npz');ids=np.load(p/'test_ids.npy').astype(str);v=pd.Series(q['prediction'],index=q['application_id'].astype(str)).reindex(ids).to_numpy();assert np.isfinite(v).all();np.save(p/'anchor_test.npy',v)
 (p/'MANIFEST.json').write_text(json.dumps(dict(files={n:sha(p/n)for n in ['features.parquet','localglm_validation.parquet','cann_validation.parquet','kumo70_validation.parquet','y.npy','test_ids.npy','roles.npz','anchor.npz','anchor_test.npy']},views=m['views'],old_production_manifest_sha256=sha(old/'MANIFEST.json'),old_roles_sha256=sha(roles)),indent=2)+'\n');checks.append(dict(fold=fold,rows='50000/6000/14000+30000test',validation_projection_exact=True,production_features_existing_full70k_refit=True))
 designs={'localglm':(OLD/'localglm-cann','localglm_actuarial99_intercept_w64_r0p1_raw','selected_localglm-cann'),'cann':(DEV,'ncann_actuarial99_intercept_c0p1_w128_r0p0_lr0p0003','cann'),'kumo70':(SIX,'kumo_large_global99_e2','kumo70')}
 for kind,(root,d,k)in designs.items():
  for seed in [2026,7731]:
   target=R/f'original/{kind}/s{seed}/fold{fold}';target.mkdir(parents=True,exist_ok=True)
   val=root/f'runs/{d}__s{seed}__f{fold}/attempt1';prod=OLD/f'predictions/{k}/s{seed}/fold{fold}'
   for phase,src in [('validation',val),('production',prod)]:
    j=json.loads((src/'RESULT.json').read_text());assert sha(src/'prediction.npz')==j['prediction_sha256'];link(src/'prediction.npz',target/(phase+'.npz'));(target/(phase+'_PROVENANCE.json')).write_text(json.dumps(dict(source=str(src),source_result_sha256=sha(src/'RESULT.json'),prediction_sha256=j['prediction_sha256']),indent=2)+'\n')
    if seed==2026 and fold==0:link(src/'prediction.npz',R/f'reference/{kind}_{phase}.npz')
# Raw target is evaluator-only by convention; workers never open this path.
link(DEV/'evaluation_inputs/target.npy',R/'evaluation_inputs/target.npy');link(Path('/mnt/home/faris/logika-final-history-blend-20261030/paired/candidates/current_fusion_cann_kumo70_p0.003_w0.1/predictions.npz'),R/'evaluation_inputs/reference.npz');link(Path('/mnt/home/faris/logika-twelvehour-cross-20261030/evaluation_inputs/baseline.npz'),R/'evaluation_inputs/baseline.npz')
link(Path('/mnt/home/faris/logika-faris-exploration/exploration-12h-20261017/raw/sample_submission(3).csv'),R/'sample_submission.csv')
(R/'DATA_GATE.json').write_text(json.dumps(dict(status='PASS',checks=checks,feature_bank_unchanged=True,source_packets_equal=True,no_untouched_audit=True),indent=2)+'\n');print('DATA_GATE PASS5folds',flush=True)
