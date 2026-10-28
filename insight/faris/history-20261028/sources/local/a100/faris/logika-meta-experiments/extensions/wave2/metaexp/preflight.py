from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .common import read,write,sha,code_hash,environment,metrics,GIB

def run(root,host,exercise=False):
    root=Path(root).resolve();machines=read(root/'machines.json')['machines'];m=machines[host];checks={};started=time.time()
    for file,want in read(root/'source_manifest.json').items():
        if sha(root/file)!=want:raise ValueError('Source checksum differs: '+file)
    checks['source_files']=True
    manifest=read(root/'bank/manifest.json')
    for file,want in manifest['files'].items():
        if sha(root/'bank'/file)!=want:raise ValueError('Bank checksum differs: '+file)
    checks['bank_files']=True
    from .features import Bank
    bank=Bank(root)
    confirm=np.load(root/'bank/confirmation_rows.npy')
    if len(np.intersect1d(bank.dev,confirm)) or len(bank.dev)!=56000 or len(confirm)!=14000:raise ValueError('Invalid partition')
    if set(np.unique(bank.folds))!={0,1,2}:raise ValueError('Invalid folds')
    checks['frozen_partitions']=True
    from scipy.stats import ks_2samp
    from .reporting import fast_ks
    y=np.array([0,1,0,0,1,1,0,1]);p=np.array([.1,.8,.3,.5,.8,.3,.5,.8])
    if abs(metrics(y,p)['ks']-ks_2samp(p[y==1],p[y==0]).statistic)>1e-12 or abs(fast_ks(y,p)-metrics(y,p)['ks'])>1e-12:raise ValueError('Metric mismatch')
    checks['independent_tied_ks']=True
    os.environ['CUDA_VISIBLE_DEVICES']=m.get('gpu_uuid','')
    from .pipeline import Preprocessor,fit_probe,predict
    rep=read(root/'catalogue.json')['representations'][0]
    fit=bank.dev[:3000];query=bank.dev[3000:3512];yy=bank.y[:3000]
    pre=Preprocessor(bank,rep,'lgb');x=pre.fit_transform(fit,yy);q=pre.transform(query)
    if pre.audit['fit_rows']!=len(fit) or np.intersect1d(pre.fit_rows,query).size:raise ValueError('Transformer fit boundary')
    checks['fold_preprocessing']=True
    backends={}
    for model in ['lgb','cat','xgb','linear']:
        pre=Preprocessor(bank,rep,model);x=pre.fit_transform(fit,yy);q=pre.transform(query)
        device='cuda' if host in ['pc2','pc3'] and model in ['cat','xgb'] else 'opencl' if host=='pc1' and model=='lgb' and m.get('opencl') else 'cpu'
        before=time.time();fitted,w,messages=fit_probe(model,x,yy,rep['config'],2026,device)
        prediction=predict(fitted,q,w)
        if not np.isfinite(prediction).all():raise ValueError('Backend prediction failure')
        if device=='cuda' and model=='xgb':
            config=json.loads(fitted.get_booster().save_config())
            if not config['learner']['generic_param']['device'].startswith('cuda'):raise ValueError('XGBoost fell back from CUDA')
        if device=='cuda' and model=='cat' and fitted.get_all_params()['task_type']!='GPU':raise ValueError('CatBoost did not use GPU')
        backends[model]={'device':device,'seconds':time.time()-before,'rows':len(fit),'queries':len(query),'warnings':messages}
        del fitted
    checks['real_data_backend_smokes']=True
    # No limits are inferred from requested properties: read actual cgroup values.
    from .runtime import local_cgroup
    cgroup=local_cgroup()
    if host in ['pc2','pc3']:
        if cgroup.get('memory.max')!=str(20*GIB) or cgroup.get('memory.swap.max')!='0':raise ValueError('Memory limits not effective')
        quota,period=map(int,cgroup['cpu.max'].split())
        if quota/period>7.00001:raise ValueError('CPU quota exceeds seven')
    checks['resource_limit_readback']=True
    if exercise:
        result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/'tests'),'-v'],cwd=root,capture_output=True,text=True,timeout=120)
        (root/'preflight').mkdir(exist_ok=True);(root/'preflight'/f'{host}_tests.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise ValueError('Acceptance tests failed: '+result.stderr[-2000:])
        checks['acceptance_tests']=True
        from .resource_exercise import exercise as resource_exercise
        resource_exercise(root,host);checks['aggregate_resource_exercise']=True
        if host in ['pc1','pc2']:
            # The real worker, three complete folds, real provided labels; kept
            # outside the campaign run ledger and experiment counts.
            dependencies=[root/'metaexp'/n for n in ['features.py','design.py','common.py','pipeline.py','reference.py','worker.py']]+sorted((root/'vendor').glob('*'))
            from .common import identity
            fit_hash=identity({str(p.relative_to(root)):sha(p) for p in dependencies if p.is_file()})
            receipt=root/'preflight'/f'{host}_fullcv_receipt.json'
            previous=read(receipt) if receipt.exists() else {}
            smoke=root/'preflight'/('cv_smoke_'+code_hash(root)[:10])
            if previous.get('fit_dependency_hash')==fit_hash:smoke=Path(previous['root'])
            smoke.mkdir(exist_ok=True)
            for name in ['bank','metaexp','vendor','.venv']:
                link=smoke/name
                if not link.exists():link.symlink_to(root/name,target_is_directory=True)
            recipe=next(r for r in read(root/'catalogue.json')['recipes'] if r['representation']['is_control'] and r['model']=='lgb') if host=='pc2' else {'id':'reference_v3'}
            task={'id':'full_cv_smoke' if host=='pc2' else 'reference_v3_gate','recipe':recipe,'mode':'cv' if host=='pc2' else 'reference','device':'cpu'}
            from .worker import execute
            if not (smoke/'runs'/task['id']/'complete.json').exists():execute(smoke,task)
            from .reporting import verify_run
            score=verify_run(smoke,task['id'],bank)
            checks['real_complete_three_fold_worker' if host=='pc2' else 'locked_v3_three_fold_refit']=True
            checks['new_protocol_baseline_ks']=score['pooled_oof']['ks']
            write(receipt,{'root':str(smoke),'fit_dependency_hash':fit_hash,'bank_id':bank.manifest['bank_id'],'task':task['id'],'source_hash':read(smoke/'runs'/task['id']/'identity.json')['code_hash']})
    out={'passed':True,'host':host,'time':time.time(),'seconds':time.time()-started,'code_hash':code_hash(root),
         'bank_id':manifest['bank_id'],'environment':environment(),'checks':checks,'backends':backends,'cgroup':cgroup}
    import lightgbm.libpath
    out['lightgbm_library_sha256']={str(p):sha(p) for p in lightgbm.libpath._find_lib_path()}
    write(root/'preflight'/f'{host}.json',out)
    print(json.dumps(out),flush=True);return out
