from __future__ import annotations
import os
for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[_name]='1'
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from .common import read,write,sha,identity,code_hash,environment,verify_oof,metrics,event,json_safe
from .features import Bank
from .pipeline import fit_predict

def execute(root,task):
    root=Path(root);recipe=task['recipe'];folder=root/'runs'/task['id'];folder.mkdir(parents=True,exist_ok=True)
    bank=Bank(root);start=time.time();kind=task.get('mode','cv');device=task.get('device','cpu')
    ident={'task':task,'bank_id':bank.manifest['bank_id'],'code_hash':code_hash(root),'environment':environment(),'command':sys.argv,'cwd':os.getcwd()}
    if (folder/'identity.json').exists() and read(folder/'identity.json')!=ident:raise ValueError('Changed run identity')
    write(folder/'identity.json',ident);write(folder/'task.json',task);write(folder/'environment.json',ident['environment'])
    write(folder/'status.json',{'state':'running','pid':os.getpid(),'mode':kind,'started':start})
    print(json.dumps({'task':task['id'],'mode':kind,'device':device,'code_hash':ident['code_hash'],'bank_id':bank.manifest['bank_id']}),flush=True)
    try:
        if kind in ['reference','reference_confirmation']:
            from .reference import run
            result=run(root,bank,task,folder)
        elif kind=='ensemble_confirmation':
            from .ensembles import confirm
            result=confirm(root,bank,task,folder)
        elif kind=='nested':
            from .ensembles import nested
            result=nested(root,bank,task,folder)
        elif kind in ['confirmation','repeat','temporal']:
            result=sensitivity(root,bank,task,folder)
        else:
            p=np.full(len(bank.y),np.nan);cdfs={};audits={}
            for fold in range(3):
                fdir=folder/f'fold_{fold}';fdir.mkdir(exist_ok=True)
                vi=np.flatnonzero(bank.folds==fold);ti=np.flatnonzero(bank.folds!=fold)
                if (fdir/'complete.json').exists():
                    saved=read(fdir/'complete.json')
                    if sha(fdir/'prediction.npy')!=saved['sha256']:raise ValueError('Changed checkpoint')
                    pv=np.load(fdir/'prediction.npy');cdf=read(fdir/'preprocessing.json')['train_quantiles']
                else:
                    write(folder/'status.json',{'state':'running','fold':fold,'started':start,'pid':os.getpid()})
                    pv,pre,model,weight,cdf,messages=fit_predict(bank,recipe,bank.dev[ti],bank.y[ti],bank.dev[vi],device,recipe['seed']+fold)
                    audit={**pre.audit,'training_application_ids_sha256':identity(bank.metadata.application_id.iloc[bank.dev[ti]].tolist()),'train_quantiles':cdf.tolist(),'warnings':messages,'weight_correction':weight,
                           'resolved_learner_parameters':json_safe(model.get_params()) if hasattr(model,'get_params') else {}}
                    write(fdir/'preprocessing.json',audit)
                    np.save(fdir/'prediction.npy',pv);write(fdir/'complete.json',{'sha256':sha(fdir/'prediction.npy'),'metrics':metrics(bank.y[vi],pv)})
                    del pre,model
                p[vi]=pv;cdfs[str(fold)]=list(cdf);audits[str(fold)]=read(fdir/'preprocessing.json')
                print(json.dumps({'fold':fold,'metrics':metrics(bank.y[vi],pv),'elapsed':time.time()-start}),flush=True)
            result=verify_oof(bank.y,p,bank.folds)
            np.save(folder/'prediction.npy',p)
            np.savez_compressed(folder/'oof.npz',application_id=bank.metadata.application_id.iloc[bank.dev].to_numpy(dtype=str),fold=bank.folds,target=bank.y,prediction=p)
            write(folder/'cdf.json',cdfs)
            result.update(kind='pipeline',recipe_id=recipe['id'],representation=recipe['representation']['id'],family=recipe['representation']['family'],model=recipe['model'])
        result.update(state='completed',elapsed_seconds=time.time()-start,device=device,finished=time.time(),bank_id=bank.manifest['bank_id'],code_hash=ident['code_hash'])
        write(folder/'scores.json',result)
        files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name not in ['complete.json','status.json','worker.log','resources.json','resources.jsonl']}
        write(folder/'complete.json',{'files':files,'score_sha256':sha(folder/'scores.json')})
        write(folder/'status.json',{'state':'completed','elapsed_seconds':time.time()-start})
        return result
    except BaseException as e:
        write(folder/'status.json',{'state':'failed','error':str(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-start});raise

def sensitivity(root,bank,task,folder):
    from sklearn.model_selection import StratifiedKFold
    recipe=task['recipe'];mode=task['mode'];device=task.get('device','cpu')
    if mode=='confirmation':
        locked=read(root/'finalists.json')
        if task['parent_id'] not in locked['ids']:raise ValueError('Unfrozen confirmation recipe')
        rows=np.load(bank.root/'confirmation_rows.npy')
        p,pre,model,w,cdf,messages=fit_predict(bank,recipe,bank.dev,bank.y,rows,device,recipe['seed'])
        # Open the confirmation labels only after predictions are finalized.
        np.save(folder/'prediction.npy',p)
        y=np.load(bank.root/'confirmation_y.npy')
        import joblib
        # Bank is memory mapped and deliberately excluded from the portable fitted preprocessor.
        pre.bank=None;joblib.dump({'preprocessing':pre,'model':model,'weight':w,'recipe':recipe},folder/'probe.joblib',compress=1)
        write(folder/'preprocessing.json',pre.audit)
        np.savez_compressed(folder/'oof.npz',application_id=bank.metadata.application_id.iloc[rows].to_numpy(dtype=str),target=y,prediction=p)
        return {'kind':'confirmation','parent_id':task['parent_id'],'metrics':metrics(y,p),'train_quantiles':cdf.tolist(),'warnings':messages}
    if mode=='repeat':
        assignments=np.empty(len(bank.y),dtype=np.int8)
        for f,(_,vi) in enumerate(StratifiedKFold(3,shuffle=True,random_state=4917).split(bank.dev,bank.y)):assignments[vi]=f
        p=np.full(len(bank.y),np.nan)
        for f in range(3):
            ti=np.flatnonzero(assignments!=f);vi=np.flatnonzero(assignments==f)
            p[vi],*_=fit_predict(bank,recipe,bank.dev[ti],bank.y[ti],bank.dev[vi],device,4917+f)
        np.save(folder/'prediction.npy',p);np.save(folder/'folds.npy',assignments)
        return {'kind':'repeat','parent_id':task['parent_id'],**verify_oof(bank.y,p,assignments)}
    ordered=np.argsort(bank.metadata.decision_datetime.iloc[bank.dev].to_numpy(),kind='stable');cut=int(.8*len(ordered));ti=ordered[:cut];vi=ordered[cut:]
    # Purge identical decision timestamps at the chronological boundary.
    dates=bank.metadata.decision_datetime.iloc[bank.dev].to_numpy();ti=ti[dates[ti]<dates[vi].min()]
    p,*_=fit_predict(bank,recipe,bank.dev[ti],bank.y[ti],bank.dev[vi],device,2026)
    np.savez_compressed(folder/'temporal.npz',rows=bank.dev[vi],prediction=p,target=bank.y[vi])
    return {'kind':'temporal','parent_id':task['parent_id'],'metrics':metrics(bank.y[vi],p),'train_rows':len(ti),'validation_rows':len(vi)}

if __name__=='__main__':
    execute(Path(sys.argv[1]),read(sys.argv[2]))
