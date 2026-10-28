"""Refit and replay the selected research recipe, sharing nested base fits."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
os.environ['MPLBACKEND']='Agg'
import contextlib,copy,gc,hashlib,json,sys,time,traceback
from pathlib import Path
import joblib,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'research_source'))
from features import ExploreBank
from adapter import fold_fit
from metaexp.pipeline import predict
from utils import read,write,sha,ident,combine
_BANK=None


def get_bank(plan):
    global _BANK
    if _BANK is None:_BANK=ExploreBank(Path(plan['bank']).parent,plan['extras'])
    return _BANK


def sources_ok():
    for name,h in read(ROOT/'research_source_manifest.json').items():
        assert sha(ROOT/'research_source'/name)==h,('Changed frozen source',name)


class Context:
    def __init__(self,folder,plan,gpu=False,cpus=1):
        self.root=Path(folder);self.inputs={'extras':plan['extras'],'bank':plan['bank']};self.family='tabm' if gpu else 'cpu';self.spec={'resources':{'cpus':cpus,'gpus':int(gpu)}};self.models=[]
    def check_stop(self):
        clock=read(ROOT/'clock.json')
        if time.time()>=clock['deadline']:raise TimeoutError('Final-refit safety deadline expired')
        if (ROOT/'stop.json').exists():raise InterruptedError('Operator stopped this refit')
    def path(self,name):
        p=self.root/name;assert p.resolve().is_relative_to(self.root.resolve());p.parent.mkdir(parents=True,exist_ok=True);return p
    def model(self,name):assert self.path(name).is_file();self.models.append(name)
    def save_json(self,name,value):write(self.path(name),value)


def fit_job(task):
    plan=read(ROOT/'plan.json');sources_ok();folder=ROOT/'fits'/task['key'];request={k:v for k,v in task.items() if k not in ['fitrows','queryrows','y']};request.update(fit_rows_hash=ident(task['fitrows'].tolist()),query_rows_hash=ident(task['queryrows'].tolist()),labels_hash=hashlib.sha256(task['y'].tobytes()).hexdigest(),plan_sha256=sha(ROOT/'plan.json'),source_sha256=sha(ROOT/'research_source_manifest.json'))
    if (folder/'complete.json').exists():
        assert read(folder/'request.json')==request
        for n,h in read(folder/'complete.json')['files'].items():assert sha(folder/n)==h
        return str(folder)
    if folder.exists():
        previous=ROOT/'partial_attempts'/(task['key']+'-'+str(time.time_ns()));previous.parent.mkdir(exist_ok=True);folder.rename(previous)
    folder.mkdir(parents=True);write(folder/'request.json',request);gpu=task['config_id']==plan['tabm_config'];os.environ['CUDA_VISIBLE_DEVICES']=plan['gpu_uuid'] if gpu else ''
    p=copy.deepcopy(read(ROOT/'research_source/configurations.json')[task['config_id']]);rep=copy.deepcopy(read(ROOT/'research_source/views.json')[p['view']]);rep['config']['weight']=p['weight'];ctx=Context(folder,plan,gpu=gpu,cpus=plan['gpu_cpus'] if gpu else 1);start=time.time();write(folder/'status.json',{'state':'running','pid':os.getpid(),'started':start,'kind':p['model']})
    try:
        with (folder/'training.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            if gpu:
                import torch
                assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'])
            values=fold_fit(ctx,get_bank(plan),p,rep,task['fitrows'],task['y'],task['queryrows'],'model',task['seed'])
            assert values.shape==(len(task['queryrows']),) and np.isfinite(values).all() and ((values>=0)&(values<=1)).all()
            np.savez_compressed(folder/'prediction.npz',rows=task['queryrows'],prediction=values)
            write(folder/'model_index.json',{'models':sorted(set(ctx.models)),'config_id':task['config_id'],'seed':task['seed'],'host':os.uname().nodename,'gpu':plan['gpu_uuid'] if gpu else None,'elapsed_seconds':time.time()-start})
        files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name not in ['status.json','complete.json']};write(folder/'complete.json',{'files':files});write(folder/'status.json',{'state':'completed','seconds':time.time()-start});return str(folder)
    except BaseException:
        write(folder/'status.json',{'state':'failed','traceback':traceback.format_exc()});raise


def restore(pre,bank):
    pre.bank=bank
    if hasattr(pre,'pre'):pre.pre.bank=bank
    return pre


def predict_base(config_id,folder,rows,plan,gpu=True):
    """Inference only; reload every saved transformer and model."""
    sources_ok();folder=Path(folder);bank=get_bank(plan);p=read(ROOT/'research_source/configurations.json')[config_id]
    if p['model']=='champion':
        bundle=joblib.load(folder/'pipeline.joblib');pre=restore(bundle['preprocessing'],bank);values=predict(bundle['model'],pre.transform(rows),bundle['weight']);return values
    pre=restore(joblib.load(folder/'preprocessing.joblib'),bank);x=pre.transform(rows)
    if p['model']=='tabm':
        import torch,neural
        torch.set_num_threads(plan['gpu_cpus']);device='cuda' if gpu and torch.cuda.is_available() else 'cpu'
        if device=='cuda':torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'])
        state=torch.load(folder/'weights.pt',map_location='cpu',weights_only=False);assert state['params'].get('encoding')=='plain';m=neural.make('tabm',state['features'],state['params'],np.zeros((1,state['features']),np.float32));m.load_state_dict(state['state'],strict=True);m.to(device);m.eval();values=[]
        with torch.inference_mode():
            for start in range(0,len(x),1024):
                q=torch.as_tensor(x[start:start+1024],device=device);z=m(q).squeeze(-1).sigmoid()
                if z.ndim==2:z=z.mean(1)
                values.append(z.float().cpu().numpy())
        a=np.concatenate(values);values=np.clip(a/(p['weight']*(1-a)+a),1e-8,1-1e-8);del m
        if device=='cuda':torch.cuda.empty_cache()
    else:
        m=joblib.load(folder/'model.joblib');values=predict(m,x,p['weight'])
    del pre,x;gc.collect();assert np.isfinite(values).all();return values


def predict_streams(rows,base_folders,meta_folders,plan):
    arrays={};query={}
    for key in plan['selected']['parents']:
        if key in plan['stream_configs']:
            arrays[key]=predict_base(plan['stream_configs'][key],base_folders[plan['stream_configs'][key]],rows,plan)
        else:
            for cid in plan['shared_stack_components']:
                if cid not in query:query[cid]=predict_base(cid,base_folders[cid],rows,plan)
            xx=np.column_stack([query[cid] for cid in plan['shared_stack_components']]);meta=joblib.load(meta_folders[key]);values=meta.predict_proba(xx)[:,1] if key=='pc2_job_000853' else np.clip(meta.predict(xx),1e-8,1-1e-8);arrays[key]=np.clip(values,1e-8,1-1e-8)
    return arrays


def verify_retained_validation(plan):
    """Prove new inference code replays the selected saved validation models."""
    from scipy.stats import ks_2samp
    rows=np.load(Path(plan['bank'])/'confirmation_rows.npy');base_folders={};meta_folders={}
    for key,ref in plan['references'].items():
        root=Path(ref['folder']);assert sha(root/'validation.npz')==ref['validation_sha256']
        if key in plan['stream_configs']:base_folders[plan['stream_configs'][key]]=root/'confirmation'
        else:
            for cid in plan['shared_stack_components']:base_folders.setdefault(cid,root/'outer_0'/('component_'+cid)/'refit')
            meta_folders[key]=root/'outer_0/meta.joblib'
    arrays=predict_streams(rows,base_folders,meta_folders,plan);diffs={}
    for key in arrays:
        z=np.load(Path(plan['references'][key]['folder'])/'validation.npz');assert np.array_equal(z['rows'],rows);diffs[key]=float(np.max(np.abs(z['prediction']-arrays[key])));assert diffs[key]<3e-6,(key,diffs[key])
    values=combine(plan['selected'],arrays);expected=np.load(ROOT/'evidence/selected_validation.npz')['prediction'];maxdiff=float(np.max(np.abs(expected-values)));assert maxdiff<3e-6
    np.savez_compressed(ROOT/'evidence/replayed_validation.npz',rows=rows,prediction=values)
    y=np.load(Path(plan['bank'])/'confirmation_y.npy');score=float(ks_2samp(values[y==0],values[y==1]).statistic);target=plan['metrics']['reused_validation']['ks'];assert abs(score-target)<2e-4,(score,target)
    result={'passed':True,'component_max_probability_difference':diffs,'ensemble_max_probability_difference':maxdiff,'ks':score,'saved_selected_ks':target,'rows':len(rows),'confirmation_is_selection_data':True};write(ROOT/'evidence/inference_replay.json',result);return result
