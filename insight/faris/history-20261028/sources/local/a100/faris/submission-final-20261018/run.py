"""One finite final refit. Retains all models, verifies replay, writes one CSV."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import argparse,concurrent.futures as cf,fcntl,importlib.metadata,json,multiprocessing,time,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd,psutil
from pipeline import ROOT,read,write,sha,ident,combine,get_bank,fit_job,predict_streams,verify_retained_validation,sources_ok


def main():
    p=argparse.ArgumentParser();p.add_argument('--replay-only',action='store_true');p.add_argument('--bank');p.add_argument('--extras');p.add_argument('--sample');a=p.parse_args();plan=read(ROOT/'plan.json')
    for key in ['bank','extras','sample']:
        if getattr(a,key):plan[key]=str(Path(getattr(a,key)).resolve())
    os.environ['CUDA_VISIBLE_DEVICES']=plan['gpu_uuid'];sources_ok();lock=(ROOT/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.replay_only:
        verify_final(plan);return
    if not (ROOT/'clock.json').exists():write(ROOT/'clock.json',{'start':time.time(),'deadline':time.time()+plan['safety_seconds']})
    assert time.time()<read(ROOT/'clock.json')['deadline'],'Refit deadline expired; do not silently start another window'
    write(ROOT/'environment.json',{'python':sys.version,'host':os.uname().nodename,'packages':{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','lightgbm','joblib','torch','tabm','interpret-core']}})
    for key,h in plan['input_sha256'].items():
        path=plan[key] if key in ['train','test','sample'] else (Path(plan['bank'])/'manifest.json' if key=='bank_manifest' else Path(plan['extras'])/'manifest.json');assert sha(path)==h
    if not (ROOT/'evidence/inference_replay.json').exists():
        write(ROOT/'status.json',{'phase':'retained_validation_replay','time':time.time()});print(json.dumps({'phase':'retained_validation_replay','result':verify_retained_validation(plan)}),flush=True)
    bank=get_bank(plan);trainrows=np.flatnonzero(bank.metadata.split.to_numpy()=='train');queryrows=np.flatnonzero(bank.metadata.split.to_numpy()=='test');labels=pd.read_csv(plan['train']).set_index('application_id').target.reindex(bank.metadata.application_id.iloc[trainrows]).to_numpy(dtype=np.int64);assert len(trainrows)==70000 and len(queryrows)==30000 and set(labels)=={0,1};assert not np.intersect1d(trainrows,queryrows).size
    tasks=[];need_stack=any(k not in plan['stream_configs'] for k in plan['selected']['parents'])
    for cid in dict.fromkeys(plan['stream_configs'][k] for k in plan['selected']['parents'] if k in plan['stream_configs']):tasks.append({'key':'full_'+cid,'config_id':cid,'fitrows':trainrows,'queryrows':queryrows,'y':labels,'seed':plan['seed']})
    split=[]
    if need_stack:
        from sklearn.model_selection import StratifiedKFold
        split=list(StratifiedKFold(3,shuffle=True,random_state=plan['inner_seed']).split(trainrows,labels))
        for cid in plan['shared_stack_components']:
            for g,(ti,vi) in enumerate(split):tasks.append({'key':'inner_'+str(g)+'_'+cid,'config_id':cid,'fitrows':trainrows[ti],'queryrows':trainrows[vi],'y':labels[ti],'seed':plan['seed']+g})
            tasks.append({'key':'full_'+cid,'config_id':cid,'fitrows':trainrows,'queryrows':queryrows,'y':labels,'seed':plan['seed']})
    write(ROOT/'training_design.json',{'unique_fit_jobs':len(tasks),'shared_stack_base_ids':plan['shared_stack_components'] if need_stack else [],'full_training_rows':len(trainrows),'test_rows':len(queryrows),'seed':plan['seed'],'inner_seed':plan['inner_seed'],'recipe_id':plan['selected']['id']})
    finished={};mp=multiprocessing.get_context('spawn')
    with cf.ProcessPoolExecutor(max_workers=plan['cpu_workers'],mp_context=mp) as cpu,cf.ProcessPoolExecutor(max_workers=1,mp_context=mp) as gpu:
        futures={(gpu if t['config_id']==plan['tabm_config'] else cpu).submit(fit_job,t):t for t in tasks}
        while futures:
            if time.time()>=read(ROOT/'clock.json')['deadline']:raise TimeoutError('Final-refit safety deadline expired')
            done,_=cf.wait(futures,timeout=5,return_when=cf.FIRST_COMPLETED)
            for f in done:
                task=futures.pop(f);finished[task['key']]=f.result();print(json.dumps({'phase':'refit','done':len(finished),'total':len(tasks),'task':task['key']}),flush=True)
            processes=[psutil.Process()]+psutil.Process().children(recursive=True);rss=0
            for proc in processes:
                try:rss+=proc.memory_info().rss
                except psutil.Error:pass
            write(ROOT/'status.json',{'phase':'refit','completed':len(finished),'total':len(tasks),'remaining':[t['key'] for t in futures.values()],'time':time.time(),'summed_rss_bytes':rss})
    base_folders={cid:Path(finished['full_'+cid])/'model' for cid in set(t['config_id'] for t in tasks)};meta_folders={}
    if need_stack:
        from sklearn.linear_model import LogisticRegression,Ridge
        inner=np.full((len(trainrows),3),np.nan)
        for j,cid in enumerate(plan['shared_stack_components']):
            for g,(ti,vi) in enumerate(split):
                z=np.load(Path(finished['inner_'+str(g)+'_'+cid])/'prediction.npz');assert np.array_equal(z['rows'],trainrows[vi]);inner[vi,j]=z['prediction']
        assert np.isfinite(inner).all();folder=ROOT/'artifacts/shared_stack';folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'inner_oof.npz',rows=trainrows,prediction=inner);write(folder/'lineage.json',{'columns':plan['shared_stack_components'],'training_rows':70000,'inner_folds':3,'inner_seed':plan['inner_seed'],'no_test_labels':True})
        for key in plan['selected']['parents']:
            if key in plan['stream_configs']:continue
            model=LogisticRegression(C=.1,max_iter=1000) if key=='pc2_job_000853' else Ridge(alpha=100);model.fit(inner,labels);file=folder/(key+'.joblib');joblib.dump(model,file,compress=1);meta_folders[key]=str(file)
    arrays={}
    for key in plan['selected']['parents']:
        if key in plan['stream_configs']:
            cid=plan['stream_configs'][key];arrays[key]=np.load(Path(finished['full_'+cid])/'prediction.npz')['prediction']
        else:
            xx=np.column_stack([np.load(Path(finished['full_'+cid])/'prediction.npz')['prediction'] for cid in plan['shared_stack_components']]);model=joblib.load(meta_folders[key]);values=model.predict_proba(xx)[:,1] if key=='pc2_job_000853' else np.clip(model.predict(xx),1e-8,1-1e-8);arrays[key]=np.clip(values,1e-8,1-1e-8)
    result=combine(plan['selected'],arrays);folder=ROOT/'artifacts';folder.mkdir(exist_ok=True);np.savez_compressed(folder/'streams.npz',rows=queryrows,**arrays)
    index={'base_folders':{k:str(v.relative_to(ROOT)) for k,v in base_folders.items()},'meta_folders':{k:str(Path(v).relative_to(ROOT)) for k,v in meta_folders.items()},'query_rows':queryrows.tolist(),'selected':plan['selected']};write(folder/'pipeline_index.json',index)
    sample=pd.read_csv(plan['sample']);assert sample.columns.tolist()==['application_id','default_probability'];assert sample.application_id.is_unique and len(sample)==30000;testids=bank.metadata.application_id.iloc[queryrows].to_numpy();ix=pd.Index(testids).get_indexer(sample.application_id);assert (ix>=0).all() and len(np.unique(ix))==30000;sample.default_probability=result[ix];sample.to_csv(folder/'submission.csv',index=False)
    write(ROOT/'status.json',{'phase':'independent_model_replay','time':time.time()});verification=verify_final(plan)
    model_files=[p for base in [ROOT/'fits',ROOT/'artifacts'] for p in base.rglob('*') if p.is_file() and p.suffix in ['.joblib','.pt']];complete={'state':'completed','finished':time.time(),'recipe':plan['selected'],'development':plan['metrics']['development'],'reused_validation':plan['metrics']['reused_validation'],'confirmation_is_selection_data':True,'submission_sha256':sha(folder/'submission.csv'),'models':{str(p.relative_to(ROOT)):sha(p) for p in model_files},'model_bytes':sum(p.stat().st_size for p in model_files),'full_training_rows':70000,'submission_rows':30000,'verification':verification,'no_upload':True};write(folder/'complete.json',complete);write(ROOT/'status.json',{'phase':'completed','time':time.time(),'submission_sha256':complete['submission_sha256']});print(json.dumps(complete),flush=True)


def verify_final(plan):
    folder=ROOT/'artifacts';index=read(folder/'pipeline_index.json');rows=np.asarray(index['query_rows']);base={k:ROOT/v for k,v in index['base_folders'].items()};meta={k:ROOT/v for k,v in index['meta_folders'].items()};arrays=predict_streams(rows,base,meta,plan);stored=np.load(folder/'streams.npz');assert np.array_equal(stored['rows'],rows);diff={k:float(np.max(np.abs(arrays[k]-stored[k]))) for k in arrays};assert max(diff.values())<3e-6,diff
    values=combine(plan['selected'],arrays);bank=get_bank(plan);ids=bank.metadata.application_id.iloc[rows].to_numpy();csv=pd.read_csv(folder/'submission.csv');sample=pd.read_csv(plan['sample']);assert csv.application_id.tolist()==sample.application_id.tolist();assert csv.application_id.is_unique and len(csv)==30000;pred=csv.default_probability.to_numpy();assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();ix=pd.Index(ids).get_indexer(csv.application_id);assert (ix>=0).all();error=float(np.max(np.abs(pred-values[ix])));assert error<3e-6
    result={'passed':True,'all_rows_replayed':len(rows),'component_max_probability_difference':diff,'csv_max_probability_difference':error,'finite_in_range':True,'exact_sample_id_order':True,'minimum_probability':float(pred.min()),'maximum_probability':float(pred.max()),'prediction_mean':float(pred.mean()),'submission_sha256':sha(folder/'submission.csv')};write(folder/'verification.json',result);return result


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        write(ROOT/'failure.json',{'time':time.time(),'traceback':traceback.format_exc()});raise
