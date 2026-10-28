"""Finite all-label fit and exact saved-model replay; never uploads anything."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,fcntl,json,shutil,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from benchmark import ROOT,bank,read,write,sha,fit,predict,trimmed


def infer(component,folder,rows):
    b=bank();folder=Path(folder)
    if component['kind'] in ['historical','resolved']:
        model=joblib.load(folder/'pipeline.joblib');pre=model['preprocessing'];pre.bank=b
        return predict(model['model'],pre.transform(rows),model['weight'])
    pre=joblib.load(folder/'model/preprocessing.joblib');pre.bank=b;pre.pre.bank=b
    model=joblib.load(folder/'model/model.joblib')
    return predict(model,pre.transform(rows),component['configuration']['weight'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replay-only',action='store_true');parser.add_argument('--fixed-secondary',action='store_true');a=parser.parse_args()
    lock=(ROOT/'build.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(ROOT/'benchmark_plan.json');result=read(ROOT/'benchmark_result.json')
    replay_secondary=a.replay_only and (ROOT/'artifacts/pipeline_index.json').exists() and read(ROOT/'artifacts/pipeline_index.json').get('decision_type')=='fixed_secondary_candidate'
    secondary=read(ROOT/'fixed_comparison.json') if a.fixed_secondary or replay_secondary else None
    if secondary:
        assert not result['promotion_gate_passed'] and secondary['primary_nested_gate_passed'] is False
        selected=secondary['selected'];assert selected['eligible'] and selected['weight'] in plan['candidate_weights']
    else:
        assert result['promotion_gate_passed'],'Frozen nested promotion gate failed; do not silently weaken it'
        selected=result['final_selection']
    component=next(c for c in plan['candidate_components'] if c['id']==selected['component'])
    b=bank();query=np.flatnonzero(b.metadata.split.to_numpy()=='test');trainrows=np.flatnonzero(b.metadata.split.to_numpy()=='train')
    folder=ROOT/'artifacts';folder.mkdir(exist_ok=True)
    if not a.replay_only:
        assert time.time()<read(ROOT/'clock.json')['deadline'],'Do not reset the finite build deadline'
        data=pd.read_csv(plan['train']).set_index('application_id');y=data.target.reindex(b.metadata.application_id.iloc[trainrows]).to_numpy(dtype=np.int64)
        winner=fit({'key':'final_'+component['id'],'component':component,'fitrows':trainrows,'y':y,'queryrows':query,'seed':2026})
        lineage={}
        for c in plan['champion_components']:
            old=Path('/home/faris/faris/submission-research-20261017/artifacts/full/primary')/c['original_id']/'pipeline.joblib'
            dest=folder/'champion'/c['original_id']/'pipeline.joblib';dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(old,dest);assert sha(old)==sha(dest)
            lineage[c['id']]={'component':c,'model':str(dest.relative_to(ROOT)),'sha256':sha(dest)}
        write(folder/'pipeline_index.json',{'champion':lineage,'addition':component,'addition_folder':str(Path(winner).relative_to(ROOT)),
            'addition_weight':selected['weight'],'benchmark_sha256':sha(ROOT/'benchmark_result.json'),'plan_sha256':sha(ROOT/'benchmark_plan.json'),
            'decision_type':'fixed_secondary_candidate' if secondary else 'nested_selector',
            'fixed_comparison_sha256':sha(ROOT/'fixed_comparison.json') if secondary else None})
        values,anchor,addition=replay(plan,query)
        sample=pd.read_csv(plan['sample']);ids=b.metadata.application_id.iloc[query].to_numpy();ix=pd.Index(ids).get_indexer(sample.application_id);assert (ix>=0).all()
        sample.default_probability=values[ix];sample.to_csv(folder/'submission.csv',index=False)
        np.savez_compressed(folder/'streams.npz',rows=query,champion=anchor,addition=addition,prediction=values)
    values,anchor,addition=replay(plan,query)
    saved=np.load(folder/'streams.npz');assert np.array_equal(saved['rows'],query)
    errors={name:float(np.max(np.abs(v-saved[name]))) for name,v in [('prediction',values),('champion',anchor),('addition',addition)]}
    assert max(errors.values())<1e-12
    sample=pd.read_csv(plan['sample']);csv=pd.read_csv(folder/'submission.csv');assert csv.columns.tolist()==sample.columns.tolist()==['application_id','default_probability']
    assert len(csv)==30000 and csv.application_id.is_unique and csv.application_id.tolist()==sample.application_id.tolist()
    ix=pd.Index(b.metadata.application_id.iloc[query]).get_indexer(csv.application_id)
    error=float(np.max(np.abs(csv.default_probability.to_numpy()-values[ix])));assert error<1e-12
    assert np.isfinite(csv.default_probability).all() and csv.default_probability.between(0,1).all()
    model_files=[p for base in [ROOT/'fits',folder] for p in base.rglob('*') if p.suffix in ['.joblib','.pt']]
    record={'state':'completed','finished':time.time(),'submission_sha256':sha(folder/'submission.csv'),'rows':len(csv),
        'all_rows_model_replay':True,'component_replay_max_error':errors,'csv_max_error':error,
        'exact_champion_replay':True,'addition':selected,'nested_benchmark':result,
        'decision_type':'fixed_secondary_candidate' if secondary else 'nested_selector',
        'primary_nested_gate_passed':result['promotion_gate_passed'],'fixed_comparison':secondary,
        'quality_status':secondary['quality_status'] if secondary else 'Passed frozen nested exploratory gate',
        'models':{str(p.relative_to(ROOT)):sha(p) for p in model_files},
        'historical_selection_reuse_disclosed':True,'no_upload':True}
    write(folder/'complete.json',record);print(json.dumps({k:v for k,v in record.items() if k!='models'},indent=2))


def replay(plan,rows):
    index=read(ROOT/'artifacts/pipeline_index.json');values={}
    for key,item in index['champion'].items():
        path=ROOT/item['model'];assert sha(path)==item['sha256']
        values[key]=infer(item['component'],path.parent,rows)
    anchor=trimmed(values,list(index['champion']))
    b=bank();sample=pd.read_csv(ROOT/'evidence/champion_submission.csv');ix=pd.Index(b.metadata.application_id.iloc[rows]).get_indexer(sample.application_id)
    assert (ix>=0).all() and np.max(np.abs(anchor[ix]-sample.default_probability.to_numpy()))<1e-12,'Champion model replay mismatch'
    addition=infer(index['addition'],ROOT/index['addition_folder'],rows)
    weight=index['addition_weight'];return (1-weight)*anchor+weight*addition,anchor,addition


if __name__=='__main__':main()
