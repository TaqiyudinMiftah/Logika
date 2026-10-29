"""Independent terminal artifact and SciPy score reconstruction; no training."""
import numpy as np
from scipy.stats import ks_2samp
from common import R,load,sha,ids_sha,dump,utc,check_seal,tasks,catalog,metrics

def main():
    check_seal();passed=[];failed=[];missing=[]
    for task,expected in tasks().items():
        d=R/f'runs/{task}/attempt1'
        if not(d/'RESULT.json').exists():
            failure=next((p for p in [d/'FAILURE.json',d/'INTERRUPTION.json']if p.exists()),None)
            (failed if failure else missing).append(dict(task=task,sha256=sha(failure))if failure else task);continue
        start=load(d/'START.json');result=load(d/'RESULT.json');raw=np.load(d/'prediction.npz');fold=expected['fold'];roles=np.load(R/f'packets/fold{fold}/roles.npz')
        assert result['status']=='complete'and start['task']==expected and start['seed']==expected['seed']+fold
        assert start['source_sha256']==sha(R/'source/worker.py') and sha(d/'START.json')==result['start_sha256'] and sha(d/'prediction.npz')==result['prediction_sha256']
        assert np.array_equal(raw['rows'],np.r_[roles['correction'],roles['validation']]) and raw['raw'].shape==(20000,) and np.isfinite(raw['raw']).all()
        assert start['context_rows_sha256']==ids_sha(roles['context'])and start['outer_labels_loaded']is False and start['correction_labels_used_for_training']is False
        assert result['weights_saved']is False and result['fitted_weights_saved']is False and result['KV_cache_saved']is False
        if catalog()[expected['design']]['resource']=='cpu':
            assert result['audit']['inner_train_rows_sha256']==start['declared_inner_train_rows_sha256']and result['audit']['stopping_rows_sha256']==start['declared_inner_stop_rows_sha256']and result['audit']['preprocessing_refit_on_context']is False
        else:
            audit=result['audit'];assert not audit['context_selection_uses_query_labels']and audit['checkpoint_sha256']and audit['engine_audit']['first_batch_replay_error']<1e-7
        passed.append(dict(task=task,result_sha256=sha(d/'RESULT.json'),prediction_sha256=sha(d/'prediction.npz')))
    board=load('SCOREBOARD.json');y=np.load(R/'evaluation_inputs/target.npy');replayed={}
    for name,item in board['candidates'].items():
        root=R/f'evaluation/cohorts/{name}';a=np.load(root/'prediction.npz');assert np.array_equal(a['rows'],np.arange(70000));p=a['prediction'];k=float(ks_2samp(p[y==0],p[y==1],method='asymp').statistic);assert abs(k-item['metrics']['KS'])<1e-12
        design,mode=name.rsplit('__',1);seeds=[]
        for seed in load('CONFIG.json')['seeds']:
            q=np.full(70000,np.nan);coverage=np.zeros(70000,int)
            for fold in range(5):
                d=R/f'composed/{design}/s{seed}/fold{fold}';r=load(d/'RESULT.json');v=np.load(d/'prediction.npz');assert sha(d/'FROZEN.json')==r['frozen_sha256']and sha(d/'prediction.npz')==r['prediction_sha256'];q[v['rows']]=v[mode];coverage[v['rows']]+=1
            assert np.all(coverage==1)and np.isfinite(q).all();seeds.append(q)
        assert np.array_equal(np.mean(seeds,axis=0),p);replayed[name]=k
    if board['best_development']:
        selected=load('FINAL_SELECTION.json');assert selected['identity']['cohort']==board['best_development']['cohort']
    forbidden=[]
    for pattern in ['*.pt','*.pth','*.pkl','*.joblib','*.cbm','*.safetensors']:forbidden+=list((R/'runs').rglob(pattern))
    assert not forbidden
    dump(dict(status='PASS',UTC=utc(),completed_fits=len(passed),failure_records=len(failed),not_completed=len(missing),passed=passed,failures=failed,missing=missing,recomputed_cohorts=replayed,source_seal_sha256=sha(R/'SOURCE_SEAL.json'),clock_sha256=sha(R/'CLOCK.json'),final_evaluation_sha256=sha(R/'FINAL_EVALUATION.json'),independent_statistical_confirmation=False,scope='Artifact/provenance and arithmetic validation only; all70k reused development',no_weights=True),R/'ARTIFACT_CONSISTENCY_AUDIT.json')

if __name__=='__main__':main()
