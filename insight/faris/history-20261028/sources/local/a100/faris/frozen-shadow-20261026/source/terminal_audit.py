"""Verify the completed, already evaluated campaign without changing selection."""
from pathlib import Path
import collections, datetime, json, sys
import numpy as np
from sklearn.model_selection import train_test_split
from common import R, dump, sha, utc


def independent_ks(y, p):
    y=np.asarray(y);p=np.asarray(p)
    assert len(y)==len(p) and np.isfinite(p).all()
    order=np.argsort(p,kind='stable');v=p[order];target=y[order]
    ends=np.r_[np.flatnonzero(np.diff(v)!=0),len(v)-1]
    difference=np.cumsum(target/np.sum(y==1)-(1-target)/np.sum(y==0))
    return float(np.max(np.abs(difference[ends])))


def main():
    clock=json.loads((R/'CLOCK.json').read_text());seal=json.loads((R/'FEATURE_FREEZE.json').read_text())
    configs=json.loads((R/'CONFIGS.json').read_text());plan=json.loads((R/'PLAN.json').read_text())
    result=json.loads((R/'evaluation/RESULT.json').read_text());selection=json.loads((R/'evaluation/FROZEN.json').read_text())
    runs=json.loads((R/'MODEL_RUN_AUDIT.json').read_text())
    assert sha(R/'FEATURE_FREEZE.json')==clock['feature_freeze_sha256']==selection['feature_freeze_sha256']
    assert sha(R/'PLAN.json')==clock['plan_sha256']
    assert sha(R/'CONFIGS.json')==plan['configs_sha256']==selection['configs_sha256']
    assert sha(R/'source/evaluate.py')==selection['source_sha256']
    assert sha(R/'evaluation/OOF.npz')==result['OOF_sha256']
    assert sha(R/'evaluation/FROZEN.json')==result['frozen_recipe_sha256']
    assert len(configs)==68 and set(selection['complete_configs'])==set(configs)
    expected={(name,fold)for name in configs for fold in range(5)}
    assert len(runs['completed'])==340 and set(map(tuple,runs['completed']))==expected
    assert not runs['failed'] and not any(runs['pending'].values()) and not runs['unresolved']
    assert result['all_configs_complete'] and result['complete_configs']==68
    assert independent_ks([0,0,1,1],[0,0,1,1])==1
    assert independent_ks([0,1,0,1],[.5,.5,.5,.5])==0
    hosts=collections.Counter();families=collections.Counter();devices=collections.Counter();worker_sources=collections.Counter()
    fit_seconds=[];latest_fit=0;fold_checks=[];allrows=[]
    for fold in range(5):
        folder=R/f'frozen/fold{fold}';manifest=json.loads((folder/'MANIFEST.json').read_text())
        assert sha(folder/'MANIFEST.json')==next(v['manifest_sha256']for v in seal['folds']if v['fold']==fold)
        assert sha(folder/'features.parquet')==manifest['features_sha256']
        assert sha(folder/'roles.npz')==manifest['roles_sha256']
        assert sha(folder/'baseline.npz')==manifest['baseline_sha256']
        roles=np.load(folder/'roles.npz');y=roles['y_correction'];rows=np.r_[roles['correction'],roles['validation']]
        assert len(roles['context'])==50000 and len(roles['correction'])==6000 and len(roles['validation'])==14000
        assert not np.intersect1d(roles['context'],rows).size
        allrows.extend(roles['validation']);predictions={}
        for name,spec in configs.items():
            d=R/f'runs/{name}/fold{fold}';start=json.loads((d/'START.json').read_text());done=json.loads((d/'RESULT.json').read_text())
            assert done['status']=='complete' and done['fitted_weights_saved'] is False
            assert start['config']==spec and start['fold']==fold and start['seed']==spec['seed']+fold
            assert start['feature_sha256']==manifest['features_sha256'] and start['freeze_sha256']==clock['feature_freeze_sha256']
            assert start['columns']==manifest['views'][spec['view']] and start['outer_labels_loaded'] is False
            assert start['context_rows']==50000 and sha(d/'prediction.npz')==done['prediction_sha256']
            p=np.load(d/'prediction.npz');assert np.array_equal(p['rows'],rows)
            values=p['raw'];assert values.shape==(20000,)and np.isfinite(values).all()and 0<=values.min()<=values.max()<=1
            predictions[name]=values;hosts[start['host']]+=1;families[spec['kind']]+=1;worker_sources[start['source_sha256']]+=1
            device=done.get('execution_device','cuda'if spec['kind']in ['kumo','contextual']else 'cpu');devices[device]+=1
            if spec['kind']in ['kumo','contextual']:assert device=='cuda'
            fit_seconds.append(done['seconds']);latest_fit=max(latest_fit,datetime.datetime.fromisoformat(done['UTC']).timestamp())
        ranking,choice=train_test_split(np.arange(6000),test_size=.5,stratify=y,random_state=261026)
        # Exact saved procedure, including its original ROC implementation, is replayed for auditing only.
        # No new recipe is created or chosen from outer results.
        from common import ks as original_ks
        order=sorted(configs,key=lambda name:(-original_ks(y[ranking],predictions[name][ranking]),name))
        baseline=np.load(folder/'baseline.npz')['raw'];best=None
        for count in sorted(set([1,min(4,len(order)),min(8,len(order)),len(order)])):
            avg=np.mean([predictions[name]for name in order[:count]],axis=0)
            for weight in [0,.1,.2,.4,.7,1.]:
                p=(1-weight)*baseline+weight*avg;key=(original_ks(y[choice],p[choice]),-weight,-count)
                if best is None or key>best[0]:best=(key,p,order[:count],weight)
        recipe=next(v for v in selection['recipes']if v['fold']==fold)
        assert recipe['members']==best[2] and recipe['weight']==best[3] and recipe['outer_labels_loaded'] is False
        frozen=np.load(R/f'evaluation/fold{fold}_primary.npz');assert np.array_equal(frozen['rows'],rows)and np.array_equal(frozen['raw'],best[1])
        fold_checks.append({'fold':fold,'columns_preserved':True,'row_roles_passed':True,'selection_replay_exact':True,'selected_members':len(best[2]),'selected_weight':best[3]})
    assert np.array_equal(np.sort(allrows),np.arange(70000))
    selected_at=datetime.datetime.fromisoformat(selection['UTC']).timestamp();completed_at=datetime.datetime.fromisoformat(result['UTC']).timestamp()
    assert latest_fit<=selected_at<completed_at<clock['deadline_epoch']
    oof=np.load(R/'evaluation/OOF.npz');y=oof['target'];assert len(y)==70000 and set(np.unique(y))=={0,1}
    baseline_score=independent_ks(y,oof['baseline']);assert abs(baseline_score-result['baseline_KS'])<1e-10
    max_error=0.
    for name,metrics in result['diagnostic_metrics'].items():
        p=oof[name];score=independent_ks(y,p);error=abs(score-metrics['KS']);max_error=max(max_error,error);assert error<1e-10
        deltas=[independent_ks(y[oof['fold']==f],p[oof['fold']==f])-independent_ks(y[oof['fold']==f],oof['baseline'][oof['fold']==f])for f in range(5)]
        assert np.max(np.abs(np.asarray(deltas)-metrics['fold_delta']))<1e-10
    replay=float(np.max(np.abs(oof['baseline_replay']-oof['baseline'])));assert replay<1e-10
    weights=[str(p.relative_to(R))for p in R.rglob('*')if p.is_file()and p.suffix.lower()in ['.pt','.pth','.ckpt','.safetensors','.cbm','.pkl','.joblib']]
    assert not weights,weights
    audit={'status':'passed','UTC':utc(),'planned_configurations':68,'verified_fits':340,'all_configurations_fivefold_complete':True,
           'terminal_failed_fits':0,'CPU_resource_incident':'19 earlier node2001 OOM attempts were disclosed separately in CPU_RESOURCE_LIMIT.json; not340successful fit records.',
           'feature_freeze_sha256':sha(R/'FEATURE_FREEZE.json'),'plan_sha256':sha(R/'PLAN.json'),'configs_sha256':sha(R/'CONFIGS.json'),
           'result_sha256':sha(R/'evaluation/RESULT.json'),'OOF_sha256':sha(R/'evaluation/OOF.npz'),'selection_sha256':sha(R/'evaluation/FROZEN.json'),
           'independent_numpy_KS_max_difference':max_error,'baseline_prediction_replay_max_abs':replay,
           'fold_checks':fold_checks,'prediction_shape_and_probability_checks':True,'parameters_seeds_feature_and_role_hashes_verified':True,
           'fit_hosts':dict(hosts),'fit_families':dict(families),'fit_devices':dict(devices),'worker_source_versions':dict(worker_sources),
           'fitted_weight_files_found':weights,'selection_UTC':selection['UTC'],'evaluation_UTC':result['UTC'],
           'freeze_to_evaluation_hours':(completed_at-clock['started_epoch'])/3600,'sum_successful_fit_seconds':sum(fit_seconds),
           'clock_reset':False,'target_reached':result['target_reached'],'promotion_eligible':False,'submission_made':False,
           'limits':'Reused development data, not an untouched audit. Exact replay confirms saved recipe and artifacts, not unseen-data generalization.'}
    dump(audit,R/'TERMINAL_AUDIT.json');print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
