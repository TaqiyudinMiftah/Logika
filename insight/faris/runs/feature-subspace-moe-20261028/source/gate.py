"""Measure the admission checks; no favorable candidate score is a gate condition."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='2'
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from common import R,load,dump,sha,ids_sha,metrics,utc,check_seal
from gating import synthetic_checks
from compose import completed
from evaluate import ks_numpy


def main():
    clock=check_seal();config=load('CONFIG.json');checks=[]
    for y,p,expected in [([0,0,1,1],[0,0,1,1],1.),([0,0,1,1],[1,1,0,0],1.),
                         ([0,1,0,1],[.5]*4,0.)]:
        assert abs(metrics(y,p)['KS']-expected)<1e-12
        assert abs(ks_numpy(np.array(y),np.array(p))-expected)<1e-12
    rng=np.random.default_rng(7731);y=rng.integers(0,2,600);p=np.round(rng.random(600),1)
    assert abs(ks_numpy(y,p)-ks_2samp(p[y==0],p[y==1]).statistic)<1e-12
    assert abs(ks_numpy(y,p)-metrics(y,p)['KS'])<1e-12
    checks.append(dict(id='evaluator',status='pass',known_cases=4,ties_match_SciPy=True))
    synthetic=synthetic_checks();checks.append(dict(id='gate_boundary_and_solver',status='pass',observed=synthetic))
    prep=load('PREPARATION_AUDIT.json');assert prep['status']=='PASS'
    evidence=[]
    for f in range(5):
        root=R/f'packets/fold{f}';manifest=json.loads((root/'MANIFEST.json').read_text());roles=np.load(root/'roles.npz')
        for name,key in [('features.parquet','feature_sha256'),('roles.npz','roles_sha256'),
                         ('gate.parquet','gate_sha256'),('anchor.npz','anchor_sha256')]:
            assert sha(root/name)==manifest[key]
        assert sha(root/'MANIFEST.json')==prep['packet_manifests'][str(f)]
        assert manifest['outer_labels_in_worker_packet']is False
        assert all(len(cols)<=99 for name,cols in manifest['views'].items()if name!='expanded_union')
        assert len(manifest['views']['expanded_union'])<=400
        frame=pd.read_parquet(root/'features.parquet');assert not {'target','application_id','customer_id'}&set(frame)
        assignment=np.load(root/'anchor.npz')['inner_fold'];audit=json.loads((root/'ANCHOR_AUDIT.json').read_text())
        for record in audit['crossfit']:
            k=record['inner'];train=np.r_[roles['context'],roles['correction'][assignment!=k]];query=roles['correction'][assignment==k]
            assert ids_sha(train)==record['spline_training_rows_sha256'] and ids_sha(query)==record['heldout_global_rows_sha256']
            assert not np.intersect1d(train,query).size and not np.intersect1d(train,roles['validation']).size
        assert all(s['correction_labels_used']is False and s['outer_labels_used']is False for s in manifest['selection'])
        evidence.append(dict(fold=f,context_class_counts=np.bincount(roles['y_context'].astype(int)).tolist(),
                             correction_class_counts=np.bincount(roles['y_correction'].astype(int)).tolist(),
                             anchor_head_error=audit['head_replay_max_abs'],anchor_spline_error=audit['spline_replay_max_abs'],
                             nested_anchor_spline_membership_verified=True,
                             views={k:len(v)for k,v in manifest['views'].items()}))
    checks.append(dict(id='boundaries_and_support',status='pass',folds=evidence,
                       prior_timing_audit_sha256=sha(R/'REPAYMENT_TIMING_AUDIT.json'),
                       snapshot_limit=prep['snapshot_semantics']))
    pilots=[]
    for task_id in config['pilot_tasks']:
        path=completed(task_id);assert path is not None,task_id
        z=np.load(path/'prediction.npz');p=z['raw'];task=config['tasks'][task_id]
        assert p.shape==(20000,) and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        roles=np.load(R/f"packets/fold{task['fold']}/roles.npz")
        result=json.loads((path/'RESULT.json').read_text())
        pilots.append(dict(task=task_id,result_sha256=sha(path/'RESULT.json'),
                           result_path=str(path.relative_to(R)),seconds=result['seconds'],
                           correction_metrics=metrics(roles['y_correction'],p[:6000]),
                           prediction_std=float(p.std()),max_RSS_KiB=result['max_RSS_KiB']))
    checks.append(dict(id='actual_runtime_pilots',status='pass',observed=pilots,
                       selection_uses_pilot_fitness=False,criterion='Completed finite bounded predictions and declared repeat checks; no score-target predicate.'))
    checks.append(dict(id='baseline_and_uncertainty',status='pass',matched_baseline=prep['baseline'],
                       constant_KS=0.,constant_AUC=.5,baseline_replay_verified=True,
                       design='Five matching outer folds, two seeds; primary paired fold/class-stratified bootstrap2000; prior exposure retained.',
                       independent_audit=False))
    required=load('VALIDATION_CONTRACT.json')['required_checks'];assert {c['id']for c in checks}==set(required)
    assert all(c['status']=='pass'for c in checks)
    dump(dict(status='PASS',scope='Declared reused-development experiment only',UTC=utc(),
              contract_sha256=clock['contract_sha256'],config_sha256=sha(R/'CONFIG.json'),
              source_seal_sha256=sha(R/'SOURCE_SEAL.json'),preparation_sha256=sha(R/'PREPARATION_AUDIT.json'),
              packet_manifests=prep['packet_manifests'],
              checks=checks,independent_confirmation_available=False,
              limitations=['All outer rows have historical development exposure.','Upstream snapshot completeness is inherited, not newly certified.']),R/'GATE.json')
    print(json.dumps(dict(stage='qualification_passed',pilot_seconds={x['task']:x['seconds']for x in pilots})),flush=True)


if __name__=='__main__':
    main()
