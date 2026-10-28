"""Terminal development evaluation of the already frozen MoE comparison."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='2'
import json
import time
from pathlib import Path
import numpy as np
from common import R,load,dump,sha,metrics,utc,check_seal


def ks_numpy(y,p,weights=None):
    order=np.argsort(p,kind='stable');values=p[order]
    ends=np.flatnonzero(np.r_[values[1:]!=values[:-1],True])
    w=np.ones(len(y))if weights is None else weights
    positive=w*y;negative=w*(1-y)
    delta=np.cumsum((positive/positive.sum()-negative/negative.sum())[order])
    return float(np.max(abs(delta[ends])))


def paired(y,p,q,fold,repetitions):
    rng=np.random.default_rng(20261028);strata=[np.flatnonzero((fold==f)&(y==c))for f in range(5)for c in [0,1]]
    orders=[]
    for value in [p,q]:
        order=np.argsort(value,kind='stable');ends=np.flatnonzero(np.r_[value[order][1:]!=value[order][:-1],True]);orders.append((order,ends))
    differences=[];pos=y.sum();neg=len(y)-pos
    for _ in range(repetitions):
        draw=np.concatenate([rng.choice(s,len(s),replace=True)for s in strata]);weight=np.bincount(draw,minlength=len(y))
        contribution=weight*(y/pos-(1-y)/neg);scores=[]
        for order,ends in orders:scores.append(float(np.max(abs(np.cumsum(contribution[order])[ends]))))
        differences.append(scores[0]-scores[1])
    return dict(delta_KS=ks_numpy(y,p)-ks_numpy(y,q),percentile95=np.quantile(differences,[.025,.975]).tolist(),
                replicates=repetitions,seed=20261028,unit='application resampled within fold/class',
                limitation='Conditional on fitted predictions; excludes retraining, previous research selection, and public leaderboard uncertainty.')


def main():
    clock=check_seal();frozen=load('FROZEN.json');assert frozen['selection_after_outer_scores']is False
    assert frozen['config_sha256']==sha(R/'CONFIG.json') and frozen['contract_sha256']==clock['contract_sha256']
    for path,digest in frozen['composed_results'].items():assert sha(R/path)==digest
    y=np.load(R/'evaluation_inputs/target.npy');baseline=np.load(R/'evaluation_inputs/baseline.npz')
    p0=baseline['prediction'];fold=baseline['fold'];config=load('CONFIG.json');output=R/'evaluation'
    if(output/'RESULT.json').exists():
        previous=json.loads((output/'RESULT.json').read_text());assert previous['frozen_sha256']==sha(R/'FROZEN.json')
        print(json.dumps(dict(stage='terminal_evaluation_replayed',status=previous['status'])),flush=True);return
    if output.exists():
        failures=R/'failures';failures.mkdir(exist_ok=True);output.rename(failures/f'evaluation_{time.time_ns()}')
    output.mkdir()
    with(R/'EXPOSURES.jsonl').open('a')as stream:
        stream.write(json.dumps(dict(UTC=utc(),role='reused_development_terminal',rows=70000,frozen_sha256=sha(R/'FROZEN.json'),
                                    selection_already_frozen=True,independent_audit=False))+'\n')
    assert abs(ks_numpy(y,p0)-metrics(y,p0)['KS'])<1e-12
    all_predictions={};seed_predictions={};results={};missing=[]
    for family in config['base_families']+['hybrid']:
        for condition in (['existing','expanded','wide'] if family in ['lightgbm','catboost'] else ['existing','expanded']):
            for combination in ['fixed','gated']:
                name=family+'__'+condition+'__'+combination;seeds=[]
                for seed in config['seeds']:
                    p=np.full(len(y),np.nan)
                    for f in range(5):
                        root=R/f'composed/{family}/{condition}/s{seed}/fold{f}'
                        if not(root/'RESULT.json').exists():break
                        done=json.loads((root/'RESULT.json').read_text());assert sha(root/'prediction.npz')==done['prediction_sha256']
                        z=np.load(root/'prediction.npz');assert np.array_equal(z['rows'],np.flatnonzero(fold==f))or set(z['rows'])==set(np.flatnonzero(fold==f))
                        p[z['rows']]=z[combination]
                    if not np.isfinite(p).all():break
                    seeds.append(p)
                if len(seeds)!=len(config['seeds']):missing.append(name);continue
                p=np.mean(seeds,axis=0);all_predictions[name]=p;seed_predictions[name]=seeds
                score=metrics(y,p);assert abs(score['KS']-ks_numpy(y,p))<1e-12
                result=dict(metrics=score,delta_KS=score['KS']-metrics(y,p0)['KS'],
                            fold_delta_KS=[ks_numpy(y[fold==f],p[fold==f])-ks_numpy(y[fold==f],p0[fold==f])for f in range(5)],
                            seed_delta_KS=[ks_numpy(y,s)-ks_numpy(y,p0)for s in seeds])
                results[name]=result
    primary=config['primary'];supported=False;contrasts={}
    if primary in all_predictions:
        p=all_predictions[primary];r=results[primary];uncertainty=paired(y,p,p0,fold,2000);r['uncertainty']=uncertainty
        for label,control in [('routing','kumo__expanded__fixed'),('expansion','kumo__existing__gated')]:
            if control in all_predictions:contrasts[label]=paired(y,p,all_predictions[control],fold,2000)
        supported=(r['delta_KS']>=.0005 and uncertainty['percentile95'][0]>0 and
                   sum(v>0 for v in r['fold_delta_KS'])>=4 and all(v>0 for v in r['seed_delta_KS']))
    dump(dict(status='complete'if not missing else'partial',UTC=utc(),baseline=metrics(y,p0),
              baseline_identity='Audited maturity-adjusted fold-local reference; nominal public V25 remains separate.',
              public_incumbent=dict(id='V25',KS=.56822),primary=primary,primary_result=results.get(primary),
              primary_development_gain_supported=bool(supported),mechanism_contrasts=contrasts,
              local_target_054_point_reached=primary in results and results[primary]['metrics']['KS']>=.54,
              results=results,incomplete_cohorts=missing,independent_confirmation=False,
              previous_outer_exposure='All70k rows are reused development; new nested boundaries do not undo historical adaptation.',
              refit_status='requires_frozen_primary_refit_in_original_remaining_reserve'if supported else'not_justified',
              submission_made=False,weights_saved=False,clock_deadline_epoch=clock['deadline_epoch'],
              frozen_sha256=sha(R/'FROZEN.json')),output/'RESULT.json')
    np.savez_compressed(output/'predictions.npz',**all_predictions)
    text=['# Feature-subspace MoE results','',
          'Reused development evidence; no untouched audit or inferred public gain.','',
          '| Cohort | KS | Delta vs matched anchor |','|---|---:|---:|']
    for name,r in results.items():text.append(f"| {name} | {r['metrics']['KS']:.10f} | {r['delta_KS']:+.10f} |")
    text+=['',f'Frozen primary: `{primary}`. Supported development gain: {bool(supported)}.',
           'Public V25 remains0.56822. No Kaggle submission performed.',
           'Missing cohorts are incomplete execution evidence, not negative model results.']
    (output/'RESULTS.md').write_text('\n'.join(text)+'\n')
    if supported:dump(dict(primary=primary,result_sha256=sha(output/'RESULT.json'),
                            deadline_epoch=clock['deadline_epoch'],scientific_changes_allowed=False,
                            policy=config['refit_policy']),R/'REFIT_REQUIRED.json')
    print(json.dumps(dict(stage='terminal_evaluation',primary=results.get(primary),supported=bool(supported),
                         incomplete=len(missing))),flush=True)


if __name__=='__main__':
    main()
