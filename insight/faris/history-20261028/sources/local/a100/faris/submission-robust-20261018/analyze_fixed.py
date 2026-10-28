"""Multiplicity-adjusted secondary comparison of the six frozen fixed recipes.

This does not relabel the failed primary nested selector as successful.
Historical shortlist reuse remains; these are conditional, post-primary results.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import concurrent.futures as cf
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'submission-final-20261018'))
from audit_generalization import ordered_statistic,weighted_ks


def ks(y,p):return weighted_ks(y,np.ones(len(y)),ordered_statistic(y,p))


def bootstrap_fold(task):
    f,y,base,options,repetitions=task
    order_b=ordered_statistic(y,base);orders=[ordered_statistic(y,p) for p in options]
    classes=[np.flatnonzero(y==c) for c in [0,1]];rng=np.random.default_rng(20261021+f)
    values=np.empty((repetitions,len(options)))
    for i in range(repetitions):
        ii=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in classes]);w=np.bincount(ii,minlength=len(y));b=weighted_ks(y,w,order_b)
        for j,order in enumerate(orders):values[i,j]=weighted_ks(y,w,order)-b
    return values


def main():
    plan=json.loads((ROOT/'benchmark_plan.json').read_text());primary=json.loads((ROOT/'benchmark_result.json').read_text())
    assert not primary['promotion_gate_passed'],'This secondary branch follows a recorded failed primary procedure'
    protocol={'state':'post_primary_secondary_analysis','frozen_candidates':[c['id'] for c in plan['candidate_components']],
        'frozen_weights':plan['candidate_weights'],'comparisons':6,'bootstrap_samples':4000,'alpha':.05,
        'method':'Paired applicant bootstrap stratified by outer fold and target, mean within-fold KS, Bonferroni six-comparison percentile intervals',
        'selection':'Among fixed recipes positive on all three folds and pooled delta>=-0.003, select highest simultaneous lower bound; prefer smaller weight on a tie.',
        'prior_observation':'The failed primary output already showed addition2 at 10% positive on all three full-population cross-fit folds. This is not a fresh independent confirmation.',
        'no_new_training_or_hyperparameters':True,'primary_result_is_not_rewritten':True,
        'historical_adaptation_not_corrected':True}
    protocol_path=ROOT/'evidence/fixed_secondary_protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text())==protocol,'Do not revise the secondary protocol after outcomes'
        result_path=ROOT/'fixed_comparison.json'
        assert result_path.exists(),'Interrupted analysis retained; do not silently replace its protocol'
        print(result_path.read_text());return
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    z=np.load(ROOT/'evidence/nested_oof.npz');y=z['target'];fold=z['fold'];base=z['champion'];options=[];specs=[]
    for c in plan['candidate_components']:
        for w in plan['candidate_weights']:
            specs.append({'component':c['id'],'original_id':c['original_id'],'weight':w})
            options.append((1-w)*base+w*z[c['id']])
    assert len(options)==6
    tasks=[(f,y[fold==f],base[fold==f],[p[fold==f] for p in options],4000) for f in range(3)]
    with cf.ProcessPoolExecutor(max_workers=3) as pool:draws=sum(pool.map(bootstrap_fold,tasks))/3
    comparisons=[];baseline=ks(y,base)
    for j,(spec,p) in enumerate(zip(specs,options)):
        ds=[ks(y[fold==f],p[fold==f])-ks(y[fold==f],base[fold==f]) for f in range(3)]
        mean=float(np.mean(ds));interval=np.quantile(draws[:,j],[.05/(2*6),1-.05/(2*6)]).tolist();pooled=ks(y,p)
        comparisons.append({**spec,'mean_fold_delta':mean,'fold_delta':ds,'pooled_ks':pooled,'baseline_pooled_ks':baseline,
            'pooled_delta':pooled-baseline,'familywise_95ci':interval,'eligible':bool(min(ds)>0 and pooled-baseline>=-.003),
            'strict_positive_conditional_bound':interval[0]>0})
    eligible=[r for r in comparisons if r['eligible']];assert eligible,'No fixed candidate has a stable positive fold contrast'
    eligible.sort(key=lambda r:(r['familywise_95ci'][0],-r['weight'],r['mean_fold_delta']),reverse=True);selected=eligible[0]
    np.savez_compressed(ROOT/'evidence/fixed_selected_oof.npz',rows=z['rows'],target=y,fold=fold,champion=base,prediction=options[specs.index({k:selected[k] for k in ['component','original_id','weight']})])
    result={'selected':selected,'comparisons':comparisons,'primary_nested_gate_passed':False,
        'benchmark_type':'Fixed pre-existing configuration with fixed probability weight; post-primary six-comparison analysis',
        'conditional_familywise_adjustment':True,'historical_selection_reuse':True,'independent_private_validation':False,
        'quality_status':'Conditional lower bound positive' if selected['strict_positive_conditional_bound'] else 'Exploratory candidate; interval includes losses',
        'authorized_output':'User-requested candidate CSV; does not promote it to proven champion or upload it'}
    (ROOT/'fixed_comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
