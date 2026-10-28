"""Independent artifact/arithmetic readback; never train or change selection."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'source'))
from common import check_seal, load, dump, sha, ids_sha, utc, catalog, tasks

def remaining(seconds=60):
    assert time.time() < load('CLOCK.json')['deadline_epoch'] - seconds, 'Original audit budget exhausted'

def ks(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0], method='asymp').statistic)

def raw_audit():
    remaining()
    clock = check_seal()
    packets = {}
    for fold in range(5):
        root = R / f'packets/fold{fold}'
        m = json.loads((root / 'MANIFEST.json').read_text())
        for filename, key in [('roles.npz', 'roles_sha256'), ('features.parquet', 'feature_sha256'), ('all_features.parquet', 'full_feature_sha256'), ('inner_roles.npz', 'inner_roles_sha256')]:
            assert sha(root / filename) == m[key], (fold, filename)
        role = np.load(root / 'roles.npz')
        ctx, meta, outer = role['context'], role['correction'], role['validation']
        assert (len(ctx), len(meta), len(outer)) == (50000, 6000, 14000)
        assert len(np.unique(np.r_[ctx, meta, outer])) == 70000
        assert not {'y_outer', 'y_validation', 'target', 'labels'} & set(role.files)
        inner, stop = train_test_split(np.arange(50000), test_size=.15, stratify=role['y_context'], random_state=20261029 + fold)
        assert ids_sha(ctx[inner]) == m['inner_train_rows_sha256']
        assert ids_sha(ctx[stop]) == m['stopping_rows_sha256']
        assert m['feature_statistics_exclude_stopping'] and m['supervised_selection_excludes_stopping']
        assert not m['correction_or_outer_labels_used_for_feature_selection']
        packets[fold] = (m, role, sha(root / 'MANIFEST.json'))
    expected, designs = tasks(), catalog()
    retained, by_design = {}, {}
    retries = load('APPROVED_RETRIES.json')
    for path in sorted((R / 'runs').glob('*/attempt*/RESULT.json')):
        remaining()
        result = json.loads(path.read_text())
        folder = path.parent
        task_id = folder.parent.name
        assert task_id in expected and result['task_id'] == task_id
        assert folder.name == 'attempt' + str(retries.get(task_id, 1)), 'Superseded success cannot enter current cohort'
        start = json.loads((folder / 'START.json').read_text())
        t = expected[task_id]
        assert start['task'] == t and start['design'] == designs[t['design']]
        assert start['seed'] == t['seed'] + t['fold']
        assert result['status'] == 'complete' and not start['pilot_parameters_changed']
        assert sha(folder / 'START.json') == result['start_sha256']
        assert sha(folder / 'prediction.npz') == result['prediction_sha256']
        assert start['source_sha256'] == sha(R / 'source/worker.py')
        for filename, key in [('CLOCK.json', 'clock_sha256'), ('CONFIG.json', 'config_sha256'), ('VALIDATION_CONTRACT.json', 'contract_sha256'), ('SOURCE_SEAL.json', 'source_seal_sha256')]:
            assert start[key] == sha(R / filename)
        assert not start['outer_labels_loaded'] and not start['correction_labels_loaded']
        assert not start['query_labels_used_for_early_stopping']
        assert not result['weights_saved'] and not result['KV_cache_saved']
        m, role, manifest_hash = packets[t['fold']]
        assert start['packet_manifest_sha256'] == manifest_hash
        assert start['columns'] == m['views'][designs[t['design']]['view']]
        assert start['feature_sha256'] == m['full_feature_sha256' if start['feature_table_filename'] == 'all_features.parquet' else 'feature_sha256']
        assert start['context_rows_sha256'] == ids_sha(role['context'])
        assert start['context_labels_sha256'] == ids_sha(role['y_context'])
        assert start['declared_inner_train_rows_sha256'] == m['inner_train_rows_sha256']
        assert start['declared_inner_stop_rows_sha256'] == m['stopping_rows_sha256']
        audit = result['audit']
        for key, expected_hash in [('inner_train_rows_sha256', m['inner_train_rows_sha256']), ('stopping_rows_sha256', m['stopping_rows_sha256'])]:
            if key in audit:
                assert audit[key] == expected_hash
        with np.load(folder / 'prediction.npz') as saved:
            assert np.array_equal(saved['rows'], np.r_[role['correction'], role['validation']])
            assert saved['raw'].shape == (20000,) and np.isfinite(saved['raw']).all()
            assert np.all((saved['raw'] >= 0) & (saved['raw'] <= 1))
        retained[task_id] = dict(result_sha256=sha(path), prediction_sha256=result['prediction_sha256'], start_sha256=result['start_sha256'])
        by_design[t['design']] = by_design.get(t['design'], 0) + 1
    failures = {str(p.relative_to(R)): sha(p) for p in (R / 'runs').glob('*/attempt*/FAILURE.json')}
    assert len(retained) == len(list((R / 'runs').glob('*/attempt*/RESULT.json')))
    return dict(status='PASS', UTC=utc(), expert_fits_verified=len(retained), failure_attempts=len(failures),
                expected_fits=len(expected), complete_designs=sorted(k for k,v in by_design.items() if v == 10),
                pending_designs={k:by_design.get(k,0) for k in designs if by_design.get(k,0) != 10},
                result_evidence=retained, failure_evidence=failures,
                source_seal_sha256=sha(R / 'SOURCE_SEAL.json'), clock_sha256=sha(R / 'CLOCK.json')) , packets

def conditional_interval(y, p, baseline, folds):
    rng = np.random.default_rng(20261029)
    strata = [np.flatnonzero((folds == f) & (y == label)) for f in range(5) for label in [0,1]]
    orders = [np.argsort(v, kind='stable') for v in [p,baseline]]
    ends = [np.flatnonzero(np.r_[v[o][1:] != v[o][:-1],True]) for v,o in zip([p,baseline],orders)]
    values=[]
    for repetition in range(2000):
        if repetition % 25 == 0:
            remaining()
        sample=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])
        counts=np.bincount(sample,minlength=len(y))
        positive=counts*(y==1);negative=counts*(y==0)
        scores=[]
        for o,e in zip(orders,ends):
            difference=np.cumsum(positive[o])/positive.sum()-np.cumsum(negative[o])/negative.sum()
            scores.append(float(np.max(np.abs(difference[e]))))
        values.append(scores[0]-scores[1])
    return np.quantile(values,[.025,.975])

def main(preflight=False):
    report, packets = raw_audit()
    if preflight:
        dump(report,R/'RAW_ARTIFACT_PREFLIGHT.json')
        print(json.dumps({k:report[k] for k in ['status','expert_fits_verified','failure_attempts','pending_designs']}))
        return
    final=load('FINAL_EVALUATION.json'); selection=load('FINAL_SELECTION.json')
    assert final['final_selection_sha256']==sha(R/'FINAL_SELECTION.json')
    snapshot=load('evaluation/FINAL_SCOREBOARD_SNAPSHOT.json')
    assert sha(R/'evaluation/FINAL_SCOREBOARD_SNAPSHOT.json')==selection['source_scoreboard_sha256']==final['source_scoreboard_sha256']
    chosen=min(snapshot['candidates'],key=lambda k:(-snapshot['candidates'][k]['metrics']['KS'],k))
    assert chosen==final['selected_cohort']==selection['identity']['cohort']
    assert selection['identity']['cohort_result_sha256']==sha(R/f'evaluation/cohorts/{chosen}/RESULT.json')
    target=np.load(R/'evaluation_inputs/target.npy');saved=np.load(R/'evaluation_inputs/baseline.npz')
    baseline,folds=saved['prediction'],saved['fold']
    assert target.shape==baseline.shape==folds.shape==(70000,)
    assert abs(ks(target,baseline)-.5306361042400267)<1e-12
    for fold,(_,role,_) in packets.items():
        assert set(role['validation'])==set(np.flatnonzero(folds==fold))
        assert np.array_equal(role['y_context'],target[role['context']])
    assert sorted(snapshot['complete_designs'])==report['complete_designs']
    modes=snapshot['expected_modes'];cohorts=0;chosen_prediction=None
    for design in report['complete_designs']:
        values={mode:[] for mode in modes}
        for seed in load('CONFIG.json')['seeds']:
            full={mode:np.full(70000,np.nan) for mode in modes};coverage=np.zeros(70000,dtype=np.uint8)
            for fold in range(5):
                root=R/f'composed/{design}/s{seed}/fold{fold}';result=json.loads((root/'RESULT.json').read_text())
                assert sha(root/'prediction.npz')==result['prediction_sha256']
                assert sha(root/'FROZEN.json')==result['frozen_sha256']
                assert not result['outer_labels_loaded'] and not result['fitted_weights_saved']
                with np.load(root/'prediction.npz') as data:
                    rows=data['rows'];assert np.array_equal(rows,packets[fold][1]['validation'])
                    assert not coverage[rows].any();coverage[rows]+=1
                    for mode in modes:full[mode][rows]=data[mode]
            assert np.all(coverage==1)
            for mode in modes:values[mode].append(full[mode])
        for mode in modes:
            remaining();key=design+'__'+mode;root=R/'evaluation/cohorts'/key
            result=json.loads((root/'RESULT.json').read_text());frozen=json.loads((root/'FROZEN.json').read_text())
            assert result==snapshot['candidates'][key]
            assert sha(root/'prediction.npz')==result['prediction_sha256']==frozen['prediction_sha256']
            assert sha(root/'FROZEN.json')==result['frozen_sha256']
            mean=np.mean(values[mode],axis=0)
            with np.load(root/'prediction.npz') as data:
                assert np.array_equal(data['rows'],np.arange(70000)) and np.array_equal(mean,data['prediction'])
            assert abs(ks(target,mean)-result['metrics']['KS'])<1e-12
            seed_delta=[ks(target,p)-ks(target,baseline) for p in values[mode]]
            fold_delta=[ks(target[folds==f],mean[folds==f])-ks(target[folds==f],baseline[folds==f]) for f in range(5)]
            assert np.allclose(seed_delta,result['seed_delta_KS'],atol=1e-12,rtol=0)
            assert np.allclose(fold_delta,result['fold_delta_KS'],atol=1e-12,rtol=0)
            if key==chosen:chosen_prediction=mean
            cohorts+=1
    assert chosen_prediction is not None
    interval=conditional_interval(target,chosen_prediction,baseline,folds)
    assert np.allclose(interval,final['uncertainty']['percentile95'],atol=1e-12,rtol=0)
    p=final['primary_result']
    supported=bool(p['delta_KS']>=.0005 and interval[0]>0 and p['positive_folds']>=4 and p['positive_seeds']==2)
    assert supported==final['primary_development_gain_supported']
    for vm in [4,5,6]:assert load(f'GPU_COLLECTION_VM{vm}_COMPLETE.json')['released']
    check_seal();remaining()
    report.update(UTC=utc(),scope='Artifact and arithmetic consistency only; no independent statistical confirmation.',
        all_cohort_scores_recomputed_with_SciPy_KS=True,complete_cohorts_verified=cohorts,
        baseline_recomputed_KS=ks(target,baseline),primary_conditional_interval_recomputed=interval.tolist(),
        primary_acceptance_rule_recomputed=supported,final_selection_sha256=sha(R/'FINAL_SELECTION.json'),
        final_evaluation_sha256=sha(R/'FINAL_EVALUATION.json'),scientific_selection_changed=False,independent_audit_available=False)
    dump(report,R/'ARTIFACT_CONSISTENCY_AUDIT.json')
    print(json.dumps(dict(status='PASS',cohorts=cohorts,fits=report['expert_fits_verified'],selected=chosen)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    main(args.preflight)
