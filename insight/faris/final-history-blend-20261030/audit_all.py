"""Independent row/hash/KS replay and conditional uncertainty after union freeze."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import hashlib
import itertools
import json
import time
import numpy as np
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata

R = Path(__file__).resolve().parent
BASE = Path('/mnt/home/faris/logika-twelvehour-cross-20261030')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    p.parent.mkdir(exist_ok=True, parents=True)
    q = p.with_name(p.name + '.audit.tmp')
    q.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    q.replace(p)


def cohorts():
    for namespace in ('results', 'greedy_results'):
        parent = R / namespace
        if not parent.is_dir():
            continue
        for folder in parent.iterdir():
            parts = [folder / f'fold{f}' for f in range(5)]
            if all((p / 'FROZEN.json').is_file() for p in parts):
                yield namespace + '/' + folder.name, parts, False
    parent = R / 'gpu_results'
    if parent.exists():
        for folder in parent.iterdir():
            parts = [folder / f's{seed}' / f'fold{fold}' for seed in (2026, 7731) for fold in range(5)]
            if all((p / 'FROZEN.json').exists() for p in parts):
                yield 'gpu_results/' + folder.name, parts, True


def controls():
    for bank in ('all', 'family'):
        outputs = {f'fixed_{bank}_{kind}_w{weight}': np.full(70000, np.nan)
                   for kind, weight in itertools.product(('probability', 'logit', 'rank'), (.05, .1, .2, .5, 1.))}
        for fold in range(5):
            z = np.load(R / f'bank/meta{fold}.npz')
            raw = np.load(R / 'bank' / (f'fold{fold}.npy' if bank == 'all' else f'family{fold}.npy'), mmap_mode='r')
            means = dict(probability=np.mean(raw[6000:], axis=1, dtype=np.float64),
                         logit=expit(np.mean(logit(np.clip(raw[6000:].astype(float), 1e-6, 1 - 1e-6)), axis=1)),
                         rank=np.mean(np.column_stack([rankdata(raw[:, j], method='average') / 20000 for j in range(raw.shape[1])]), axis=1)[6000:])
            for kind, weight in itertools.product(('probability', 'logit', 'rank'), (.05, .1, .2, .5, 1.)):
                outputs[f'fixed_{bank}_{kind}_w{weight}'][z['rows']] = (1 - weight) * z['anchor'][6000:] + weight * means[kind]
        for name, p in outputs.items():
            assert np.isfinite(p).all()
            folder = R / 'controls' / name
            folder.mkdir(exist_ok=True, parents=True)
            if (folder / 'FROZEN.json').exists():
                previous = json.loads((folder / 'FROZEN.json').read_text())
                assert sha(folder / 'prediction.npz') == previous['prediction_sha256']
                assert np.array_equal(np.load(folder / 'prediction.npz')['prediction'], p)
                continue
            np.savez_compressed(folder / 'prediction.npz', rows=np.arange(70000), prediction=p)
            dump(folder / 'FROZEN.json', dict(recipe=name, prediction_sha256=sha(folder/'prediction.npz'),
                 outer_labels_loaded=False, no_parameters_fitted=True, bank_frozen_sha256=sha(R/'BANK_FROZEN.json')))


def interval(y, selected, reference, folds):
    rng = np.random.default_rng(20261030)
    strata = [np.flatnonzero((folds == f) & (y == k)) for f in range(5) for k in (0, 1)]
    orders, ends = [], []
    for p in (selected, reference):
        order = np.argsort(p, kind='stable')
        orders.append(order)
        ends.append(np.flatnonzero(np.r_[p[order][1:] != p[order][:-1], True]))
    deltas = []
    for repeat in range(2000):
        assert time.time() < json.loads((R/'PLAN.json').read_text())['deadline_epoch'] - 20
        sample = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        count = np.bincount(sample, minlength=70000)
        value = count * (y/y.sum() - (1-y)/(len(y)-y.sum()))
        a, b = [np.max(abs(np.cumsum(value[order])[end])) for order, end in zip(orders, ends)]
        deltas.append(float(a-b))
    return dict(replicates=2000, seed=20261030, percentile95=np.quantile(deltas,[.025,.975]).tolist(),
                excludes_adaptive_selection_and_refitting=True, independent_confirmation=False)


def main():
    if (R/'FINAL_AUDIT.json').exists():
        print('Existing terminal audit preserved; no repeated final assessment')
        return
    assert time.time() < json.loads((R/'PLAN.json').read_text())['deadline_epoch'] - 180
    controls()  # Freeze all fixed controls before opening outer targets.
    frozen_bank = json.loads((R/'BANK_FROZEN.json').read_text())
    for mapping in frozen_bank['bank_hashes'].values():
        for name, digest in mapping.items():
            assert sha(R/'bank'/name) == digest
    raw_records, rejected = [], []
    candidates = {}
    for name, parts, two_seeds in cohorts():
        receipts = [json.loads((p/'FROZEN.json').read_text()) for p in parts]
        if any(v.get('eligible_for_scoring') is False for v in receipts):
            rejected.append(dict(cohort=name, reason='optimizer_convergence_limit_not_predictive_negative'))
            continue
        probability = np.zeros(70000)
        coverage = np.zeros(70000, int)
        for folder, value in zip(parts, receipts):
            assert value['outer_labels_loaded'] is False
            assert sha(folder/'prediction.npz') == value['prediction_sha256']
            a = np.load(folder/'prediction.npz')
            assert a['prediction'].shape == a['rows'].shape == (14000,) and np.isfinite(a['prediction']).all()
            probability[a['rows']] += a['prediction'] / (2 if two_seeds else 1)
            coverage[a['rows']] += 1
            raw_records.append(dict(path=str(folder.relative_to(R)), frozen_sha256=sha(folder/'FROZEN.json'), prediction_sha256=sha(folder/'prediction.npz')))
        assert np.all(coverage == (2 if two_seeds else 1))
        folder = R/'union_predictions'/name
        folder.mkdir(exist_ok=True, parents=True)
        if (folder/'FROZEN.json').exists():
            old=json.loads((folder/'FROZEN.json').read_text());assert sha(folder/'prediction.npz')==old['prediction_sha256']
            assert np.array_equal(np.load(folder/'prediction.npz')['prediction'],probability)
        else:
            np.savez_compressed(folder/'prediction.npz', rows=np.arange(70000), prediction=probability)
            dump(folder/'FROZEN.json',dict(source_freezes=[sha(p/'FROZEN.json')for p in parts],prediction_sha256=sha(folder/'prediction.npz'),frozen_before_outer_score=True))
        candidates[name] = folder/'prediction.npz'
    for folder in (R/'controls').iterdir():
        a = json.loads((folder/'FROZEN.json').read_text())
        assert a['prediction_sha256'] == sha(folder/'prediction.npz')
        candidates['controls/'+folder.name] = folder/'prediction.npz'
    y = np.load(BASE/'evaluation_inputs/target.npy').astype(int)
    baseline = np.load(BASE/'evaluation_inputs/baseline.npz')['prediction']
    previous = np.load(BASE/'evaluation_inputs/previous.npz')['prediction']
    ks = lambda p: float(ks_2samp(p[y==0],p[y==1],method='asymp').statistic)
    assert abs(ks(baseline)-.5306361042400267)<1e-12 and abs(ks(previous)-.5316414268134972)<1e-12
    rows = [np.load(R/f'bank/meta{f}.npz')['rows'] for f in range(5)]
    fold_id = np.empty(70000, int)
    for f, ix in enumerate(rows):fold_id[ix] = f
    records = []
    for name, path in candidates.items():
        p = np.load(path)['prediction']
        score = ks(p)
        d = [float(ks_2samp(p[ix][y[ix]==0],p[ix][y[ix]==1],method='asymp').statistic - ks_2samp(previous[ix][y[ix]==0],previous[ix][y[ix]==1],method='asymp').statistic) for ix in rows]
        records.append(dict(cohort=name, KS=score, delta_current=score-ks(previous), delta_baseline=score-ks(baseline), fold_delta_current=d, prediction_sha256=sha(path), prediction_path=str(path)))
    records.sort(key=lambda d:(-d['KS'],d['cohort']))
    selected = records[0]
    if (R/'UNION_SELECTION.json').exists():
        assert json.loads((R/'UNION_SELECTION.json').read_text())['selected']==selected, 'Final union choice cannot change after conditional intervals'
    else:
        dump(R/'UNION_SELECTION.json', dict(selected=selected, rule='Largest complete eligible blend KS;lexicalties',
             frozen_before_conditional_interval=True, source_sha256=sha(__file__), all70k_adaptive_development=True))
    probability = np.load(selected['prediction_path'])['prediction']
    prior_ci, baseline_ci = interval(y,probability,previous,fold_id), interval(y,probability,baseline,fold_id)
    result = dict(status='audited_adaptive_history_blend_assessment', epoch=time.time(), best=selected,
         conditional_interval_current=prior_ci, conditional_interval_baseline=baseline_ci,
         candidates=records, complete_blend_recipes=len(records), excluded=rejected,
         source_bank_complete_pipelines=frozen_bank['eligible_complete_pipelines'],
         fold_prediction_packets=len(raw_records), frozen_packet_records=raw_records,
         artifact_hash_row_SciPy_replay='PASS', untouched_audit_available=False,
         target_054_reached=selected['KS']>=.54,
         supported_new_gain=bool(selected['delta_current']>=.0005 and prior_ci['percentile95'][0]>0 and sum(d>0 for d in selected['fold_delta_current'])>=4),
         full_data_refits_performed=False, new_Kaggle_submission=False,
         test_prediction_availability='Whole1925basebank lacks matching fulltestpredictions for everymember; unscored historicalCSV blends are separate, no hiddentestKS inferred.',
         limitations='All70k adaptive reused; CIconditional on fittedpreds, excludeshistoricalselection/refitting. DeterministicCPUblends average bothbase seeds beforecombining; GPUhead seeds reflect metamodel RNG, not independent base reruns.')
    dump(R/'FINAL_AUDIT.json',result)
    print(json.dumps(dict(best=selected, interval_current=prior_ci, complete_recipes=len(records))),flush=True)


if __name__=='__main__':main()
