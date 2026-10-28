"""Bounded second pass: fixed OOF combinations, seed verification, one confirmation."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import concurrent.futures as cf
import contextlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import sys
import time
import traceback
import joblib
import numpy as np
import pandas as pd
import psutil
from scipy.stats import ks_2samp
from metaexp.features import Bank
from metaexp.pipeline import fit_predict, predict
from metaexp.common import read, write, sha, identity, json_safe

ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT.parent / 'submission-research-20261017'


def ks(y, p):
    assert len(y) == len(p) and np.isfinite(p).all()
    return float(ks_2samp(p[y == 0], p[y == 1]).statistic)


def combine(spec, arrays):
    x = np.column_stack([arrays[k] for k in spec['parents']])
    assert np.isfinite(x).all() and ((x >= 0) & (x <= 1)).all()
    if spec['method'] == 'trimmed':
        assert x.shape[1] >= 3
        return np.sort(x, axis=1)[:, 1:-1].mean(axis=1)
    if spec['method'] == 'mean':
        return x.mean(axis=1)
    if spec['method'] == 'geometric':
        return np.exp(np.log(np.clip(x, 1e-8, 1)).mean(axis=1))
    if spec['method'] == 'logit':
        x = np.clip(x, 1e-8, 1-1e-8)
        logits = np.log(x / (1-x)).mean(axis=1)
        return 1 / (1 + np.exp(-logits))
    raise ValueError(spec['method'])


def describe(y, p, folds, baseline):
    per = [ks(y[folds == f], p[folds == f]) for f in range(3)]
    old = [ks(y[folds == f], baseline[folds == f]) for f in range(3)]
    delta = np.array(per) - old
    score = ks(y, p)
    return {'ks': score, 'fold_ks': per, 'fold_delta': delta.tolist(),
            'delta': score - ks(y, baseline),
            'gate': bool((delta >= -1e-12).sum() >= 2 and delta.min() >= -.003 and score >= ks(y, baseline))}


def train_job(task):
    if time.time() >= task['deadline']:
        raise TimeoutError('Second-pass deadline expired')
    folder = ROOT / 'fits' / task['mode'] / task['key'] / str(task['seed']) / str(task['fold'])
    folder.mkdir(parents=True, exist_ok=True)
    request = {k: v for k, v in task.items() if k != 'deadline'}
    if (folder / 'complete.json').exists():
        assert read(folder / 'request.json') == request
        for name, checksum in read(folder / 'complete.json')['files'].items():
            assert sha(folder / name) == checksum
        return str(folder)
    bank = Bank(Path(task['bank_root']))
    mode, fold = task['mode'], task['fold']
    if mode == 'cv':
        ti, vi = np.flatnonzero(bank.folds != fold), np.flatnonzero(bank.folds == fold)
        train_rows, query_rows, labels = bank.dev[ti], bank.dev[vi], bank.y[ti]
    elif mode == 'confirmation':
        train_rows, labels = bank.dev, bank.y
        query_rows = np.load(bank.root / 'confirmation_rows.npy')
    elif mode == 'full':
        labels_table = pd.read_csv(PRIOR / 'inputs/train_applications.csv').set_index('application_id').target
        train_rows = np.flatnonzero(bank.metadata.split.to_numpy() == 'train')
        query_rows = np.flatnonzero(bank.metadata.split.to_numpy() == 'test')
        labels = labels_table.reindex(bank.metadata.application_id.iloc[train_rows]).to_numpy(dtype=int)
        assert len(train_rows) == 70000 and len(query_rows) == 30000
        assert np.array_equal(labels_table.reindex(bank.metadata.application_id.iloc[bank.dev]).to_numpy(), bank.y)
    else:
        raise ValueError(mode)
    assert not np.intersect1d(train_rows, query_rows).size
    write(folder / 'request.json', request)
    write(folder / 'status.json', {'state': 'running', 'pid': os.getpid(), 'started': time.time()})
    started = time.time()
    try:
        with (folder / 'training.log').open('w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            p, pre, model, weight, cdf, warnings = fit_predict(bank, task['recipe'], train_rows, labels, query_rows, 'cpu', task['seed'] + (fold if mode == 'cv' else 0))
            assert p.shape == (len(query_rows),) and np.isfinite(p).all()
            np.save(folder / 'prediction.npy', p)
            np.save(folder / 'query_rows.npy', query_rows)
            write(folder / 'audit.json', {'preprocessing': pre.audit, 'parameters': json_safe(model.get_params()), 'warnings': warnings, 'fit_rows_sha256': identity(train_rows.tolist()), 'query_rows_sha256': identity(query_rows.tolist()), 'elapsed_seconds': time.time()-started, 'host': os.uname().nodename})
            pre.bank = None
            joblib.dump({'preprocessing': pre, 'model': model, 'weight': weight}, folder / 'pipeline.joblib', compress=1)
        write(folder / 'complete.json', {'files': {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name not in ['complete.json', 'status.json']}})
        write(folder / 'status.json', {'state': 'completed'})
        return str(folder)
    except BaseException:
        write(folder / 'status.json', {'state': 'failed', 'traceback': traceback.format_exc()})
        raise


def run_jobs(plan, spec, seeds, mode, deadline):
    tasks = [dict(key=k, recipe=plan['recipes'][k], seed=seed, mode=mode, fold=fold,
                  bank_root=plan['bank_root'], bank_id=plan['bank_id'], deadline=deadline,
                  source_manifest_sha256=sha(ROOT / 'source_manifest.json'))
             for seed in seeds for k in spec['parents'] for fold in (range(3) if mode == 'cv' else [-1])]
    folders = {}
    with cf.ProcessPoolExecutor(max_workers=plan['limits']['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(train_job, t): t for t in tasks}
        done_count = 0
        while pending:
            if time.time() >= deadline:
                raise TimeoutError('Deadline reached; unfinished work remains partial')
            done, _ = cf.wait(pending, timeout=3, return_when=cf.FIRST_COMPLETED)
            processes = [psutil.Process()] + psutil.Process().children(recursive=True)
            rss, cpu = 0, 0
            for process in processes:
                try:
                    rss += process.memory_info().rss
                    cpu += sum(process.cpu_times()[:2])
                except psutil.Error:
                    pass
            with (ROOT / 'resources.jsonl').open('a') as handle:
                handle.write(json.dumps({'time': time.time(), 'mode': mode, 'rss_bytes': rss, 'cpu_seconds': cpu, 'processes': len(processes)}) + '\n')
            for future in done:
                task = pending.pop(future)
                folder = Path(future.result())
                folders[(task['key'], task['seed'], task['fold'])] = folder
                done_count += 1
                print(json.dumps({'phase': mode, 'complete': done_count, 'total': len(tasks), 'component': task['key'], 'seed': task['seed'], 'fold': task['fold']}), flush=True)
    return folders


def arrays_from(folders, keys, seeds, bank, mode):
    result = {}
    for key in keys:
        values = []
        for seed in seeds:
            if mode == 'cv':
                p = np.full(len(bank.y), np.nan)
                for fold in range(3):
                    folder = folders[(key, seed, fold)]
                    vi = np.flatnonzero(bank.folds == fold)
                    assert np.array_equal(np.load(folder / 'query_rows.npy'), bank.dev[vi])
                    p[vi] = np.load(folder / 'prediction.npy')
            else:
                folder = folders[(key, seed, -1)]
                expected = np.load(bank.root / 'confirmation_rows.npy') if mode == 'confirmation' else np.flatnonzero(bank.metadata.split.to_numpy() == 'test')
                assert np.array_equal(np.load(folder / 'query_rows.npy'), expected)
                p = np.load(folder / 'prediction.npy')
            assert np.isfinite(p).all()
            values.append(p)
        result[key] = np.mean(values, axis=0)
    return result


def bootstrap(y, p, ref):
    rng = np.random.default_rng(20261020)
    groups = [np.flatnonzero(y == c) for c in [0, 1]]
    deltas = []
    for _ in range(500):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        deltas.append(ks(y[ix], p[ix]) - ks(y[ix], ref[ix]))
    return np.quantile(deltas, [.025, .975]).tolist()


def main():
    plan = read(ROOT / 'plan.json')
    for name, checksum in read(ROOT / 'source_manifest.json').items():
        assert sha(ROOT / 'source' / name) == checksum
    statepath = ROOT / 'clock.json'
    if not statepath.exists():
        write(statepath, {'start': time.time(), 'deadline': time.time()+plan['limits']['max_seconds']})
    deadline = read(statepath)['deadline']
    if time.time() >= deadline:
        raise TimeoutError('Original second-pass deadline expired')
    bank = Bank(Path(plan['bank_root']))
    assert bank.manifest['bank_id'] == plan['bank_id']
    write(ROOT / 'environment.json', {'python': sys.version, 'host': os.uname().nodename, 'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'catboost', 'xgboost', 'joblib']}})
    original = {k: np.load(ROOT / 'evidence' / (k + '.npy')) for k in plan['recipes']}
    baseline = combine(plan['original'], original)
    scores = [{**spec, **describe(bank.y, combine(spec, original), bank.folds, baseline)} for spec in plan['candidates']]
    scores.sort(key=lambda s: s['ks'], reverse=True)
    write(ROOT / 'screening.json', scores)
    winner = next(s for s in scores if s['gate'])
    spec = {k: winner[k] for k in ['id', 'parents', 'method']}
    write(ROOT / 'screening_nominee.json', {'spec': spec, 'score': winner, 'plan_sha256': sha(ROOT / 'plan.json')})
    print(json.dumps({'phase': 'screening_complete', 'candidate_count': len(scores), 'nominee': winner}), flush=True)
    folders = run_jobs(plan, spec, plan['additional_seeds'], 'cv', deadline)
    all_arrays = [original] + [arrays_from(folders, spec['parents'], [seed], bank, 'cv') for seed in plan['additional_seeds']]
    bag_arrays = {k: np.mean([a[k] for a in all_arrays], axis=0) for k in spec['parents']}
    bagp = combine(spec, bag_arrays)
    bagscore = describe(bank.y, bagp, bank.folds, baseline)
    seeds = [2026] + plan['additional_seeds'] if bagscore['gate'] and bagscore['ks'] > winner['ks'] else [2026]
    chosen = bagp if len(seeds) > 1 else combine(spec, original)
    np.save(ROOT / 'candidate_oof.npy', chosen)
    write(ROOT / 'seed_verification.json', {'bagged': bagscore, 'individual_seeds': {str(seed): describe(bank.y, combine(spec, arrays), bank.folds, baseline) for seed, arrays in zip([2026]+plan['additional_seeds'], all_arrays)}})
    frozen = {'spec': spec, 'seeds': seeds, 'development': describe(bank.y, chosen, bank.folds, baseline), 'plan_sha256': sha(ROOT / 'plan.json'), 'caveat': plan['caveat']}
    write(ROOT / 'frozen_recipe.json', frozen)
    (ROOT / 'frozen_recipe.sha256').write_text(sha(ROOT / 'frozen_recipe.json')+'\n')
    print(json.dumps({'phase': 'frozen', **frozen}), flush=True)
    if frozen['development']['delta'] <= 1e-12:
        write(ROOT / 'result.json', {'state': 'completed', 'promoted': False, 'reason': 'No development improvement', 'frozen': frozen})
        return
    confirm_folders = run_jobs(plan, spec, seeds, 'confirmation', deadline)
    cp = combine(spec, arrays_from(confirm_folders, spec['parents'], seeds, bank, 'confirmation'))
    np.save(ROOT / 'confirmation_prediction.npy', cp)
    # Labels opened only after the sole nominated recipe and predictions are durable.
    cy = np.load(bank.root / 'confirmation_y.npy').astype(int)
    priorp = np.load(PRIOR / 'artifacts/confirmation/primary/combined.npy')
    reference = np.load(PRIOR / 'evidence/reference_confirmation.npy')
    confirm = {'ks': ks(cy, cp), 'prior_ks': ks(cy, priorp), 'v3_reference_ks': ks(cy, reference),
               'delta_vs_prior': ks(cy, cp)-ks(cy, priorp), 'paired_bootstrap_95ci_vs_prior': bootstrap(cy, cp, priorp),
               'caveat': plan['caveat'], 'recipe_sha256': sha(ROOT / 'frozen_recipe.json')}
    write(ROOT / 'confirmation.json', confirm)
    promoted = confirm['delta_vs_prior'] > 0 and frozen['development']['gate']
    print(json.dumps({'phase': 'confirmation_complete', 'promoted': promoted, **confirm}), flush=True)
    if not promoted:
        write(ROOT / 'result.json', {'state': 'completed', 'promoted': False, 'reason': 'Confirmation gate did not improve', 'frozen': frozen, 'confirmation': confirm})
        return
    assert sha(PRIOR / 'inputs/train_applications.csv') == bank.manifest['raw_sha256']['train_applications']
    full_folders = run_jobs(plan, spec, seeds, 'full', deadline)
    full = arrays_from(full_folders, spec['parents'], seeds, bank, 'full')
    p = combine(spec, full)
    query = np.flatnonzero(bank.metadata.split.to_numpy() == 'test')
    replay = {}
    for key in spec['parents']:
        values = []
        for seed in seeds:
            folder = full_folders[(key, seed, -1)]
            obj = joblib.load(folder / 'pipeline.joblib')
            obj['preprocessing'].bank = bank
            values.append(predict(obj['model'], obj['preprocessing'].transform(query[:256]), obj['weight']))
        replay[key] = np.mean(values, axis=0)
        assert np.allclose(replay[key], full[key][:256], atol=1e-12, rtol=0)
    assert np.allclose(combine(spec, replay), p[:256], atol=1e-12, rtol=0)
    sample = pd.read_csv(PRIOR / 'inputs/sample_submission.csv')
    values = pd.Series(p, index=bank.metadata.application_id.iloc[query].to_numpy())
    ordered = sample.application_id.map(values).to_numpy()
    df = pd.DataFrame({'application_id': sample.application_id, 'default_probability': ordered})
    assert len(df) == 30000 and df.application_id.is_unique and np.isfinite(ordered).all() and ((ordered >= 0) & (ordered <= 1)).all()
    df.to_csv(ROOT / 'submission.csv.tmp', index=False, float_format='%.12f')
    (ROOT / 'submission.csv.tmp').replace(ROOT / 'submission.csv')
    readback = pd.read_csv(ROOT / 'submission.csv')
    assert readback.application_id.tolist() == sample.application_id.tolist()
    assert np.allclose(readback.default_probability, ordered, atol=5e-13, rtol=0)
    write(ROOT / 'result.json', {'state': 'completed', 'promoted': True, 'frozen': frozen, 'confirmation': confirm,
                              'submission_sha256': sha(ROOT / 'submission.csv'), 'sample_order_exact': True,
                              'model_replay_rows': 256, 'full_fit_models': len(full_folders), 'target_ks_reached': frozen['development']['ks'] >= plan['target_ks'],
                              'models': {str(f.relative_to(ROOT)): sha(f / 'pipeline.joblib') for f in full_folders.values()}})


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        write(ROOT / 'failure.json', {'state': 'failed_or_partial', 'traceback': traceback.format_exc(), 'time': time.time()})
        raise
