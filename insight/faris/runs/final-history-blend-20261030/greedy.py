"""Caruana-inspired held-meta ensemble selection; no outer targets opened."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import time
import traceback
import numpy as np

R = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    p.parent.mkdir(exist_ok=True, parents=True)
    q = p.with_name(p.name + '.tmp')
    q.write_text(json.dumps(value, indent=2) + '\n')
    q.replace(p)


def scores(y, values, ks):
    if not ks:
        clipped = np.clip(values, 1e-8, 1 - 1e-8)
        return -np.mean(y[:, None] * np.log(clipped) + (1 - y[:, None]) * np.log1p(-clipped), axis=0)
    order = np.argsort(values, axis=0, kind='stable')
    yy = y[order]
    sorted_values = np.take_along_axis(values, order, axis=0)
    ends = np.r_[sorted_values[1:] != sorted_values[:-1], np.ones((1, values.shape[1]), dtype=bool)]
    difference = np.cumsum(yy, axis=0) / y.sum() - np.cumsum(1 - yy, axis=0) / (len(y) - y.sum())
    return np.max(np.where(ends, abs(difference), 0.), axis=0)


def worker(fold, bank, ks):
    plan = json.loads((R / 'GREEDY_PLAN.json').read_text())
    assert (R / 'GREEDY_PREPUBLICATION.json').exists()
    freeze = json.loads((R / 'BANK_FROZEN.json').read_text())
    filename = f'fold{fold}.npy' if bank == 'all' else f'family{fold}.npy'
    assert sha(R / 'bank' / filename) == freeze['bank_hashes'][str(fold)][filename]
    raw = np.load(R / 'bank' / filename, mmap_mode='r')
    z = np.load(R / f'bank/meta{fold}.npz')
    y = z['y_meta'].astype(int)
    anchor = z['anchor'].astype(np.float64)
    check = scores(np.array([0, 0, 1, 1]), np.array([[0., .5], [.1, .5], [.9, .5], [1., .5]]), True)
    assert np.allclose(check, [1., 0.], atol=1e-12)
    individual = scores(y, raw[:6000], ks)
    order = sorted(range(len(individual)), key=lambda j: (-individual[j], j))[:64]
    matrix = np.column_stack([anchor, raw[:, order]]).astype(np.float64)
    for strength in (1, 4, 16):
        cumulative = strength * anchor
        selected = []
        for step in range(1, 21):
            assert time.time() < plan['model_deadline_epoch'] - 15
            candidates = (cumulative[:6000, None] + matrix[:6000]) / (strength + step)
            chosen = int(np.argmax(scores(y, candidates, ks)))
            cumulative += matrix[:, chosen]
            selected.append(chosen)
            if step not in (1, 5, 20):
                continue
            name = f'greedy_{bank}_' + ('KS' if ks else 'logloss') + f'_anchor{strength}_steps{step}'
            out = R / 'greedy_results' / name / f'fold{fold}'
            assert not out.exists(), 'No implicit attempt retry'
            out.mkdir(parents=True)
            prediction = cumulative[6000:] / (strength + step)
            assert np.isfinite(prediction).all()
            np.savez_compressed(out / 'prediction.npz', rows=z['rows'], prediction=prediction)
            dump(out / 'FROZEN.json', dict(recipe=name, fold=fold, source_sha256=sha(__file__),
                 prediction_sha256=sha(out / 'prediction.npz'), bank_sha256=freeze['bank_hashes'][str(fold)][filename],
                 meta_only_selected_candidate_indices=order, selection_path_indices=selected,
                 meta_rows=6000, selected_all1925_candidates_considered=(bank == 'all'), shortlist=64,
                 outer_labels_loaded=False, learned_tensor_or_model_weights_saved=False,
                 base_seed_policy='Both complete base seeds averaged before deterministic meta selection'))
        print(json.dumps(dict(fold=fold, bank=bank, metric='KS' if ks else 'logloss', anchor=strength, status='20step_selection_completed')), flush=True)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--fold', type=int, required=True)
    a.add_argument('--bank', choices=('all', 'family'), required=True)
    a.add_argument('--ks', action='store_true')
    args = a.parse_args()
    try:
        worker(args.fold, args.bank, args.ks)
    except BaseException:
        p = R / 'private'
        p.mkdir(exist_ok=True)
        (p / f'GREEDY_FAILURE_{args.fold}_{args.bank}_{args.ks}.log').write_text(traceback.format_exc())
        raise
