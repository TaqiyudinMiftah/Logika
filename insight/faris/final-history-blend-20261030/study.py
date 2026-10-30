"""Held-meta blending of every compatible complete historical pipeline.

No outer labels are opened by inventory, bank construction, dispatch or workers.
All learned blend parameters stay in memory; private predictions are retained.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import subprocess
import time
import traceback
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import ks_2samp

R = Path(__file__).resolve().parent
BASE = Path('/mnt/home/faris/logika-twelvehour-cross-20261030')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def config():
    return json.loads((R / 'PLAN.json').read_text())


def check_time():
    assert time.time() < config()['model_deadline_epoch'], 'Human final blend model cutoff reached'


def studies():
    home = Path('/mnt/home/faris')
    roots = []
    for name in os.listdir(home):
        if not name.startswith('logika-'):
            continue
        parent = home / name
        if not parent.is_dir() or parent == R:
            continue
        candidates = [parent]
        for child in os.listdir(parent):
            p = parent / child
            if p.is_dir() and (p / 'CONFIG.json').is_file():
                candidates.append(p)
        for p in candidates:
            if (p / 'CONFIG.json').is_file() and (p / 'runs').is_dir():
                roots.append(p)
    return sorted(set(roots), key=str)


def prepare():
    check_time()
    assert not (R / 'BANK_FROZEN.json').exists(), 'Prepared prediction bank is immutable'
    plan = config()
    assert plan['preregistered_before_blend_training']
    assert (R / 'PREPUBLICATION.json').is_file()
    roles = [np.load(BASE / f'packets/fold{f}/roles.npz') for f in range(5)]
    expected_rows = [np.r_[z['correction'], z['validation']] for z in roles]
    entries, excluded, roots_record, seen = [], [], [], {}
    for root in studies():
        check_time()
        try:
            raw_config = json.loads((root / 'CONFIG.json').read_text())
            designs = raw_config.get('designs', {})
            if not isinstance(designs, dict):
                excluded.append(dict(root=str(root), reason='legacy_configuration_schema_requires_separate_adapter'))
                continue
            complete_allowed = None
            board = root / 'SCOREBOARD.json'
            if (root / 'COMPLETION.json').exists() and board.exists():
                complete_allowed = set(json.loads(board.read_text()).get('complete_designs', []))
            compatible = True
            for f in range(5):
                other = np.load(root / f'packets/fold{f}/roles.npz')
                if not all(np.array_equal(other[k], roles[f][k]) for k in ('context', 'correction', 'validation')):
                    compatible = False
            if not compatible:
                excluded.append(dict(root=str(root), reason='different_row_or_training_boundaries'))
                continue
            root_count = 0
            for design, spec in designs.items():
                if complete_allowed is not None and design not in complete_allowed:
                    excluded.append(dict(root=str(root), design=design, reason='not_in_frozen_completed_designs'))
                    continue
                paths, hashes = [], []
                reason = None
                for seed in plan['base_seeds']:
                    for fold in range(5):
                        folder = root / 'runs' / f'{design}__s{seed}__f{fold}' / 'attempt1'
                        result_path = folder / 'RESULT.json'
                        if not result_path.is_file():
                            reason = 'incomplete_or_different_seed_or_attempt_schema'
                            break
                        value = json.loads(result_path.read_text())
                        start = json.loads((folder / 'START.json').read_text())
                        if value.get('status') != 'complete' or start.get('outer_labels_loaded') is not False:
                            reason = 'unqualified_status_or_outer_label_boundary'
                            break
                        if start.get('correction_labels_used_for_training') is not False:
                            reason = 'meta_labels_used_by_base_training'
                            break
                        if sha(folder / 'START.json') != value['start_sha256'] or sha(folder / 'prediction.npz') != value['prediction_sha256']:
                            raise AssertionError('Raw prediction or START hash mismatch: ' + str(folder))
                        raw = np.load(folder / 'prediction.npz')
                        if not (np.array_equal(raw['rows'], expected_rows[fold]) and raw['raw'].shape == (20000,) and np.isfinite(raw['raw']).all()):
                            reason = 'prediction_row_or_shape_incompatible'
                            break
                        paths.append(str(folder / 'prediction.npz'))
                        hashes.append(value['prediction_sha256'])
                    if reason:
                        break
                if reason:
                    excluded.append(dict(root=str(root), design=design, reason=reason))
                    continue
                identity = hashlib.sha256(json.dumps(hashes).encode()).hexdigest()
                if identity in seen:
                    excluded.append(dict(root=str(root), design=design, reason='identical_complete_prediction_packet', same_as=seen[identity]))
                    continue
                namespace = root.name
                key = str(root) + '::' + design
                seen[identity] = key
                family = namespace + '/' + str(spec.get('family', spec.get('model', spec.get('kind', 'unspecified'))))
                entries.append(dict(id=key, root=str(root), design=design, family=family,
                                    prediction_paths=paths, prediction_sha256=hashes,
                                    base_source_sha256=sha(root / 'source/worker.py') if (root / 'source/worker.py').exists() else None,
                                    config_sha256=sha(root / 'CONFIG.json')))
                root_count += 1
            roots_record.append(dict(root=str(root), registered_designs=len(designs), eligible_complete_designs=root_count))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            excluded.append(dict(root=str(root), reason='legacy_or_unavailable_packet_schema', exception_type=type(exc).__name__))
    assert entries, 'No compatible complete pipelines'
    families = sorted(set(e['family'] for e in entries))
    dump(R / 'PIPELINE_INVENTORY.json', dict(UTC=utc(), eligible=entries, excluded=excluded, studies=roots_record,
         all_accessible_canonical_registered_studies_enumerated=True, every_historical_location_verified=False,
         outer_labels_loaded=False, eligible_count=len(entries), family_count=len(families)))
    (R / 'bank').mkdir(exist_ok=True)
    bank_hashes = {}
    for fold in range(5):
        check_time()
        matrix = np.lib.format.open_memmap(R / f'bank/fold{fold}.npy', mode='w+', dtype=np.float32, shape=(20000, len(entries)))
        for j, entry in enumerate(entries):
            # Same-seed raw members are averaged before the deterministic blend.
            matrix[:, j] = np.mean([np.load(entry['prediction_paths'][s * 5 + fold])['raw'].astype(np.float64) for s in range(2)], axis=0)
        matrix.flush()
        groups = np.column_stack([np.mean(matrix[:, [j for j, e in enumerate(entries) if e['family'] == family]], axis=1, dtype=np.float64)
                                  for family in families]).astype(np.float32)
        np.save(R / f'bank/family{fold}.npy', groups)
        anchor = np.load(BASE / f'packets/fold{fold}/anchor.npz')
        np.savez_compressed(R / f'bank/meta{fold}.npz', rows=roles[fold]['validation'], y_meta=roles[fold]['y_correction'],
                            anchor=np.r_[anchor['correction_oof'], anchor['validation']])
        bank_hashes[str(fold)] = {name: sha(R / 'bank' / name) for name in (f'fold{fold}.npy', f'family{fold}.npy', f'meta{fold}.npz')}
        del matrix
    frozen = dict(UTC=utc(), inventory_sha256=sha(R / 'PIPELINE_INVENTORY.json'), bank_hashes=bank_hashes,
                  eligible_complete_pipelines=len(entries), families=families,
                  source_sha256=sha(__file__), outer_labels_loaded=False,
                  seeds=plan['base_seeds'], all_base_members_complete_two_seeds_five_folds=True,
                  training_policy='Fit new blend on6000heldmeta only;50000context raw bases;14000outer never opened')
    dump(R / 'BANK_FROZEN.json', frozen)
    print(json.dumps({k: frozen[k] for k in ('UTC', 'eligible_complete_pipelines', 'outer_labels_loaded')}), flush=True)


def objective(theta, x, offset, y, penalty):
    eta = offset + theta[0] + x @ theta[1:]
    error = expit(eta) - y
    coefficient_penalty = penalty * x.shape[1]
    value = np.mean(np.logaddexp(0., eta) - y * eta) + .5 * coefficient_penalty * np.dot(theta[1:], theta[1:])
    gradient = np.r_[np.mean(error), x.T @ error / len(y) + coefficient_penalty * theta[1:]]
    return float(value), gradient


def worker(fold, bank, nonnegative):
    check_time()
    frozen = json.loads((R / 'BANK_FROZEN.json').read_text())
    name = f'fold{fold}.npy' if bank == 'all' else f'family{fold}.npy'
    p = R / 'bank' / name
    assert sha(p) == frozen['bank_hashes'][str(fold)][name]
    meta_name = f'meta{fold}.npz'
    assert sha(R / 'bank' / meta_name) == frozen['bank_hashes'][str(fold)][meta_name]
    z = np.load(R / 'bank' / meta_name)
    raw = np.load(p, mmap_mode='r')
    anchor = np.asarray(z['anchor'], float)
    x = logit(np.clip(raw.astype(np.float64), 1e-6, 1 - 1e-6)) - logit(np.clip(anchor, 1e-6, 1 - 1e-6))[:, None]
    meta, outer = np.ascontiguousarray(x[:6000]), np.ascontiguousarray(x[6000:])
    offset = logit(np.clip(anchor, 1e-6, 1 - 1e-6))
    y = np.asarray(z['y_meta'], float)
    assert y.shape == (6000,) and len(z['rows']) == 14000
    for penalty in config()['penalties']:
        check_time()
        ident = f'{bank}_' + ('positive' if nonnegative else 'signed') + '_l' + str(penalty)
        output = R / 'results' / ident / f'fold{fold}'
        if (output / 'FROZEN.json').exists():
            continue
        started = time.time()
        bounds = [(-5., 5.)] + ([(0., 1.)] if nonnegative else [(-1., 1.)]) * meta.shape[1]
        fit = minimize(objective, np.zeros(meta.shape[1] + 1), args=(meta, offset[:6000], y, penalty), jac=True,
                       method='L-BFGS-B', bounds=bounds, options=dict(maxiter=150, ftol=1e-10, gtol=1e-6))
        assert np.isfinite(fit.x).all()
        prediction = expit(offset[6000:] + fit.x[0] + outer @ fit.x[1:])
        replay = np.concatenate([expit(offset[6000 + k:6000 + min(k + 997, 14000)] + fit.x[0] + outer[k:k + 997] @ fit.x[1:])
                                 for k in range(0, 14000, 997)])
        assert np.max(np.abs(prediction - replay)) <= 5e-6
        assert np.isfinite(prediction).all() and np.all((prediction >= 0) & (prediction <= 1))
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / 'prediction.npz', rows=z['rows'], prediction=prediction)
        dump(output / 'FROZEN.json', dict(UTC=utc(), fold=fold, recipe=ident, penalty=penalty, bank=bank,
             nonnegative=nonnegative, parameters=int(len(fit.x)), seconds=time.time() - started,
             optimizer_converged=bool(fit.success), iterations=int(fit.nit),
             optimizer_message=str(fit.message), eligible_for_scoring=bool(fit.success),
             probability_replay_max_abs=float(np.max(np.abs(prediction - replay))),
             base_bank_sha256=frozen['bank_hashes'][str(fold)][name], source_sha256=sha(__file__),
             prediction_sha256=sha(output / 'prediction.npz'), trained_rows=6000, outer_labels_loaded=False,
             fitted_coefficients_saved=False, seed_policy='Both originalbase seed packets averaged; deterministic meta optimization'))
        print(json.dumps(dict(fold=fold, recipe=ident, seconds=time.time() - started, converged=bool(fit.success))), flush=True)


def evaluate():
    plan = config()
    assert time.time() < plan['deadline_epoch'] - 30
    y = np.load(BASE / 'evaluation_inputs/target.npy').astype(int)
    reference = np.load(BASE / 'evaluation_inputs/previous.npz')['prediction']
    baseline = np.load(BASE / 'evaluation_inputs/baseline.npz')['prediction'] if (BASE / 'evaluation_inputs/baseline.npz').exists() else None
    score = lambda p: float(ks_2samp(p[y == 0], p[y == 1], method='asymp').statistic)
    records, incomplete = [], []
    for bank in ('all', 'family'):
        for positive in (False, True):
            for penalty in plan['penalties']:
                ident = f'{bank}_' + ('positive' if positive else 'signed') + '_l' + str(penalty)
                folders = [R / 'results' / ident / f'fold{f}' for f in range(5)]
                if not all((p / 'FROZEN.json').exists() for p in folders):
                    incomplete.append(ident)
                    continue
                receipts = [json.loads((p / 'FROZEN.json').read_text()) for p in folders]
                if not all(v['eligible_for_scoring'] for v in receipts):
                    incomplete.append(ident + ':optimizer_not_converged')
                    continue
                probability = np.full(70000, np.nan)
                coverage = np.zeros(70000, int)
                for folder, receipt in zip(folders, receipts):
                    assert sha(folder / 'prediction.npz') == receipt['prediction_sha256']
                    a = np.load(folder / 'prediction.npz')
                    probability[a['rows']] = a['prediction']
                    coverage[a['rows']] += 1
                assert np.all(coverage == 1) and np.isfinite(probability).all()
                out = R / 'evaluation' / ident
                out.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out / 'prediction.npz', rows=np.arange(70000), prediction=probability)
                dump(out / 'FROZEN.json', dict(prediction_sha256=sha(out / 'prediction.npz'), inputs=[sha(p/'FROZEN.json') for p in folders], source_sha256=sha(__file__), frozen_before_outer_score=True))
                ks = score(probability)
                folds = [np.load(R / f'bank/meta{f}.npz')['rows'] for f in range(5)]
                fold_delta = [float(ks_2samp(probability[rows][y[rows] == 0], probability[rows][y[rows] == 1], method='asymp').statistic - ks_2samp(reference[rows][y[rows] == 0], reference[rows][y[rows] == 1], method='asymp').statistic) for rows in folds]
                record = dict(recipe=ident, KS=ks, delta_current=ks - score(reference), fold_delta=fold_delta,
                              prediction_sha256=sha(out / 'prediction.npz'), independent_confirmation=False)
                dump(out / 'RESULT.json', record)
                records.append(record)
    records.sort(key=lambda d: (-d['KS'], d['recipe']))
    dump(R / 'FINAL_SELECTION.json', dict(UTC=utc(), best=records[0] if records else None, rule='largest complete converged fivefold blend KS;lexicaltie', conditional_interval_not_yet_opened=True))
    dump(R / 'RESULTS.json', dict(UTC=utc(), complete_recipes=len(records), results=records, incomplete=incomplete,
         reference_KS=score(reference), matched_baseline_KS=score(baseline) if baseline is not None else .5306361042400267,
         all70000_adaptively_reused=True, no_untouched_audit=True, gain_supported=False,
         gain_support_not_inferred_from_point_only=True, no_submission_made=True))
    print(json.dumps(dict(complete_recipes=len(records), best=records[0] if records else None)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['prepare', 'worker', 'evaluate'])
    parser.add_argument('--fold', type=int)
    parser.add_argument('--bank', choices=['all', 'family'])
    parser.add_argument('--positive', action='store_true')
    args = parser.parse_args()
    try:
        if args.operation == 'prepare':
            prepare()
        elif args.operation == 'worker':
            worker(args.fold, args.bank, args.positive)
        else:
            evaluate()
    except BaseException:
        folder = R / 'private'
        folder.mkdir(exist_ok=True)
        (folder / ('failure_' + str(os.getpid()) + '.log')).write_text(traceback.format_exc())
        raise
