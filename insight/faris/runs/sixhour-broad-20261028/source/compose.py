"""Incremental held-out-meta composition; this module never opens outer labels."""
from pathlib import Path
import fcntl
import json
import os
import time
import numpy as np
from common import R, load, dump, sha, ids_sha, utc, check_seal, catalog, tasks
from gating import calibrate, calibrated, fit_stack, stacked

FIXED_WEIGHTS = {'fixed_005': .05, 'fixed_010': .1, 'fixed_020': .2, 'fixed_040': .4}
WEIGHT_GRID = [0., .025, .05, .1, .15, .2, .3, .4, .6, 1.]
MODES = ['standalone', *FIXED_WEIGHTS, 'meta_adaptive', 'stack']


def design_catalog():
    return catalog()


def ks_numpy(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    assert len(y) == len(p) and set(np.unique(y)) == {0, 1} and np.isfinite(p).all()
    order = np.argsort(p, kind='stable')
    ends = np.flatnonzero(np.r_[p[order][1:] != p[order][:-1], True])
    difference = np.cumsum(y[order]) / y.sum() - np.cumsum(1 - y[order]) / (1 - y).sum()
    return float(np.max(np.abs(difference[ends])))


def _task_lookup(design, seed, fold):
    matches = [key for key, item in tasks().items()
               if item['design'] == design and item['seed'] == seed and item['fold'] == fold]
    assert len(matches) == 1, ('Task identity is ambiguous', design, seed, fold)
    return matches[0]


def completed_raw(task_id):
    root = R / 'runs' / task_id / 'attempt1'
    if not (root / 'RESULT.json').exists():
        return None
    result = json.loads((root / 'RESULT.json').read_text())
    assert result.get('status') == 'complete'
    assert sha(root / 'prediction.npz') == result['prediction_sha256']
    start = json.loads((root / 'START.json').read_text())
    assert sha(root / 'START.json') == result['start_sha256']
    assert result['weights_saved'] is False and result['KV_cache_saved'] is False
    expected = tasks()[task_id]
    assert start['task'] == expected, ('Run task differs from registered task', task_id)
    assert start['seed'] == expected['seed'] + expected['fold']
    assert start['source_sha256'] == sha(R / 'source/worker.py')
    for filename, key in [('SOURCE_SEAL.json', 'source_seal_sha256'),
                          ('CONFIG.json', 'config_sha256'),
                          ('VALIDATION_CONTRACT.json', 'contract_sha256'),
                          ('CLOCK.json', 'clock_sha256')]:
        assert start[key] == sha(R / filename)
    assert start['outer_labels_loaded'] is False
    assert start['correction_labels_used_for_training'] is False
    return root


def _save_arrays(path, **arrays):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def compose(design, seed, fold):
    assert design in catalog() and Path(design).name == design
    task_id = _task_lookup(design, seed, fold)
    raw_root = completed_raw(task_id)
    if raw_root is None:
        return None
    packet = R / f'packets/fold{fold}'
    roles = np.load(packet / 'roles.npz')
    anchor = np.load(packet / 'anchor.npz')
    correction, validation = roles['correction'], roles['validation']
    assert (len(roles['context']), len(correction), len(validation)) == (50000, 6000, 14000)
    assert len(np.unique(np.r_[roles['context'], correction, validation])) == 70000
    assert 'y_validation' not in roles.files and 'y_outer' not in roles.files
    raw = np.load(raw_root / 'prediction.npz')
    assert np.array_equal(raw['rows'], np.r_[correction, validation])
    assert raw['raw'].shape == (20000,) and np.isfinite(raw['raw']).all()
    inputs = {'raw_prediction_sha256': sha(raw_root / 'prediction.npz'),
              'raw_result_sha256': sha(raw_root / 'RESULT.json'),
              'roles_sha256': sha(packet / 'roles.npz'),
              'anchor_sha256': sha(packet / 'anchor.npz')}
    root = R / f'composed/{design}/s{seed}/fold{fold}'
    if (root / 'RESULT.json').exists():
        result = json.loads((root / 'RESULT.json').read_text())
        assert result['inputs'] == inputs
        assert sha(root / 'prediction.npz') == result['prediction_sha256']
        assert sha(root / 'FROZEN.json') == result['frozen_sha256']
        return root
    if (root / 'FROZEN.json').exists():
        frozen = json.loads((root / 'FROZEN.json').read_text())
        assert frozen['inputs'] == inputs and frozen['design_spec'] == catalog()[design]
        assert frozen['prediction_sha256'] == sha(root / 'prediction.npz')
        # Recover only the missing completion receipt. Do not refit or replace
        # a prediction identity that has already been frozen.
        dump(dict(status='complete', UTC=utc(), task_id=task_id, design=design,
                  seed=int(seed), fold=int(fold), inputs=inputs, modes=MODES,
                  selected_meta_weight=frozen['selected_meta_weight'],
                  prediction_sha256=frozen['prediction_sha256'],
                  frozen_sha256=sha(root / 'FROZEN.json'),
                  outer_labels_loaded=False, fitted_weights_saved=False,
                  recovery='completion receipt reconstructed from immutable freeze'), root / 'RESULT.json')
        return root
    if root.exists():
        # An interrupted pre-score compose attempt has no score identity yet.
        # Preserve rather than overwrite any already frozen attempt.
        failure = R / 'failures' / f'compose_{design}_s{seed}_f{fold}_{time.time_ns()}'
        failure.parent.mkdir(exist_ok=True)
        root.rename(failure)
    root.mkdir(parents=True, exist_ok=False)
    y = np.asarray(roles['y_correction'], int)
    assignment = np.asarray(anchor['inner_fold'], int)
    assert y.shape == assignment.shape == (6000,) and set(np.unique(assignment)) == {0, 1, 2}
    prediction = np.asarray(raw['raw'], float)
    inner = np.full(6000, np.nan)
    calibration_boundaries = []
    for block in range(3):
        train, heldout = np.flatnonzero(assignment != block), np.flatnonzero(assignment == block)
        assert len(train) == 4000 and len(heldout) == 2000 and not np.intersect1d(train, heldout).size
        theta = calibrate(prediction[train], y[train])
        inner[heldout] = calibrated(theta, prediction[heldout])
        calibration_boundaries.append(dict(block=block, training_rows=len(train), heldout_rows=len(heldout),
                                           training_global_rows_sha256=ids_sha(correction[train]),
                                           heldout_global_rows_sha256=ids_sha(correction[heldout])))
    assert np.isfinite(inner).all()
    full_theta = calibrate(prediction[:6000], y)
    candidate = calibrated(full_theta, prediction[6000:])
    anchor_meta, anchor_outer = np.asarray(anchor['correction_oof'], float), np.asarray(anchor['validation'], float)
    assert anchor_meta.shape == (6000,) and anchor_outer.shape == (14000,)
    assert np.isfinite(anchor_meta).all() and np.isfinite(anchor_outer).all()
    grid_scores = [ks_numpy(y, (1 - weight) * anchor_meta + weight * inner) for weight in WEIGHT_GRID]
    # Ascending grid plus first argmax implements the frozen smaller-weight tie rule.
    selected = WEIGHT_GRID[int(np.argmax(grid_scores))]
    output = {'standalone': candidate}
    output.update({name: (1 - weight) * anchor_outer + weight * candidate
                   for name, weight in FIXED_WEIGHTS.items()})
    output['meta_adaptive'] = (1 - selected) * anchor_outer + selected * candidate
    stack_theta, stack_audit = fit_stack(anchor_meta, inner, y, penalty=.03)
    output['stack'] = stacked(stack_theta, anchor_outer, candidate)
    for value in output.values():
        assert value.shape == (14000,) and np.isfinite(value).all() and np.all((value >= 0) & (value <= 1))
    _save_arrays(root / 'prediction.npz', rows=validation, **output)
    frozen = dict(UTC=utc(), design=design, design_spec=catalog()[design], seed=int(seed), fold=int(fold),
                  task_id=task_id, inputs=inputs, source_sha256=sha(__file__),
                  gating_source_sha256=sha(R / 'source/gating.py'),
                  contract_sha256=sha(R / 'VALIDATION_CONTRACT.json'),
                  config_sha256=sha(R / 'CONFIG.json'),
                  prediction_sha256=sha(root / 'prediction.npz'),
                  correction_rows_sha256=ids_sha(correction), validation_rows_sha256=ids_sha(validation),
                  calibration='three-fold monotone logistic OOF on6000 meta rows; full calibration fits6000 for outer inference',
                  calibration_penalty=.005, calibration_boundaries=calibration_boundaries,
                  fixed_weights=FIXED_WEIGHTS, meta_weight_grid=WEIGHT_GRID,
                  meta_weight_tie_rule='smaller candidate weight', selected_meta_weight=selected,
                  meta_weight_training_KS=grid_scores, stack=stack_audit,
                  outer_labels_opened=False, learned_coefficients_serialized=False,
                  independent_audit_available=False,
                  interpretation='Nested training boundaries on historically exposed development; not independent confirmation.')
    dump(frozen, root / 'FROZEN.json')
    dump(dict(status='complete', UTC=utc(), task_id=task_id, design=design, seed=int(seed), fold=int(fold),
              inputs=inputs, modes=MODES, selected_meta_weight=selected,
              prediction_sha256=sha(root / 'prediction.npz'), frozen_sha256=sha(root / 'FROZEN.json'),
              outer_labels_loaded=False, fitted_weights_saved=False), root / 'RESULT.json')
    return root


def freeze_design(design):
    """Freeze one complete two-seed/five-fold design before aggregate scoring."""
    config = load('CONFIG.json')
    results = {}
    for seed in config['seeds']:
        for fold in range(5):
            root = R / f'composed/{design}/s{seed}/fold{fold}'
            if not (root / 'RESULT.json').exists():
                return None
            result = json.loads((root / 'RESULT.json').read_text())
            assert result['prediction_sha256'] == sha(root / 'prediction.npz')
            assert result['frozen_sha256'] == sha(root / 'FROZEN.json')
            results[str((root / 'RESULT.json').relative_to(R))] = sha(root / 'RESULT.json')
    path = R / 'composed' / design / 'FROZEN.json'
    if path.exists():
        frozen = json.loads(path.read_text())
        assert frozen['composed_results'] == results and frozen['design_spec'] == catalog()[design]
    else:
        dump(dict(UTC=utc(), design=design, design_spec=catalog()[design],
                  seeds=config['seeds'], folds=5, modes=MODES, composed_results=results,
                  config_sha256=sha(R / 'CONFIG.json'),
                  contract_sha256=sha(R / 'VALIDATION_CONTRACT.json'),
                  source_sha256=sha(__file__), outer_scores_used_for_this_design_composition=False,
                  prior_exposure='All70k rows historically exposed; newly appended designs may adapt to previous development results.',
                  independent_audit_available=False), path)
    return path


def main():
    check_seal()
    assert load('GATE.json')['status'] == 'PASS', 'Composition requires a matching scientific gate'
    lock = (R / 'compose.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = load('CONFIG.json')
    complete, pending = [], []
    for design in catalog():
        for seed in config['seeds']:
            for fold in range(5):
                root = compose(design, seed, fold)
                (complete if root is not None else pending).append([design, seed, fold])
        freeze_design(design)
    dump(dict(UTC=utc(), complete=complete, pending=pending,
              no_outer_labels_loaded=True, historical_development_exposure=True), R / 'COMPOSITION_STATE.json')
    print(json.dumps(dict(stage='composition', complete=len(complete), pending=len(pending))), flush=True)


if __name__ == '__main__':
    main()
