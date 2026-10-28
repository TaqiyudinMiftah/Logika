"""Audit the completed matched CPU comparison without changing selection."""
from pathlib import Path
import json
import sys
import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'source'))
from common import check_seal, catalog, tasks, load, sha, dump, utc, ids_sha

check_seal()
designs = [key for key in catalog() if key.startswith('cpu_ordercontrast_')]
assert len(designs) == 4
evidence, starts = {}, {}
for key, task in tasks().items():
    if task['design'] not in designs:
        continue
    attempt = int(load('APPROVED_RETRIES.json').get(key, 1))
    folder = R / 'runs' / key / ('attempt' + str(attempt))
    result = json.loads((folder / 'RESULT.json').read_text())
    start = json.loads((folder / 'START.json').read_text())
    assert result['status'] == 'complete'
    assert sha(folder / 'prediction.npz') == result['prediction_sha256']
    assert sha(folder / 'START.json') == result['start_sha256']
    assert start['seed'] == task['seed'] + task['fold']
    assert start['context_rows'] == 50000 and start['query_rows'] == 20000
    assert not start['outer_labels_loaded'] and not start['correction_labels_loaded']
    assert not start['query_labels_used_for_early_stopping']
    assert result['audit']['inner_train_rows'] == 42500
    assert result['audit']['stopping_rows'] == 7500
    assert result['audit']['refit_context_rows'] == 50000
    assert not result['weights_saved'] and not result['KV_cache_saved']
    with np.load(folder / 'prediction.npz') as saved:
        assert len(saved['rows']) == len(np.unique(saved['rows'])) == 20000
        assert len(saved['raw']) == 20000 and np.isfinite(saved['raw']).all()
        row_sha = ids_sha(saved['rows'])
    evidence[key] = dict(result_sha256=sha(folder / 'RESULT.json'),
                         start_sha256=result['start_sha256'],
                         prediction_sha256=result['prediction_sha256'], rows_sha256=row_sha)
    starts[(task['design'], task['seed'], task['fold'])] = (start, result)
assert len(evidence) == 40
assessment = {}
for view in ['global99', 'actuarial99']:
    plain = 'cpu_ordercontrast_' + view + '_plain'
    ordered = 'cpu_ordercontrast_' + view + '_ordered'
    for seed in load('CONFIG.json')['seeds']:
        for fold in range(5):
            left, lr = starts[(plain, seed, fold)]
            right, rr = starts[(ordered, seed, fold)]
            for key in ['columns', 'context_rows_sha256', 'seed', 'fold', 'view']:
                assert left[key] == right[key]
            for key in ['inner_train_rows_sha256', 'stopping_rows_sha256']:
                assert lr['audit'][key] == rr['audit'][key]
            assert evidence[f'{plain}__s{seed}__f{fold}']['rows_sha256'] == evidence[f'{ordered}__s{seed}__f{fold}']['rows_sha256']
    p = load('evaluation/cohorts/' + plain + '__standalone/RESULT.json')
    o = load('evaluation/cohorts/' + ordered + '__standalone/RESULT.json')
    assessment[view] = dict(plain_KS=p['metrics']['KS'], ordered_KS=o['metrics']['KS'],
        ordered_minus_plain_KS=o['metrics']['KS'] - p['metrics']['KS'],
        seed_difference=[b-a for a,b in zip(p['seed_delta_KS'], o['seed_delta_KS'])],
        fold_difference=[b-a for a,b in zip(p['fold_delta_KS'], o['fold_delta_KS'])],
        plain_stack_KS=load('evaluation/cohorts/' + plain + '__stack/RESULT.json')['metrics']['KS'],
        ordered_stack_KS=load('evaluation/cohorts/' + ordered + '__stack/RESULT.json')['metrics']['KS'])
check_seal()
dump(dict(status='PASS', UTC=utc(), completed_fits=40, designs=4, evidence=evidence,
          matched_results=assessment, comparison='Explicit CPU Ordered versus Plain, identical feature views, context/inner/stop/query identities, base seeds and folds.',
          conclusion='Mixed effects across feature views; neither standalone nor stack exceeded campaign incumbent. No general Ordered advantage established.',
          reused_development=True, independent_audit=False,
          uncertainty='Descriptive paired effects only; no new independent uncertainty claim.',
          scientific_selection_unchanged=True), R / 'CPU_ORDERED_ASSESSMENT.json')
print(json.dumps(assessment))
