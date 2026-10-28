"""Only complete fivefold runs enter the pooled comparison."""
from pathlib import Path
import datetime, hashlib, json, sys
import numpy as np
sys.path.insert(0, '/mnt/home/faris/logika-faris-exploration/transfer-20261025/source')
import baseline as b

R = Path(__file__).resolve().parents[1]
ref = np.load('/mnt/home/faris/logika-faris-exploration/transfer-20261025/ROBUSTNESS_OOF.npz')
y, folds = ref['target'], ref['fold']
baseline = ref['maturity__original']
results, incomplete = [], []
out = R/'reports'
out.mkdir(exist_ok=True)
for d in sorted((R/'storage/runs').iterdir()):
    paths = [d/f'config0/fold{f}' for f in range(5)]
    n = sum((p/'RESULT.json').exists() for p in paths)
    if n != 5:
        incomplete.append({'model': d.name, 'completed_folds': n})
        continue
    raw, corrected = np.full(len(y), np.nan), np.full(len(y), np.nan)
    evidence, runtime = [], 0.
    key = None
    for f, p in enumerate(paths):
        result = json.loads((p/'RESULT.json').read_text())
        assert result['status'] == 'fold_complete' and result['fold'] == f
        key = result['key']; runtime += result['seconds']
        z = np.load(p/'predictions.npz')
        rows = z['rows'][6000:]
        assert np.array_equal(rows, np.flatnonzero(folds == f))
        raw[rows], corrected[rows] = z['raw'][6000:], z['corrected']
        evidence.append({'fold': f, 'prediction_sha256': b.sha(p/'predictions.npz')})
    assert np.isfinite(raw).all() and np.isfinite(corrected).all()
    score = {'model': d.name, 'key': key, 'variant': 'default', 'raw': b.scores(y, raw),
             'corrected': b.scores(y, corrected), 'seconds': runtime, 'evidence': evidence}
    score['corrected_delta_vs_maturity'] = score['corrected']['KS'] - b.scores(y, baseline)['KS']
    score['fold_corrected_KS'] = [b.scores(y[folds==f], corrected[folds==f])['KS'] for f in range(5)]
    np.savez_compressed(out/f'{d.name}_OOF.npz', target=y, fold=folds, raw=raw, corrected=corrected)
    results.append(score)
results.sort(key=lambda x: x['corrected']['KS'], reverse=True)
payload = {'as_of_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'complete_models': len(results),
           'baseline_maturity': b.scores(y, baseline), 'results': results, 'incomplete': incomplete,
           'limits': ['Reused development labels; no independent confirmation.', 'Default configurations on selected 90-feature representation.',
                      'Historical runs before retention update may contain saved estimators. New runs retain recipes and predictions only.']}
b.dump(payload, out/'COMPARISON.json')
lines = ['# Model comparison', '', 'Original five folds; 70,000 pooled out-of-fold predictions. Defaults only. No independent confirmation.', '',
         '| Model | Raw KS | Corrected KS | Change vs maturity |', '|---|---:|---:|---:|']
for r in results:
    lines.append(f"| {r['model']} | {r['raw']['KS']:.6f} | {r['corrected']['KS']:.6f} | {r['corrected_delta_vs_maturity']:+.6f} |")
(out/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
matrix = json.loads((R/'MODEL_MATRIX.json').read_text())
by_key = {r['key']: r for r in results}
for entry in matrix['entries']:
    if entry['variant'] == 'default' and entry.get('registry_key') in by_key:
        r = by_key[entry['registry_key']]
        entry.update(status='fivefold_complete', local_KS=r['corrected']['KS'], raw_KS=r['raw']['KS'], reason='Defaults; original fivefold pooled KS; reused development data.')
b.dump(matrix, R/'MODEL_MATRIX.json')
print(json.dumps({'complete_models': len(results), 'scores': [{k:v for k,v in r.items() if k in ['model','raw','corrected','corrected_delta_vs_maturity']} for r in results]}, indent=2))
