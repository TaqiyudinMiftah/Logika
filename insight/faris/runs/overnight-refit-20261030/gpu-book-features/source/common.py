"""Immutable identities and append-only adaptive-development experiment catalog."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import numpy as np

R = Path(__file__).resolve().parents[1]

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def ids_sha(rows):
    return hashlib.sha256(np.asarray(rows, dtype='<i8').tobytes()).hexdigest()

def dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    temporary.replace(path)

def load(name):
    return json.loads((R / name).read_text())

def source_seal():
    return {str(p.relative_to(R)): sha(p) for p in sorted((R / 'source').rglob('*.py'))
            if '__pycache__' not in p.parts}

def check_seal():
    clock = load('CLOCK.json')
    assert sha(R / 'CONFIG.json') == clock['config_sha256']
    assert sha(R / 'VALIDATION_CONTRACT.json') == clock['contract_sha256']
    assert sha(R / 'SOURCE_SEAL.json') == clock['source_seal_sha256']
    assert source_seal() == load('SOURCE_SEAL.json'), 'Sealed scientific source changed'
    for relative, expected in load('VALIDATION_CONTRACT.json')['immutable_input_metadata'].items():
        assert sha(R / relative) == expected, relative
    return clock

def catalog():
    result = dict(load('CONFIG.json')['designs'])
    ledger = R / 'EXTRA_DESIGNS.jsonl'
    if ledger.exists():
        previous = None
        for line in ledger.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            assert item['id'] not in result, 'Never overwrite a registered design'
            signature = hashlib.sha256(json.dumps(item['design'], sort_keys=True).encode()).hexdigest()
            assert item['signature'] == signature
            assert item.get('previous_record_sha256') == previous
            assert item['source_reference'] and 'exposure_references' in item
            previous = hashlib.sha256(line.encode()).hexdigest()
            result[item['id']] = item['design']
    return result

def tasks():
    return {f'{key}__s{seed}__f{fold}': {'design': key, 'seed': seed, 'fold': fold}
            for key in catalog() for seed in load('CONFIG.json')['seeds'] for fold in range(5)}

def metrics(y, p):
    from sklearn.metrics import roc_curve, roc_auc_score, log_loss
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    assert y.shape == p.shape and np.isfinite(p).all()
    fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
    return {'KS': float(np.max(np.abs(tpr - fpr))), 'AUC': float(roc_auc_score(y, p)),
            'logloss': float(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9), labels=[0, 1])), 'rows': len(y)}
