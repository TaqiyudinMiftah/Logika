"""Immutable identities, atomic evidence and the official pooled binary KS."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    temporary.replace(path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load(name):
    return json.loads((R / name).read_text())


def ids_sha(indices):
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def metrics(y, p):
    from sklearn.metrics import roc_curve, roc_auc_score, log_loss
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    assert len(y) == len(p) and np.isfinite(p).all()
    fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
    return dict(KS=float(np.max(np.abs(tpr - fpr))), AUC=float(roc_auc_score(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])), rows=len(y))


def source_seal():
    return {str(p.relative_to(R)): sha(p) for p in sorted((R / 'source').rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py', '.txt'}}


def check_seal():
    expected = load('SOURCE_SEAL.json')
    assert source_seal() == expected, 'Campaign source changed after sealing'
    clock = load('CLOCK.json')
    assert sha(R / 'SOURCE_SEAL.json') == clock['source_seal_sha256']
    assert sha(R / 'CONFIG.json') == clock['config_sha256']
    assert sha(R / 'VALIDATION_CONTRACT.json') == clock['contract_sha256']
    for relative, expected in load('VALIDATION_CONTRACT.json')['immutable_input_metadata'].items():
        assert sha(R / relative) == expected, relative
    return clock
