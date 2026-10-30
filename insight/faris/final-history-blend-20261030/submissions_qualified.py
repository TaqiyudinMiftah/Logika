"""Inventory historical test CSVs and produce explicitly unscored test blends.

No test label exists in this workflow. Disagreement is a diagnostic, never KS.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def discover(roots):
    visited = set()
    stack = [Path(r) for r in roots]
    skip = {'.git', '.cache', '__pycache__', 'site-packages', 'node_modules', '.venv', 'env', 'env311', 'env_gpu',
            'venv', 'models', 'feature_build', 'raw', 'packets', 'data', 'storage'}
    while stack:
        folder = stack.pop()
        try:
            stat = folder.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in visited:
                continue
            visited.add(identity)
            for item in folder.iterdir():
                if item.is_dir():
                    if item.name not in skip and not item.name.startswith('env'):
                        stack.append(item)
                elif item.suffix.lower() == '.csv' and 'submission' in item.name.lower() and 'sample' not in item.name.lower():
                    yield item
        except (PermissionError, FileNotFoundError, OSError):
            continue


def inventory(roots, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    entries, excluded, values, seen, master = [], [], [], {}, None
    for path in discover(roots):
        try:
            with path.open(newline='') as stream:
                reader = csv.DictReader(stream)
                names = reader.fieldnames or []
                key = next((k for k in ('application_id', 'id') if k in names), None)
                probability = next((k for k in ('default_probability', 'prediction', 'probability', 'target') if k in names and k != key), None)
                if len(names) != 2 or key is None or probability is None:
                    excluded.append(dict(path=str(path), reason='not_two_column_submission_schema'))
                    continue
                rows = list(reader)
            if len(rows) != 30000:
                excluded.append(dict(path=str(path), reason='not30000testrows', rows=len(rows)))
                continue
            ids = np.array([row[key] for row in rows])
            p = np.array([float(row[probability]) for row in rows], dtype=np.float64)
            assert len(set(ids)) == 30000 and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
            order = np.argsort(ids)
            ids, p = ids[order], p[order]
            if master is None:
                master = ids
            if not np.array_equal(ids, master):
                excluded.append(dict(path=str(path), reason='different_test_ID_set'))
                continue
            digest = hashlib.sha256(p.tobytes()).hexdigest()
            record = dict(path=str(path), csv_sha256=sha(path), vector_sha256=digest,
                          rows=30000, mean=float(p.mean()), std=float(p.std()),
                          quantiles=np.quantile(p, [0., .01, .1, .5, .9, .99, 1.]).tolist(),
                          local_KS_not_available_from_test_CSV=True)
            if digest in seen:
                record['duplicate_vector_of'] = seen[digest]
            else:
                seen[digest] = len(values)
                record['column'] = len(values)
                values.append(p)
            entries.append(record)
        except (OSError, AssertionError, ValueError, KeyError) as exc:
            excluded.append(dict(path=str(path), reason='invalid_or_unreadable_submission', exception_type=type(exc).__name__))
    report = dict(epoch=time.time(), roots=[str(r) for r in roots], accessible_named_submission_CSVs=entries,
                  excluded=excluded, unique_vectors=len(values), hidden_test_KS_unknown=True,
                  coverage='Accessible named submission CSVs; denied/unavailable roots and files only in compressedarchives not claimed covered')
    (output / 'SUBMISSION_INVENTORY.json').write_text(json.dumps(report, indent=2) + '\n')
    assert values, 'No valid test CSVs discovered'
    np.savez_compressed(output / 'private_test_bank.npz', ids=master, probabilities=np.column_stack(values))
    print(json.dumps(dict(valid_files=len(entries), unique_vectors=len(values), excluded=len(excluded))), flush=True)


def compare(inputs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ids, vectors, sources, seen = None, [], [], set()
    exclusions = []
    for directory in inputs:
        directory = Path(directory)
        z = np.load(directory / 'private_test_bank.npz')
        if ids is None:
            ids = z['ids']
        assert np.array_equal(ids, z['ids'])
        data = json.loads((directory / 'SUBMISSION_INVENTORY.json').read_text())
        for record in data['accessible_named_submission_CSVs']:
            if 'column' not in record or record['vector_sha256'] in seen:
                continue
            if record['std'] < 1e-10 or 'template' in Path(record['path']).name.lower():
                exclusions.append(dict(path=record['path'], reason='constant_or_named_template_not_a_model_submission', vector_sha256=record['vector_sha256']))
                continue
            seen.add(record['vector_sha256'])
            vectors.append(z['probabilities'][:, record['column']])
            sources.append(record)
    matrix = np.column_stack(vectors)
    assert matrix.shape == (30000, len(sources))
    ranks = np.column_stack([rankdata(p, method='average') for p in vectors]) / 30000
    candidates = dict(uniform_probability=np.mean(matrix, axis=1), uniform_logit=expit(np.mean(logit(np.clip(matrix, 1e-6, 1 - 1e-6)), axis=1)), uniform_rank=np.mean(ranks, axis=1))
    output_files = {}
    for name, prediction in candidates.items():
        path = output / ('submission_UNSCORED_all_' + name + '.csv')
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['application_id', 'default_probability'])
            writer.writerows(zip(ids, prediction))
        output_files[name] = dict(sha256=sha(path), path=str(path), local_KS=None, public_KS=None, submitted=False)
    normalized = ranks - ranks.mean(axis=0)
    norm = np.sqrt(np.sum(normalized**2, axis=0))
    normalized /= np.maximum(norm, 1e-15)
    correlation = normalized.T @ normalized
    top = np.argsort(matrix, axis=0)[-1500:]
    pairs = []
    for a in range(len(sources)):
        for b in range(a):
            pairs.append((float(correlation[a, b]), a, b))
    pairs.sort()
    differences = []
    for c, a, b in pairs[:30]:
        intersection = len(np.intersect1d(top[:, a], top[:, b]))
        differences.append(dict(source_a=sources[a]['path'], source_b=sources[b]['path'], spearman=c,
                                top5percent_overlap_fraction=intersection / 1500,
                                mean_abs_probability_difference=float(np.mean(abs(matrix[:, a] - matrix[:, b])))))
    report = dict(epoch=time.time(), unique_test_vectors=len(sources), sources=sources,
                  excluded_nonmodel_templates=exclusions, correlations=dict(min=float(np.min(correlation)), max=float(np.max(correlation)), pairs=len(pairs)),
                  most_disagreeing_pairs=differences, unscored_blends=output_files,
                  hidden_test_KS_unknown=True, predictive_complementarity_not_established_by_disagreement=True,
                  no_Kaggle_upload=True, no_hidden_labels_or_leaderboard_queries=True)
    (output / 'TEST_DISAGREEMENT_REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(unique_test_vectors=len(sources), pairs=len(pairs), unscored_blends=len(output_files))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['inventory', 'compare'])
    parser.add_argument('--roots', nargs='+')
    parser.add_argument('--inputs', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    inventory(args.roots, args.output) if args.operation == 'inventory' else compare(args.inputs, args.output)
