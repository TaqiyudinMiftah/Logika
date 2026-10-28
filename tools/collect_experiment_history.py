"""Inventory experiment trees without reading environments, caches or large fitted models.

Runs on local/remote hosts using only Python's standard library. File inventories
are private working evidence until the public exporter sanitizes their contents.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

PRUNE = {'.git', '.venv', 'venv', 'env', 'env311', 'node_modules', '__pycache__',
         'site-packages', 'vendor', 'storage', '.cache', '.history-publish-20261028',
         'github-history-20261028', 'logika-history-export-20261028', '.uv-cache', '.uv', '.ruff_cache', '.pytest_cache'}
UNRELATED = ('dicefest', 'schematics', 'license-plate')
TEXT = {'.py', '.md', '.rst', '.txt', '.json', '.jsonl', '.yaml', '.yml', '.toml',
        '.ini', '.csv', '.tsv', '.log', '.sh', '.ipynb', '.html', '.svg', '.sql',
        '.sha256', '.service', '.patch', '.diff'}
IMAGES = {'.png', '.jpg', '.jpeg', '.webp'}
WEIGHTS = {'.pt', '.pth', '.ckpt', '.safetensors', '.joblib', '.pkl', '.cbm', '.onnx', '.bin'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def category(path):
    name = path.name.lower()
    if name.startswith('.env') or name in {'kaggle.json', 'credentials.json', 'token', 'access_token'} or name.endswith('.pem'):
        return 'credential_excluded'
    if path.suffix.lower() in WEIGHTS:
        return 'fitted_or_pretrained_weights_private'
    if path.suffix.lower() in {'.npz', '.npy', '.parquet', '.arrow', '.feather', '.db', '.sqlite', '.sqlite3'}:
        return 'data_or_case_level_evidence_private'
    if any(name.endswith(ext+'.gz') for ext in TEXT):
        return 'public_candidate_compressed_text'
    if path.suffix.lower() in TEXT or name in {'LICENSE', 'NOTICE', 'Dockerfile', 'Makefile'}:
        return 'public_candidate_text'
    if path.suffix.lower() in IMAGES:
        return 'public_candidate_figure'
    return 'binary_or_packaging_private'


def inventory(roots, base, resume=None):
    previous={}
    if resume and Path(resume).exists():
        opener=gzip.open if str(resume).endswith('.gz') else open
        with opener(resume,'rt')as stream:
            for line in stream:
                item=json.loads(line)
                if item.get('kind')=='file':previous[item['path']]=item
    seen = set()
    for root in roots:
        root = root.resolve()
        if not root.exists():
            print(json.dumps({'kind':'missing_root','path':str(root)}), flush=True)
            continue
        for folder, dirs, files in os.walk(root, followlinks=False):
            kept = []
            for name in sorted(dirs):
                p = Path(folder)/name
                if name in PRUNE or name.startswith(UNRELATED) or p.is_symlink():
                    print(json.dumps({'kind':'excluded_tree','path':str(p),'reason':'environment_dependency_cache_symlink_or_unrelated'}))
                else:
                    kept.append(name)
            dirs[:] = kept
            for name in sorted(files):
                p = Path(folder)/name
                if p in seen:
                    continue
                seen.add(p)
                if p.is_symlink() or not p.is_file():
                    continue
                try:
                    stat = p.stat()
                    kind = category(p)
                    # Reuse unchanged snapshot hashes and avoid reading private fitted
                    # artifacts merely to log their exclusion from the public archive.
                    cached=previous.get(str(p),{})
                    unchanged=cached.get('mtime_ns')==stat.st_mtime_ns and cached.get('bytes')==stat.st_size
                    sha=cached.get('sha256')if unchanged else None
                    if sha is None and kind.startswith('public_'):sha=digest(p)
                    rel = str(p.relative_to(base)) if p.is_relative_to(base) else str(p)
                    print(json.dumps({'kind':'file','path':str(p),'relative':rel,'category':kind,
                                      'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':sha}))
                except (OSError, PermissionError) as error:
                    print(json.dumps({'kind':'scan_error','path':str(p),'error':str(error)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('roots', nargs='+')
    args = parser.parse_args()
    inventory([Path(p) for p in args.roots], Path(args.base).resolve())
