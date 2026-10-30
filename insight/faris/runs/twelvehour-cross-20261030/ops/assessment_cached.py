"""Cache byte-immutable catalog metadata; execute original sealed scientific code."""
from pathlib import Path
import argparse
import functools
import hashlib
import json
import runpy
import sys

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'source'))
import common

parser = argparse.ArgumentParser()
parser.add_argument('--module', choices=['compose', 'evaluate', 'audit'], required=True)
args, extra = parser.parse_known_args()
paths = [R / name for name in ['CONFIG.json', 'EXTRA_DESIGNS.jsonl', 'CLOCK.json',
                              'SOURCE_SEAL.json', 'VALIDATION_CONTRACT.json']]
identities = {str(p): common.sha(p) for p in paths}
common.check_seal()
original_load, original_catalog, original_tasks = common.load, common.catalog, common.tasks
config = original_load('CONFIG.json')


def cached_load(name):
    return config if name == 'CONFIG.json' else original_load(name)


common.load = cached_load
common.catalog = functools.lru_cache(maxsize=1)(original_catalog)
common.tasks = functools.lru_cache(maxsize=1)(original_tasks)
# Complete structural equality, including iteration order and exact seed/fold IDs.
expected = {f'{key}__s{seed}__f{fold}': dict(design=key, seed=seed, fold=fold)
            for key in original_catalog() for seed in config['seeds'] for fold in range(5)}
assert common.tasks() == expected and list(common.tasks()) == list(expected)
assert common.catalog() == original_catalog()
print(json.dumps(dict(status='immutable_catalog_cache_validated', module=args.module,
                      tasks=len(expected), wrapper_sha256=common.sha(__file__),
                      original_scientific_source_unchanged=True)), flush=True)
sys.argv = [str(R / 'source' / (args.module + '.py')), *extra]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
finally:
    assert all(common.sha(Path(p)) == digest for p, digest in identities.items()), 'Immutable metadata changed during assessment'
