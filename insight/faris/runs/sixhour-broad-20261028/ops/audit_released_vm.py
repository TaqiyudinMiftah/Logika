"""Verify a released owned VM's retained evidence against canonical lab files.

This is an artifact/boundary audit, not final scientific selection or scoring.
No VM is contacted, allocated or stopped. No labels or predictions are printed.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import shlex
import subprocess

R = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(number):
    mirror = R / f'private/vm{number}/mirror'
    assert (R / f'private/vm{number}/RELEASED.json').exists()
    assert (mirror / f'GPU_VM{number}_QUEUE_FINAL.json').exists()
    config = json.loads((R / 'CONFIG.json').read_text())
    clock = json.loads((R / 'CLOCK.json').read_text())
    files = {}
    complete = failed = 0
    for path in sorted((mirror / 'runs').glob('*/attempt*/*')):
        if path.name not in ['START.json', 'RESULT.json', 'FAILURE.json', 'prediction.npz']:
            continue
        files[str(path.relative_to(mirror))] = dict(bytes=path.stat().st_size, sha256=sha(path))
        if path.name == 'RESULT.json':
            value = json.loads(path.read_text())
            start = json.loads((path.parent / 'START.json').read_text())
            assert value['status'] == 'complete'
            assert sha(path.parent / 'prediction.npz') == value['prediction_sha256']
            assert sha(path.parent / 'START.json') == value['start_sha256']
            assert (start['context_rows'], start['correction_rows'],
                    start['validation_rows'], start['query_rows']) == (50000, 6000, 14000, 20000)
            assert start['task']['seed'] in config['seeds']
            assert start['fold'] in range(config['folds'])
            assert start['task']['fold'] == start['fold']
            # The sealed worker derives the actual RNG seed from base + fold.
            assert start['seed'] == start['task']['seed'] + start['fold']
            assert start['task_id'] == path.parent.parent.name
            assert not start['outer_labels_loaded'] and not start['correction_labels_loaded']
            assert not start['query_labels_used_for_early_stopping']
            assert start['config_sha256'] == clock['config_sha256']
            assert start['contract_sha256'] == clock['contract_sha256']
            assert start['clock_sha256'] == sha(R / 'CLOCK.json')
            assert start['source_seal_sha256'] == clock['source_seal_sha256']
            assert start['source_sha256'] == sha(R / 'source/worker.py')
            assert not value.get('weights_saved', False) and not value.get('KV_cache_saved', False)
            complete += 1
        elif path.name == 'FAILURE.json':
            failed += 1
    assert complete and files
    code = """from pathlib import Path
import json,hashlib,sys
r=Path('/mnt/home/faris/logika-sixhour-broad-20261028');v=json.loads(sys.stdin.read())
for name,item in v['files'].items():
 p=r/name;assert p.stat().st_size==item['bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],name
flag=json.loads((r/('GPU_COLLECTION_VM'+str(v['number'])+'_COMPLETE.json')).read_text());assert flag['released']
print(json.dumps({'status':'PASS','release_flag':flag}))
"""
    result = subprocess.run(
        ['ssh', '-oBatchMode=yes', '-oConnectTimeout=20', 'faris@100.68.29.126',
         'python3 -c ' + shlex.quote(code)],
        input=json.dumps(dict(number=number, files=files)), text=True,
        capture_output=True, timeout=180, check=True)
    remote = json.loads(result.stdout)
    summary = dict(
        status='PASS', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        VM=number, completed_fit_attempts=complete, failed_attempts=failed,
        private_files_checked=len(files), local_and_canonical_hashes_equal=True,
        completed_run_context_and_query_rows_checked=True,
        seeds_and_folds_checked=True, effective_seed_rule='declared base seed + fold',
        source_config_contract_clock_checked=True, query_labels_not_loaded=True,
        fitted_weights_or_KV_saved=False, case_level_artifacts_private=True,
        scope='Released-VM artifact/boundary audit; not final scientific selection or independent confirmation',
        release_flag=remote['release_flag'], audit_source_sha256=sha(Path(__file__)))
    private = R / f'private/vm{number}/TERMINAL_COLLECTION_AUDIT.json'
    # Preserve an earlier audit rather than silently replacing its scope.
    if private.exists():
        previous = private.with_name('TERMINAL_COLLECTION_AUDIT.' + sha(private)[:16] + '.json')
        if not previous.exists():
            previous.write_bytes(private.read_bytes())
    private.write_text(json.dumps(dict(summary, manifest=files), indent=2) + '\n')
    summary['private_manifest_sha256'] = sha(private)
    (R / f'VM{number}_COLLECTION_AUDIT.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', type=int, required=True, choices=[4, 5, 6])
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.number)))
