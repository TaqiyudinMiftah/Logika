"""Verify and mirror raw-data research prerequisites before queue activation."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import base64
import hashlib
import json
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / 'v7'
sys.path.insert(0, str(ROOT.parent))
from node480_task import client, BASE


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(128 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.activation.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def main():
    build = json.loads((V / 'data_fe/reports/FEATURE_BUILD.json').read_text())
    preflight = json.loads((V / 'data_fe/reports/PREFLIGHT.json').read_text())
    local = json.loads((V / 'reports/LOCAL_CPU_PREFLIGHT.json').read_text())
    assert build['status'] == 'COMPLETE' and not build['target_used']
    assert build['feature_sha256'] == digest(V / 'data_fe/data/item_price_resilience.parquet')
    assert preflight['status'] == 'PASS' and preflight['source_sha256'] == digest(V / 'scripts/data_fe_worker.py')
    assert local['status'] == 'PASS' and local['all5fold_original_V25_validation_predictions_replayed']
    assert local['original_V25_test_mean_max_abs_error'] < 1e-12
    scripts = ['platform_runtime.py', 'conservative_adapters.py', 'modeling_features.py',
               'continuous_research.py', 'continue_frozen_kumo_queue.py', 'upload_local_score_run.py',
               'sync_completed_experiment.py', 'prepare_public_progress.py', 'master_log.py',
               'item_price_resilience_features.py', 'data_fe_worker.py', 'scan_item_price_signals.py',
               'preflight_local_cpu.py', 'mirror_cpu_training_cache.py', 'activate_data_research.py']
    for name in scripts:
        ast.parse((V / 'scripts' / name).read_text(encoding='utf-8'))
    structured = json.loads((V / 'structured_head_plan.json').read_text())
    relocated = []
    for spec in structured:
        if spec['id'] in ['V240', 'V241', 'V242']:
            assert not (V / 'runs' / spec['id'] / 'RESULT.json').exists()
            spec['execution_host'] = 'local_windows'
            relocated.append(spec['id'])
    write(V / 'structured_head_plan.json', structured)
    plans = json.loads((V / 'data_fe_plan.json').read_text())
    assert [s['id'] for s in plans] == [f'V{i}' for i in range(243, 251)]
    for spec in plans:
        spec['research_ready'] = True
        assert spec['execution_host'] == 'local_windows' and spec['Kumo_frozen_fraction'] == 1.0
    write(V / 'data_fe_plan.json', plans)
    write(V / 'reports/CPU_RUNTIME_RELOCATION.json', dict(
        UTC=datetime.now(timezone.utc).isoformat(), experiments=relocated,
        reason='Node480 memory cgroup full; replay-verified local CPU correction uses identical node480 cached Kumo/spline scores.',
        algorithm_or_hyperparameter_change=False, Kumo_frozen_fraction=1.0,
        preflight='v7/reports/LOCAL_CPU_PREFLIGHT.json', new_batch='V243-V250'))
    c = client()
    c.RETRIES = 2
    for folder in ['v7/data_fe', 'v7/data_fe/data', 'v7/data_fe/reports']:
        try:
            value = c.request('GET', 'api/contents/' + BASE + folder + '?content=0')
            assert value['type'] == 'directory'
        except RuntimeError as error:
            if '404' not in str(error):
                raise
            c.request('PUT', 'api/contents/' + BASE + folder, dict(type='directory'))
    files = [V / 'scripts' / name for name in scripts]
    files += [V / name for name in ['data_fe_plan.json', 'structured_head_plan.json',
              'DATA_SIGNAL_RESEARCH_PLAN.md', 'reports/LOCAL_CPU_PREFLIGHT.json',
              'reports/CPU_RUNTIME_RELOCATION.json']]
    files += sorted(p for p in (V / 'data_fe/reports').iterdir() if p.is_file())
    files += [V / 'data_fe/data/item_price_resilience.parquet']
    manifest = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        payload = path.read_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        c.request('PUT', 'api/contents/' + BASE + rel,
                  dict(type='file', format='base64', content=base64.b64encode(payload).decode()))
        actual = hashlib.sha256()
        url = c.base_url.rstrip('/') + '/files/' + quote(BASE + rel)
        with c.opener.open(url, timeout=40) as response:
            for block in iter(lambda: response.read(128 * 1024), b''):
                actual.update(block)
        assert actual.hexdigest() == sha, rel
        manifest.append(dict(path=rel, sha256=sha, bytes=len(payload)))
        print('MIRRORED_VERIFIED', rel, flush=True)
    # Registry is exposed only after every executable prerequisite was mirrored.
    registry = json.loads((V / 'CONTINUOUS_PLAN_FILES.json').read_text())
    if 'data_fe_plan.json' not in registry:
        registry.append('data_fe_plan.json')
    write(V / 'CONTINUOUS_PLAN_FILES.json', registry)
    c.request('PUT', 'api/contents/' + BASE + 'v7/CONTINUOUS_PLAN_FILES.json',
              dict(type='file', format='text', content=(V / 'CONTINUOUS_PLAN_FILES.json').read_text()))
    write(V / 'reports/DATA_RESEARCH_ACTIVATION.json', dict(
        status='PREREQUISITES_VERIFIED', UTC=datetime.now(timezone.utc).isoformat(),
        registered_batch='V243-V250', pending_before_activation=11,
        runtime_relocated=relocated, files=manifest))
    print('DATA_RESEARCH_READY_11_PENDING', flush=True)


if __name__ == '__main__':
    main()
