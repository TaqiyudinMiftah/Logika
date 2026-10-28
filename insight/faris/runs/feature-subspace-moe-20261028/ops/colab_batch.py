"""Run the unchanged Kumo cohort on one admitted A100, bounded by CLOCK.json."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'source'))
from common import check_seal, dump, load, sha, utc

def main():
    clock = check_seal()
    assert load('COLAB_REPLAY_GATE.json')['status'] == 'PASS'
    assert load('COLAB_REPLAY_GATE.json')['original_clock_sha256'] == sha(R / 'CLOCK.json')
    tasks = [key for key, value in load('CONFIG.json')['tasks'].items() if value['family'] == 'kumo']
    assert len(tasks) == 80
    done, failed = [], []
    for task in tasks:
        folder = R / f'runs/{task}/attempt2'
        if (folder / 'RESULT.json').exists():
            result = json.loads((folder / 'RESULT.json').read_text())
            assert sha(folder / 'prediction.npz') == result['prediction_sha256']
            done.append(task)
            continue
        if folder.exists():
            failed.append({'task': task, 'reason': 'Existing incomplete attempt preserved; no automatic retry.'})
            continue
        if time.time() >= min(clock['admission_end_epoch'], clock['deadline_epoch'] - 7200):
            break
        packet = R / ('packets/fold' + str(load('CONFIG.json')['tasks'][task]['fold']))
        # Transfer overlaps fitting. An existing file is insufficient: its exact
        # sealed manifest and full feature/role hashes must be complete first.
        while True:
            ready = False
            if (packet / 'MANIFEST.json').exists() and (packet / 'roles.npz').exists():
                manifest = json.loads((packet / 'MANIFEST.json').read_text())
                ready = (sha(packet / 'MANIFEST.json') == load('GATE.json')['packet_manifests'][str(load('CONFIG.json')['tasks'][task]['fold'])]
                         and sha(packet / 'features.parquet') == manifest['feature_sha256']
                         and sha(packet / 'roles.npz') == manifest['roles_sha256'])
            if ready:
                break
            assert time.time() < min(clock['admission_end_epoch'], clock['deadline_epoch'] - 7200)
            dump({'stage': 'waiting_for_verified_packet', 'task': task, 'completed': len(done),
                  'expected': 80, 'UTC': utc()}, R / 'COLAB_LIVE_STATE.json')
            time.sleep(15)
        dump({'stage': 'modelling', 'UTC': utc(), 'completed': len(done), 'expected': 80,
              'failed': failed, 'active': task, 'original_clock_sha256': sha(R / 'CLOCK.json')}, R / 'COLAB_LIVE_STATE.json')
        log = R / f'colab_logs/{task}.log'
        log.parent.mkdir(exist_ok=True)
        command = [sys.executable, str(R / 'ops/colab_execution_adapter.py'), '--task', task, '--attempt', '2']
        limit = min(1800, int(clock['deadline_epoch'] - 7200 - time.time()))
        try:
            with log.open('ab') as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=limit, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'})
            if result.returncode:
                failed.append({'task': task, 'returncode': result.returncode, 'scientific_conclusion': 'none; execution failure'})
            else:
                record = json.loads((folder / 'RESULT.json').read_text())
                assert sha(folder / 'prediction.npz') == record['prediction_sha256']
                done.append(task)
        except subprocess.TimeoutExpired:
            # subprocess.run terminates its directly launched worker. No retry.
            failed.append({'task': task, 'reason': 'Execution time limit; no model-quality conclusion.'})
            dump({'status': 'TIME_LIMIT', 'task': task, 'UTC': utc()}, folder / 'EXECUTION_TIMEOUT.json')
        dump({'stage': 'modelling', 'UTC': utc(), 'completed': len(done), 'expected': 80,
              'failed': failed, 'active': None, 'original_clock_sha256': sha(R / 'CLOCK.json')}, R / 'COLAB_LIVE_STATE.json')
        if failed:
            # Prevent a resource problem from creating a large repeated-failure loop.
            break
    dump({'stage': 'complete' if len(done) == 80 else 'partial', 'UTC': utc(),
          'completed': len(done), 'expected': 80, 'completed_tasks': done, 'failed': failed,
          'pending': [task for task in tasks if task not in done],
          'original_clock_sha256': sha(R / 'CLOCK.json'),
          'fitted_weights_saved': False, 'KV_caches_saved': False}, R / 'COLAB_BATCH_RESULT.json')

if __name__ == '__main__':
    try:
        main()
    except BaseException:
        dump({'stage': 'failed', 'UTC': utc(), 'error': traceback.format_exc()}, R / 'COLAB_BATCH_FAILURE.json')
        raise
