"""Install an isolated pinned environment, prepare features, and run Kumo Large.

All new state remains under this project within ~/faris. Existing environments,
raw data, and historical submissions are read only. No Kaggle calls are made.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/taqiyudinmiftah/Logika/data/raw')
SDK = '329a707bbcf9ca147aa45a63fef8b2029f4ee37b'
PYTHON = ROOT / '.venv/bin/python'
STATE = ROOT / 'artifacts/node380'
STAGE = 'initializing'


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def status(state, **extra):
    write(STATE / 'status.json', dict(
        state=state, stage=STAGE, pid=os.getpid(),
        updated_utc=datetime.now(timezone.utc).isoformat(), **extra))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(stage, command):
    global STAGE
    STAGE = stage
    print('Starting', stage, flush=True)
    with (STATE / (stage + '.log')).open('ab') as log:
        child = subprocess.Popen([str(arg) for arg in command], cwd=ROOT,
                                 stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT)
        while child.poll() is None:
            status('running', child_pid=child.pid)
            time.sleep(5)
        if child.returncode:
            raise RuntimeError(f'{stage} exited {child.returncode}; see {stage}.log')
    print('Completed', stage, flush=True)


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    identity = dict(pid=os.getpid(),
                    proc_start_ticks=Path('/proc/self/stat').read_text().split()[21],
                    root=str(ROOT), created_utc=datetime.now(timezone.utc).isoformat())
    write(STATE / 'launcher.json', identity)
    os.environ.update(
        UV_CACHE_DIR=str(ROOT / '.uv-cache'),
        UV_PYTHON_INSTALL_DIR=str(ROOT / '.python'),
        HF_HOME=str(ROOT / 'hf_cache'),
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MPLBACKEND='Agg')
    uv = '/opt/conda/bin/uv'
    if not PYTHON.exists():
        run('environment', [uv, 'venv', '--python', '3.11.13', ROOT / '.venv'])
    run('torch_install', [uv, 'pip', 'install', '--python', PYTHON,
                          'torch==2.7.1', '--index-url',
                          'https://download.pytorch.org/whl/cu126'])
    run('dependencies', [uv, 'pip', 'install', '--python', PYTHON,
                         'numpy==1.24.1', 'pandas==1.5.3', 'pyarrow==22.0.0',
                         'scipy==1.10.1', 'scikit-learn==1.2.1', 'psutil==7.0.0',
                         'huggingface-hub==0.34.3', 'safetensors', 'matplotlib==3.7.5'])
    run('sdk_install', [uv, 'pip', 'install', '--python', PYTHON, '--no-deps',
                        'git+https://github.com/NVIDIA/structured-data-models@' + SDK])
    run('checkpoint', [PYTHON, '-u', '-c',
        "from huggingface_hub import hf_hub_download; "
        "[print(hf_hub_download('nvidia/Kumo-Tabular', name, revision='v1.0.1', token=False), flush=True) "
        "for name in ['config.json','large/classifier.pt']]"])
    os.environ['HF_HUB_OFFLINE'] = '1'
    run('environment_check', [PYTHON, '-u', ROOT / 'scripts/verify_environment.py'])
    run('features', [PYTHON, '-u', ROOT / 'scripts/prepare_kumo_features.py',
                     '--raw-dir', RAW, '--sample', RAW / 'sample_submission.csv'])
    common = ['--features', ROOT / 'data/kumo_v3/features.parquet',
              '--sample', RAW / 'sample_submission.csv']
    run('pilot', [PYTHON, '-u', ROOT / 'scripts/train_kumo_large.py',
                  '--pilot', *common, '--output', STATE / 'pilot'])
    run('full', [PYTHON, '-u', ROOT / 'scripts/train_kumo_large.py',
                 *common, '--output', STATE / 'full'])
    source = STATE / 'full/submission_kumo_large_logika_v3.csv'
    held = ROOT / 'submissions_held'
    held.mkdir(exist_ok=True)
    destination = held / source.name
    shutil.copyfile(source, destination)
    if digest(source) != digest(destination):
        raise RuntimeError('Held submission checksum differs')
    status('completed', submission=str(destination), submission_sha256=digest(destination))
    print('Completed:', destination, flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if STATE.is_dir():
            status('failed', error=str(error), traceback=traceback.format_exc())
        raise
