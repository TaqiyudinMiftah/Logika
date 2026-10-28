"""Execution-only, idempotent task launch with an original-clock timeout."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'source'))
from common import check_seal, dump, utc

parser = argparse.ArgumentParser()
parser.add_argument('--task', required=True)
parser.add_argument('--attempt', type=int, default=1)
parser.add_argument('--pilot', action='store_true')
a = parser.parse_args()
clock = check_seal()
root = R / f'dispatch/{a.task}/attempt{a.attempt}'
root.mkdir(parents=True, exist_ok=False)
limit = min(900, int(clock['model_deadline_epoch'] - time.time() - 10))
assert limit > 0 and time.time() < clock['admission_end_epoch']
dump({'pid': os.getpid(), 'task': a.task, 'attempt': a.attempt, 'UTC': utc(),
      'started_epoch': time.time(), 'limit_seconds': limit}, root / 'START.json')
command = [sys.executable, str(R / 'source/worker.py'), '--task', a.task, '--attempt', str(a.attempt)]
if a.pilot:
    command.append('--pilot')
(R / 'logs').mkdir(exist_ok=True)
with (R / f'logs/{a.task}_a{a.attempt}.log').open('ab') as log:
    try:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=limit)
        status = {'returncode': process.returncode, 'execution_timeout': False}
    except subprocess.TimeoutExpired:
        status = {'returncode': 124, 'execution_timeout': True,
                  'scientific_conclusion': 'none; bounded execution limit'}
dump(dict(status, UTC=utc(), task=a.task, attempt=a.attempt), root / 'STATUS.json')
