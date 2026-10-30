"""Private compact evidence mirror on ThinkCentre42; never copy credentials."""
from pathlib import Path
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'source'))
from common import load, dump, sha, utc

SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-i',
       '/mnt/home/faris/.ssh/id_ed25519_logika_backup119', 'faris@100.98.169.42']
DEST = '/home/faris/logika-cross-checkpoints-20261030'


def main():
    clock = load('CLOCK.json')
    while time.time() < clock['deadline_epoch']:
        try:
            subprocess.run([*SSH, 'mkdir -p ' + DEST + ' && chmod 700 ' + DEST], check=True, timeout=20,
                           capture_output=True)
            subprocess.run(['rsync', '-az', '--exclude=private/***', '--exclude=packets/***',
                            '--exclude=logs/***', '--exclude=*.log', '--exclude=__pycache__/***',
                            '--exclude=*.tar', '--exclude=*.tar.gz', '-e', shlex.join(SSH[:-1]),
                            str(R) + '/', SSH[-1] + ':' + DEST + '/'], check=True, timeout=180,
                           capture_output=True)
            code = '''from pathlib import Path
import hashlib,json
r=Path(ROOT);checked=0
for path in r.glob('runs/*/attempt1/RESULT.json'):
 value=json.load(open(path))
 assert hashlib.sha256((path.parent/'START.json').read_bytes()).hexdigest()==value['start_sha256']
 assert hashlib.sha256((path.parent/'prediction.npz').read_bytes()).hexdigest()==value['prediction_sha256']
 checked+=1
print(json.dumps({'status':'PASS','complete_fit_records':checked}))
'''.replace('ROOT', repr(DEST))
            query = subprocess.run([*SSH, 'python3 -'], input=code, capture_output=True,
                                   text=True, check=True, timeout=90)
            value = json.loads(query.stdout)
            dump(dict(UTC=utc(), **value, destination=SSH[-1] + ':' + DEST,
                      credentials_excluded=True, no_weights_or_KV=True), R / 'CHECKPOINT42_STATE.json')
            if (R / 'COMPLETION.json').exists():
                return
        except Exception as exc:
            dump(dict(UTC=utc(), status='checkpoint_retry', exception=type(exc).__name__,
                      source_evidence_preserved=True), R / 'CHECKPOINT42_ERROR.json')
        time.sleep(60)


if __name__ == '__main__':
    main()

