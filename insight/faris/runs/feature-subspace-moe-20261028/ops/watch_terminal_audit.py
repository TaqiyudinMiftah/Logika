"""Wait for terminal predictions and audit them within the same original clock."""
from pathlib import Path
import fcntl
import hashlib
import json
import subprocess
import sys
import time
import traceback

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'source'))
from common import check_seal, dump, load, sha, utc

def main():
    lock = (R / 'artifact_audit.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    clock = check_seal()
    while time.time() < clock['deadline_epoch'] - 1200:
        state = load('LIVE_STATE.json')
        if state['stage'] == 'failed':
            dump({'status': 'SKIPPED_CORE_FAILURE', 'UTC': utc(), 'scientific_conclusion': 'none'},
                 R / 'ARTIFACT_CONSISTENCY_AUDIT.json')
            return
        if state['stage'] in ['complete', 'refit_required']:
            if (R / 'ARTIFACT_CONSISTENCY_AUDIT.json').exists():
                return
            limit = min(1800, int(clock['deadline_epoch'] - time.time() - 60))
            dump({'UTC': utc(), 'source_sha256': sha(R / 'ops/audit_terminal.py'),
                  'clock_sha256': sha(R / 'CLOCK.json'), 'limit_seconds': limit,
                  'scientific_selection_change_allowed': False}, R / 'ARTIFACT_AUDIT_DISPATCH.json')
            with (R / 'artifact_audit_private.log').open('ab') as log:
                result = subprocess.run([sys.executable, str(R / 'ops/audit_terminal.py')],
                                        stdout=log, stderr=subprocess.STDOUT, timeout=limit)
            if result.returncode:
                dump({'status': 'FAILED', 'UTC': utc(), 'returncode': result.returncode,
                      'private_diagnostic_sha256': sha(R / 'artifact_audit_private.log'),
                      'promotion_allowed': False}, R / 'ARTIFACT_CONSISTENCY_AUDIT.json')
            return
        time.sleep(30)
    dump({'status': 'SKIPPED_ORIGINAL_DEADLINE', 'UTC': utc(), 'promotion_allowed': False},
         R / 'ARTIFACT_CONSISTENCY_AUDIT.json')

if __name__ == '__main__':
    try:
        main()
    except BaseException:
        (R / 'artifact_audit_failure_private.log').write_text(traceback.format_exc())
        dump({'status': 'EXECUTION_FAILED', 'UTC': utc(),
              'private_diagnostic_sha256': sha(R / 'artifact_audit_failure_private.log'),
              'promotion_allowed': False}, R / 'ARTIFACT_CONSISTENCY_AUDIT.json')
        raise
