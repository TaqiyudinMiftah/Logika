"""Read back terminal artifacts inside the unchanged computation clock."""
from pathlib import Path
import fcntl
import subprocess
import sys
import time
import traceback

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'source'))
from common import check_seal,load,dump,sha,utc

def main():
    with (R/'artifact_audit.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        clock=check_seal()
        while time.time()<clock['deadline_epoch']-90:
            if (R/'FINAL_EVALUATION.json').exists() and (R/'CAMPAIGN_COMPLETION.json').exists():
                assert not (R/'ARTIFACT_CONSISTENCY_AUDIT.json').exists(), 'Reconcile previous audit before another attempt'
                limit=min(900,int(clock['deadline_epoch']-time.time()-30))
                dump(dict(UTC=utc(),audit_source_sha256=sha(R/'ops/audit_terminal.py'),
                          clock_sha256=sha(R/'CLOCK.json'),limit_seconds=limit,
                          scope='Hashes, registered attempts, packet/row boundaries, complete-cohort reconstruction, independent KS replay and conditional arithmetic only.',
                          scientific_selection_change_allowed=False),R/'ARTIFACT_AUDIT_DISPATCH.json')
                with (R/'logs/artifact_audit_private.log').open('ab') as log:
                    result=subprocess.run([sys.executable,str(R/'ops/audit_terminal.py')],stdout=log,stderr=subprocess.STDOUT,timeout=limit)
                if result.returncode:
                    dump(dict(status='FAILED',UTC=utc(),returncode=result.returncode,
                              private_diagnostic_sha256=sha(R/'logs/artifact_audit_private.log'),promotion_allowed=False),R/'ARTIFACT_CONSISTENCY_AUDIT.json')
                return
            time.sleep(10)
        dump(dict(status='SKIPPED_ORIGINAL_DEADLINE',UTC=utc(),promotion_allowed=False),R/'ARTIFACT_CONSISTENCY_AUDIT.json')

if __name__=='__main__':
    try:main()
    except BaseException:
        (R/'logs/artifact_audit_failure_private.log').write_text(traceback.format_exc())
        dump(dict(status='EXECUTION_FAILED',UTC=utc(),private_diagnostic_sha256=sha(R/'logs/artifact_audit_failure_private.log'),promotion_allowed=False),R/'ARTIFACT_CONSISTENCY_AUDIT.json')
        raise
