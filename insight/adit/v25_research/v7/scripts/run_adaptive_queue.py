"""Finish pending fixed fusion, sync, then run five resource-aware follow-ups."""
from pathlib import Path
import json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'
sys.path[:0]=[str(ROOT.parent),str(V/'scripts')]
from continue_frozen_kumo_queue import persistent_status


def state(stage,**extra):
    value=dict(status=stage,pid=os.getpid(),UTC=time.time(),Kumo_frozen_fraction=1.,experiments=['V186','V187','V188','V189','V190'],**extra)
    persistent_status(V/'reports/ADAPTIVE_CONTROLLER.json',value);print(json.dumps(value),flush=True)


def run(*args):
    subprocess.run(list(map(str,args)),cwd=ROOT,check=True)


def main():
    receipt=V/'modeling/reports/GITHUB_SYNC_V184.json'
    assert receipt.exists(),'Record and push V184 before subsequent recipes'
    head=json.loads(receipt.read_text())['commit']
    env=ROOT/'.venv-local-score/Scripts/python.exe'
    if not (V/'modeling/reports/GITHUB_SYNC_V185.json').exists():
        if not (V/'runs/V185/RESULT.json').exists():
            state('RUNNING_PENDING_V185',reviewed_head=head)
            run(env,V/'scripts/score_fusion_local.py','--run','V185','--expected-head',head)
        state('AUDITING_AND_UPLOADING_V185')
        run(env,V/'scripts/audit_frozen_result.py','V185')
        run(sys.executable,V/'scripts/upload_local_score_run.py','V185')
        from node480_task import client
        command="""python - <<'PY'
import subprocess
r='/home/fariselhakim/adit/Logika/insight/goal054_offset_baseline_20261019'
subprocess.run(['/home/fariselhakim/adit/.venv-logika-offset-20261027/bin/python',r+'/v7/scripts/master_log.py'],cwd=r,check=True)
PY"""
        assert client().execute(command,'python3')==0,'Ledger refresh failed'
        state('PUBLISHING_V185')
        run(sys.executable,V/'scripts/sync_completed_experiment.py','V185','--expected-head',head)
    head=json.loads((V/'modeling/reports/GITHUB_SYNC_V185.json').read_text())['commit']
    state('STARTING_FIVE_REAL_WORKERS_WHEN_CAPACITY_FITS',reviewed_head=head)
    run(sys.executable,V/'scripts/continue_frozen_kumo_queue.py','--expected-head',head,'--after','V185','--runs','V186','V187','V188','V189','V190','--worker','adaptive_adapters.py')
    state('COMPLETE')


if __name__=='__main__':
    try:main()
    except Exception:
        state('STOPPED_FOR_FAILURE_OR_PEER_REVIEW',traceback=traceback.format_exc());raise
