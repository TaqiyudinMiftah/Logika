"""Finite campaign coordinator: collect, freeze, confirm, assess; no submissions."""
from pathlib import Path
import datetime,json,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1]
REMOTE=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','2003','faris@gpu-labkc.ub.ac.id']
PY='/mnt/home/faris/faris/kumo-coverage-placement-20261021/.venv/bin/python'

def remote(command):
    return subprocess.run(REMOTE+[command],text=True,capture_output=True,timeout=90)
def collect():
    subprocess.run(['rsync','-az','-e','ssh -p 2003','faris@gpu-labkc.ub.ac.id:'+str(R)+'/',str(R)+'/'],check=True,timeout=300)
def state(stage,**kw):
    payload={'stage':stage,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw}
    (R/'LIVE_STATE.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload),flush=True)
def running(units):
    r=remote('systemctl --user is-active '+' '.join(units))
    if r.returncode==255:raise ConnectionError(r.stderr)
    return any(s=='active' for s in r.stdout.splitlines())
def wait_for(units,deadline,stage):
    while time.time()<deadline:
        if (R/'STOP').exists():raise RuntimeError('Campaign STOP requested')
        if not running(units):return
        time.sleep(45)
    state(stage+'_deadline_reached')

def main():
    plan=json.loads((R/'PLAN.json').read_text())
    state('screening_running',catalog_features=21514,expected_packs=290)
    wait_for(['logika-horizontal-lane0-20261026.service','logika-horizontal-lane1-20261026.service'],plan['search_deadline_epoch']+930,'screening')
    collect()
    sys.path.insert(0,str(R/'source'))
    import synthesize
    complete=synthesize.complete_packs();expected=json.loads((R/'CATALOG.json').read_text())['pack_count']
    if len(complete)<3:raise RuntimeError(f'Only {len(complete)} complete packs; cannot select')
    if not (R/'selection/SELECTION.json').exists():synthesize.select(allow_partial=len(complete)<expected)
    synthesize.report('screening')
    state('screening_complete',complete_packs=len(complete),expected_packs=expected,coverage_complete=len(complete)==expected)
    subprocess.run(['rsync','-az','-e','ssh -p 2003',str(R/'selection')+'/', 'faris@gpu-labkc.ub.ac.id:'+str(R/'selection')+'/'],check=True,timeout=180)
    if time.time()>plan['deadline_epoch']-3600:
        state('finished_with_screening_only',reason='Insufficient confirmation reserve');return
    for lane in [0,1]:
        cmd=f'systemd-run --user --collect --unit=logika-horizontal-confirm-lane{lane}-20261026 -p MemoryMax=1G -p RuntimeMaxSec=14400 -p WorkingDirectory={R} --setenv=CUDA_VISIBLE_DEVICES={lane} {PY} {R}/source/confirmation.py --lane {lane}'
        r=remote(cmd);assert r.returncode==0,(r.stdout,r.stderr)
    state('larger_model_confirmation_running')
    wait_for(['logika-horizontal-confirm-lane0-20261026.service','logika-horizontal-confirm-lane1-20261026.service'],plan['deadline_epoch']-600,'confirmation')
    collect()
    selection=json.loads((R/'selection/SELECTION.json').read_text())
    missing=[t for t in selection['tasks'] if not (R/f'runs/{t["size"]}{t["estimators"]}/{t["pack"]}/fold{t["fold"]}/RESULT.json').exists()]
    if missing:
        state('finished_with_incomplete_confirmation',missing=missing);return
    synthesize.report('confirmation')
    subprocess.run([sys.executable,str(R/'source/uncertainty.py')],check=True,timeout=900)
    state('evaluated',report=str(R/'reports/confirmation_RESULT.json'),submission_made=False)

if __name__=='__main__':
    try:main()
    except BaseException:
        state('failed',error=traceback.format_exc());raise
