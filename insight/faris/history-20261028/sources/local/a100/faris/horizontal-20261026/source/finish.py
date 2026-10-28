"""Finite home-directory campaign coordinator for node2001, node2002 and node2003."""
from pathlib import Path
import datetime,json,shlex,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1]
DEFAULT={'name':'node2003','ssh':['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','2003','faris@gpu-labkc.ub.ac.id'],
         'root':str(R),'python':'/mnt/home/faris/faris/kumo-coverage-placement-20261021/.venv/bin/python',
         'screening_units':['logika-horizontal-lane0-20261026.service','logika-horizontal-lane1-20261026.service'],
         'HF_HOME':'/mnt/home/faris/logika-model-arena-20261025/storage/huggingface',
         'bank':'/mnt/home/faris/faris/logika-meta-experiments/meta-20261016-v1/bank',
         'confirmation_lanes':[{'lane':0,'gpu':0},{'lane':1,'gpu':1}]}

def nodes():
    return json.loads((R/'TOPOLOGY.json').read_text())['nodes'] if (R/'TOPOLOGY.json').exists() else [DEFAULT]
def remote(node,command):
    return subprocess.run(node['ssh']+[command],text=True,capture_output=True,timeout=90)
def transfer(node,source,destination):
    subprocess.run(['rsync','-az','-e',shlex.join(node['ssh'][:-1]),source,destination],check=True,timeout=300)
def collect():
    for node in nodes():
        for directory in ['runs','logs']:
            (R/directory).mkdir(exist_ok=True)
            transfer(node,node['ssh'][-1]+':'+node['root']+'/'+directory+'/',str(R/directory)+'/')
def state(stage,**kw):
    payload={'stage':stage,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw}
    (R/'LIVE_STATE.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload),flush=True)
def running(stage):
    active=False
    for node in nodes():
        units=node['screening_units'] if stage=='screening' else [f'logika-horizontal-confirm-lane{x["lane"]}-20261026.service' for x in node['confirmation_lanes']]
        r=remote(node,'systemctl --user is-active '+' '.join(units))
        if r.returncode==255:raise ConnectionError(r.stderr)
        active=active or any(s=='active' for s in r.stdout.splitlines())
    return active
def wait_for(deadline,stage):
    while time.time()<deadline:
        if (R/'STOP').exists():raise RuntimeError('Campaign STOP requested')
        if not running(stage):return
        time.sleep(45)
    state(stage+'_deadline_reached')

def main():
    plan=json.loads((R/'PLAN.json').read_text())
    state('screening_running',catalog_features=21514,expected_packs=290,compute_nodes=[n['name'] for n in nodes()],coordinator='node2001')
    wait_for(plan['search_deadline_epoch']+930,'screening')
    collect()
    sys.path.insert(0,str(R/'source'))
    import synthesize
    complete=synthesize.complete_packs();expected=json.loads((R/'CATALOG.json').read_text())['pack_count']
    if len(complete)<3:raise RuntimeError(f'Only {len(complete)} complete packs; cannot select')
    if not (R/'selection/SELECTION.json').exists():synthesize.select(allow_partial=len(complete)<expected)
    synthesize.report('screening')
    state('screening_complete',complete_packs=len(complete),expected_packs=expected,coverage_complete=len(complete)==expected)
    for node in nodes():
        transfer(node,str(R/'selection')+'/',node['ssh'][-1]+':'+node['root']+'/selection/')
    if time.time()>plan['deadline_epoch']-3600:
        state('finished_with_screening_only',reason='Insufficient confirmation reserve');return
    lanes=sum(len(n['confirmation_lanes']) for n in nodes())
    for node in nodes():
        for slot in node['confirmation_lanes']:
            lane,gpu=slot['lane'],slot['gpu']
            command=['systemd-run','--user','--collect',f'--unit=logika-horizontal-confirm-lane{lane}-20261026',
                     '-p','MemoryMax=1G','-p','RuntimeMaxSec=14400','-p','WorkingDirectory='+node['root'],
                     '--setenv=CUDA_VISIBLE_DEVICES='+str(gpu),'--setenv=HF_HOME='+node['HF_HOME'],
                     '--setenv=LOGIKA_FEATURE_BANK='+node['bank'],node['python'],node['root']+'/source/confirmation.py',
                     '--lane',str(lane),'--lanes',str(lanes)]
            r=remote(node,shlex.join(command));assert r.returncode==0,(r.stdout,r.stderr)
    state('larger_model_confirmation_running',GPU_workers=lanes)
    wait_for(plan['deadline_epoch']-600,'confirmation')
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
