"""Gate node2002 by exact replay, then delegate disjoint packs and restart collection."""
from pathlib import Path
import datetime,hashlib,json,shlex,subprocess,time,traceback
import numpy as np
R=Path(__file__).resolve().parents[1]
D='/home/faris/logika-horizontal-20261026'
SOCK=str(R/'runtime/node2002/ssh.sock')
N2=['ssh','-o','ConnectTimeout=15','-S',SOCK,'-p','2002','faris@gpu-labkc.ub.ac.id']
N3=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','2003','faris@gpu-labkc.ub.ac.id']
PY=D+'/runtime/env/bin/python'
def state(stage,**kw):
    record={'stage':stage,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw}
    (R/'NODE2002_ADMISSION.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
def remote(node,cmd,timeout=120):
    return subprocess.run(node+[cmd],text=True,capture_output=True,check=True,timeout=timeout)
def sync(node,src,dest):
    subprocess.run(['rsync','-az','-e',shlex.join(node[:-1]),src,dest],check=True,timeout=180)
def main():
    deadline=json.loads((R/'PLAN.json').read_text())['search_deadline_epoch']
    state('waiting_for_runtime_copy')
    while time.time()<deadline:
        b=json.loads((R/'NODE2002_BOOTSTRAP.json').read_text())
        if b['stage']=='failed':raise RuntimeError(b['error'])
        if b['stage']=='inputs_and_runtime_verified':break
        time.sleep(15)
    else:raise TimeoutError('Frozen search deadline reached during setup')
    sync(N2,str(R/'source')+'/',N2[-1]+':'+D+'/source/')
    verify='''
from pathlib import Path
import hashlib,json
r=Path(REMOTE_ROOT)
for f in range(5):
 p=r/f'packets/fold{f}';m=json.loads((p/'manifest.json').read_text())
 for name,want in m['files'].items():
  assert hashlib.sha256((p/name).read_bytes()).hexdigest()==want,(f,name)
print('All five packet manifests verified')
'''.replace('REMOTE_ROOT',repr(D))
    r=subprocess.run(N2+[PY+' -'],input=verify,text=True,capture_output=True,timeout=120)
    assert r.returncode==0,(r.stdout,r.stderr)
    state('baseline_replay_running',packet_verification=r.stdout)
    args=['systemd-run','--user','--quiet','--wait','--pipe','--collect','--unit=logika-horizontal-node2002-replay-20261026',
          '-p','MemoryMax=18G','-p','MemorySwapMax=0','-p','CPUQuota=200%','-p','RuntimeMaxSec=900',
          '-p','WorkingDirectory='+D,'--setenv=CUDA_VISIBLE_DEVICES=0',
          '--setenv=HF_HOME='+D+'/cache','--setenv=LOGIKA_FEATURE_BANK='+D+'/bank',
          PY,D+'/source/worker.py','--pack','baseline90','--fold','0']
    r=remote(N2,shlex.join(args),timeout=950)
    evidence=R/'evidence/node2002_baseline_replay';evidence.mkdir(parents=True)
    sync(N2,N2[-1]+':'+D+'/runs/small2/baseline90/fold0/',str(evidence)+'/')
    a=np.load(evidence/'prediction.npz');reference=R/'runs/small2/baseline90/fold0';b=np.load(reference/'prediction.npz')
    assert np.array_equal(a['rows'],b['rows'])
    delta=float(np.max(abs(a['raw']-b['raw'])));assert delta<=1e-7,delta
    old=json.loads((reference/'START.json').read_text());new=json.loads((evidence/'START.json').read_text())
    for key in ['columns','seed','num_estimators','model_source_sha256','checkpoint_sha256','packet_manifest_sha256','torch_version']:
        assert old[key]==new[key],key
    gate={'status':'passed','max_abs_prediction_difference':delta,'rows':len(a['raw']),
          'matching_source_model_checkpoint_columns_seed_packet':True,'fitted_weights_saved':False,
          'reference_prediction_sha256':hashlib.sha256((reference/'prediction.npz').read_bytes()).hexdigest(),
          'new_prediction_sha256':hashlib.sha256((evidence/'prediction.npz').read_bytes()).hexdigest(),
          'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (R/'NODE2002_PORTABILITY_GATE.json').write_text(json.dumps(gate,indent=2)+'\n')
    remote(N2,'mkdir -p '+D+'/portability/baseline90; mv '+D+'/runs/small2/baseline90/fold0 '+D+'/portability/baseline90/fold0')
    sync(N3,str(R/'source')+'/',N3[-1]+':'+str(R)+'/source/')
    catalog=json.loads((R/'CATALOG.json').read_text());delegated=[p['id'] for p in catalog['packs'][150:]]
    reserve='''
from pathlib import Path
import json
r=Path(ROOT);packs=PACKS
for pack in packs:
 for fold in range(5):
  p=r/f'runs/small2/{pack}/fold{fold}'
  assert not p.exists(),str(p)
for pack in packs:
 for fold in range(5):
  p=r/f'runs/small2/{pack}/fold{fold}';p.mkdir(parents=True)
  (p/'DELEGATED.json').write_text(json.dumps({'execution_host':'node2002','pack':pack,'fold':fold})+'\\n')
print('Reserved disjoint node2002 packs:',len(packs))
'''.replace('ROOT',repr(str(R))).replace('PACKS',repr(delegated))
    r=subprocess.run(N3+['python3 -'],input=reserve,text=True,capture_output=True,timeout=60)
    assert r.returncode==0,(r.stdout,r.stderr)
    topology={'nodes':[
        {'name':'node2003','ssh':N3,'root':str(R),'python':'/mnt/home/faris/faris/kumo-coverage-placement-20261021/.venv/bin/python',
         'screening_units':['logika-horizontal-lane0-20261026.service','logika-horizontal-lane1-20261026.service'],
         'HF_HOME':'/mnt/home/faris/logika-model-arena-20261025/storage/huggingface',
         'bank':'/mnt/home/faris/faris/logika-meta-experiments/meta-20261016-v1/bank',
         'confirmation_lanes':[{'lane':0,'gpu':0},{'lane':1,'gpu':1}]},
        {'name':'node2002','ssh':N2,'root':D,'python':PY,
         'screening_units':['logika-horizontal-node2002-gpu0-20261026.service','logika-horizontal-node2002-gpu1-20261026.service'],
         'HF_HOME':D+'/cache','bank':D+'/bank',
         'confirmation_lanes':[{'lane':2,'gpu':0},{'lane':3,'gpu':1}]}],
        'coordinator':'node2001','delegation_start_pack_index':150,'scientific_protocol_changed':False,
        'user_authorization':'Move work from A100 to nodes 2001, 2002 and 2003'}
    for gpu in [0,1]:
        args=['systemd-run','--user','--collect',f'--unit=logika-horizontal-node2002-gpu{gpu}-20261026',
              '-p','MemoryMax=1G','-p','RuntimeMaxSec='+str(max(1,int(deadline-time.time()+30))),
              '-p','WorkingDirectory='+D,'--setenv=CUDA_VISIBLE_DEVICES='+str(gpu),
              '--setenv=HF_HOME='+D+'/cache','--setenv=LOGIKA_FEATURE_BANK='+D+'/bank',
              PY,D+'/source/controller.py','--lane',str(gpu),'--lanes','2','--start-index','150']
        remote(N2,shlex.join(args))
    (R/'TOPOLOGY.json').write_text(json.dumps(topology,indent=2)+'\n')
    subprocess.run(['systemctl','--user','stop','logika-horizontal-finish-20261026.service'],check=True)
    args=['systemd-run','--user','--collect','--unit=logika-horizontal-finish-cluster-20261026',
          '-p','MemoryMax=8G','-p','RuntimeMaxSec='+str(max(1,int(json.loads((R/'PLAN.json').read_text())['deadline_epoch']-time.time()+300))),
          '-p','WorkingDirectory='+str(R),'/mnt/home/faris/logika-model-arena-20261025/env311/bin/python',str(R/'source/finish.py')]
    subprocess.run(args,check=True)
    state('cluster_screening_running',node2002_packs=len(delegated),node2003_packs=150,GPU_workers=4,
          coordinator='node2001',portability_max_abs=delta,A100_active=False)
if __name__=='__main__':
    try:main()
    except BaseException:state('failed',error=traceback.format_exc());raise
