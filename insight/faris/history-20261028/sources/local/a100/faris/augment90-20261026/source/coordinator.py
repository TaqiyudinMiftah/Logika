"""Finite, idle-GPU dispatcher; feature gate and exact replay precede broad fits."""
from pathlib import Path
import datetime,hashlib,json,shlex,subprocess,sys,time,traceback
import numpy as np
R=Path(__file__).resolve().parents[1]
OLD=Path('/mnt/home/faris/logika-horizontal-20261026')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state(stage,**kw):
    record={'stage':stage,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw}
    (R/'LIVE_STATE.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
def remote(node,cmd,timeout=90):
    return subprocess.run(node['ssh']+[cmd],text=True,capture_output=True,check=True,timeout=timeout)
def sync(node,source,destination):
    subprocess.run(['rsync','-az','-e',shlex.join(node['ssh'][:-1]),source,destination],check=True,timeout=180)
def deploy(node):
    root=node['root'];old=node['previous_root']
    code='''
from pathlib import Path
r=Path(ROOT);h=Path(OLD)
assert r.is_relative_to(Path.home())
r.mkdir(exist_ok=True)
for name in ['CATALOG.json','BANK_MANIFEST.json','ANCHOR_SOURCE.json','extra.parquet','packets']:
 p=r/name
 if not p.exists():p.symlink_to(h/name,target_is_directory=name=='packets')
'''.replace('ROOT',repr(root)).replace('OLD',repr(old))
    r=subprocess.run(node['ssh']+['python3 -'],input=code,text=True,capture_output=True,timeout=40);assert r.returncode==0,(r.stdout,r.stderr)
    for directory in ['source','selection','replay_reference']:
        sync(node,str(R/directory)+'/',node['ssh'][-1]+':'+root+'/'+directory+'/')
    for name in ['PLAN.json','SOURCE_LEDGER.json','FEATURE_GATE.json','SOURCE_SEAL.json']:
        sync(node,str(R/name),node['ssh'][-1]+':'+root+'/'+name)
def resources(node):
    code="import json,subprocess;from pathlib import Path;g=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);m=dict(l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines());print(json.dumps({'GPU':[list(map(int,l.split(','))) for l in g.strip().splitlines()],'available_GiB':int(m['MemAvailable'].split()[0])/2**20}))"
    return json.loads(remote(node,'python3 -c '+shlex.quote(code),timeout=30).stdout)
def command(node,gpu,task,unit,limit):
    return ['systemd-run','--user','--quiet','--wait','--pipe','--collect','--unit='+unit,
            '-p','MemoryMax=22G','-p','MemorySwapMax=0','-p','CPUQuota=200%',
            '-p','RuntimeMaxSec='+str(limit),'-p','WorkingDirectory='+node['root'],
            '--setenv=CUDA_VISIBLE_DEVICES='+str(gpu),'--setenv=HF_HOME='+node['HF_HOME'],
            '--setenv=LOGIKA_FEATURE_BANK='+node['bank'],node['python'],node['root']+'/source/worker.py',
            '--pack',task['pack'],'--fold',str(task['fold']),'--size',task['size'],'--estimators','8']
def collect(node,task):
    rel=f'runs/{task["size"]}8/{task["pack"]}/fold{task["fold"]}'
    (R/rel).mkdir(parents=True,exist_ok=True)
    sync(node,node['ssh'][-1]+':'+node['root']+'/'+rel+'/',str(R/rel)+'/')
    return R/rel
def check_prediction(root,fold,expected_columns=None):
    result=json.loads((root/'RESULT.json').read_text());start=json.loads((root/'START.json').read_text())
    assert result['status']=='complete' and result['fitted_weights_saved'] is False and result['replay_max_abs']<1e-7
    assert sha(root/'prediction.npz')==result['prediction_sha256']
    z=np.load(root/'prediction.npz');roles=np.load(R/f'packets/fold{fold}/roles.npz')
    assert np.array_equal(z['rows'],np.r_[roles['correction'],roles['validation']])
    assert len(z['raw'])==20000 and np.isfinite(z['raw']).all() and 0<=z['raw'].min()<=z['raw'].max()<=1
    if expected_columns is not None:assert start['columns']==expected_columns
    return z['raw']
def pilot(nodes,plan):
    task={'pack':'replay89','fold':0,'size':'medium'};state('waiting_for_replay_gpu')
    while time.time()<plan['search_deadline_epoch']:
        for node in nodes:
            info=resources(node)
            if info['available_GiB']<14:continue
            for gpu,used,util in info['GPU']:
                if used>=256 or util>=20:continue
                state('replay_pilot_running',node=node['name'],GPU=gpu)
                cmd=command(node,gpu,task,'logika-augment90-replay89-20261026',1200)
                r=remote(node,shlex.join(cmd),timeout=1230)
                root=collect(node,task);reference=json.loads((R/'replay_reference/START.json').read_text())
                raw=check_prediction(root,0,reference['columns']);expected=np.load(R/'replay_reference/prediction.npz')['raw']
                error=float(np.max(abs(raw-expected)));assert error<1e-7,error
                new=json.loads((root/'START.json').read_text())
                for k in ['seed','num_estimators','model_source_sha256','checkpoint_sha256','torch_version','packet_manifest_sha256']:
                    assert new[k]==reference[k],k
                gate={'status':'passed_for_declared_development_experiment','replay89_max_abs':error,
                      'all90_preserved_for_new_controls':True,'selector_outer_labels_loaded':False,
                      'feature_gate_sha256':sha(R/'FEATURE_GATE.json'),'plan_sha256':sha(R/'PLAN.json'),
                      'source_seal_sha256':sha(R/'SOURCE_SEAL.json'),'prior_data_gate_sha256':sha(OLD/'DATA_GATE.json'),
                      'logic_gate_sha256':sha(R/'LOGIC_GATE.json'),
                      'prior_timing_audit_sha256':sha(OLD/'REPAYMENT_TIMING_AUDIT.json'),
                      'limits':'Reused development only. Upstream repayment snapshot semantics remain an assumption.',
                      'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
                (R/'GATE.json').write_text(json.dumps(gate,indent=2)+'\n');return
        time.sleep(15)
    raise TimeoutError('No idle GPU for bounded replay before search deadline')
def main():
    plan=json.loads((R/'PLAN.json').read_text());seal=json.loads((R/'SOURCE_SEAL.json').read_text())
    for name,want in seal.items():assert sha(R/'source'/name)==want,name
    assert json.loads((R/'LOGIC_GATE.json').read_text())['status']=='passed'
    state('waiting_for_feature_selection')
    while not (R/'FEATURE_GATE.json').exists():
        if time.time()>=plan['search_deadline_epoch']:raise TimeoutError('Feature-selection deadline')
        r=subprocess.run(['systemctl','--user','is-active','logika-augment90-features-20261026'],text=True,capture_output=True)
        if r.stdout.strip()!='active':raise RuntimeError('Feature selection stopped without a gate')
        time.sleep(10)
    assert json.loads((R/'FEATURE_GATE.json').read_text())['status']=='complete'
    nodes=json.loads((R/'TOPOLOGY.json').read_text())['nodes']
    for node in nodes:deploy(node)
    if not (R/'GATE.json').exists():pilot(nodes,plan)
    tasks=[]
    for group in ['baseline90']+plan['groups']:
        for fold in range(5):
            selected=json.loads((R/f'selection/fold{fold}.json').read_text())
            if group!='baseline90' and not selected['groups'][group]:continue
            for size in plan['sizes']:tasks.append({'pack':group,'fold':fold,'size':size})
    (R/'TASKS.json').write_text(json.dumps(tasks,indent=2)+'\n')
    pending=[];finished=[];failed=[];active={};(R/'logs').mkdir(exist_ok=True)
    for task in tasks:
        root=R/f'runs/{task["size"]}8/{task["pack"]}/fold{task["fold"]}'
        if (root/'RESULT.json').exists():check_prediction(root,task['fold']);finished.append(task)
        else:pending.append(task)
    state('training',expected=len(tasks),completed=len(finished),pending=len(pending),active=0)
    while pending or active:
        for slot,job in list(active.items()):
            rc=job['process'].poll()
            if rc is None:continue
            job['log'].close();task=job['task'];root=collect(job['node'],task)
            if rc==0:
                selected=json.loads((R/f'selection/fold{task["fold"]}.json').read_text())
                columns=selected['baseline_features']+([] if task['pack']=='baseline90' else selected['groups'][task['pack']])
                check_prediction(root,task['fold'],columns);finished.append(task)
            else:failed.append({**task,'returncode':rc,'node':job['node']['name']})
            del active[slot]
            state('training',expected=len(tasks),completed=len(finished),pending=len(pending),active=len(active),failed=failed)
        stop=(R/'STOP').exists() or time.time()>=plan['search_deadline_epoch']
        if not stop and pending:
            for node in nodes:
                if len(active)>=4:break
                try:info=resources(node)
                except (subprocess.SubprocessError,ValueError):continue
                if info['available_GiB']<14:continue
                for gpu,used,util in info['GPU']:
                    slot=(node['name'],gpu)
                    if slot in active or used>=256 or util>=20 or not pending or len(active)>=4:continue
                    task=pending.pop(0);unit=f'logika-augment90-{task["size"]}-{task["pack"]}-f{task["fold"]}'
                    limit=int(min(1200,plan['deadline_epoch']-time.time()-600));assert limit>0
                    path=R/'logs'/f'{task["size"]}_{task["pack"]}_f{task["fold"]}_{node["name"]}.log'
                    log=path.open('ab');p=subprocess.Popen(node['ssh']+[shlex.join(command(node,gpu,task,unit,limit))],stdout=log,stderr=subprocess.STDOUT)
                    active[slot]={'process':p,'node':node,'task':task,'log':log}
        if stop and not active:break
        if pending or active:time.sleep(10)
    if pending or failed:
        state('finished_incomplete',expected=len(tasks),completed=len(finished),pending=pending,failed=failed);return
    state('freezing_and_evaluating',completed=len(finished))
    subprocess.run([sys.executable,str(R/'source/finalize.py')],check=True,timeout=900)
    result=json.loads((R/'evaluation/RESULT.json').read_text())
    state('evaluated',completed_fits=len(finished),primary_KS=result['primary']['KS'],delta_KS=result['primary']['delta_KS'],
          target_reached=result['target_reached'],seed_confirmation_pending=result['seed_replication_required_before_promotion'],submission_made=False)
if __name__=='__main__':
    try:main()
    except BaseException:state('failed',error=traceback.format_exc());raise
