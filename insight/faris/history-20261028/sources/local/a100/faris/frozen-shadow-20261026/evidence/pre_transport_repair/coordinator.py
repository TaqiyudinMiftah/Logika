"""Use available lab CPU/GPU slots for the frozen model-only queue, with a fixed clock."""
from pathlib import Path
import datetime,json,shlex,subprocess,sys,time,traceback
import numpy as np
from common import R,dump,sha,utc
def state(stage,**kw):
    value={'stage':stage,'UTC':utc(),**kw};dump(value,R/'LIVE_STATE.json');print(json.dumps(value),flush=True)
def remote(node,cmd,timeout=60):return subprocess.run(node['ssh']+[cmd],text=True,capture_output=True,check=True,timeout=timeout)
def sync(node,src,dst):
    subprocess.run(['rsync','-az','-e',shlex.join(node['ssh'][:-1]),src,dst],check=True,timeout=300)
def deploy(node):
    if node.get('local'):return
    remote(node,'mkdir -p '+node['root'])
    for folder in ['source','frozen']:
        sync(node,str(R/folder)+'/',node['ssh'][-1]+':'+node['root']+'/'+folder+'/')
    for name in ['FEATURE_FREEZE.json','CONFIGS.json','CHECKPOINTS.json','PLAN.json','GATE.json']:
        sync(node,str(R/name),node['ssh'][-1]+':'+node['root']+'/'+name)
    code="from pathlib import Path;import hashlib,json;r=Path("+repr(node['root'])+");m=json.loads((r/'FEATURE_FREEZE.json').read_text());[(__import__('builtins').exec(\"assert hashlib.sha256((r/f'frozen/fold{f}/features.parquet').read_bytes()).hexdigest()==json.loads((r/f'frozen/fold{f}/MANIFEST.json').read_text())['features_sha256']\",{'hashlib':hashlib,'json':json,'r':r,'f':f})) for f in range(5)];print('All5frozen tables verified')"
    remote(node,'python3 -c '+shlex.quote(code),timeout=180)
def resources(node):
    code="from pathlib import Path;import json,os,subprocess;m=dict(l.split(':',1)for l in Path('/proc/meminfo').read_text().splitlines());g=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);print(json.dumps({'cores':os.cpu_count(),'RAM_GiB':int(m['MemAvailable'].split()[0])/2**20,'GPU':[list(map(int,l.split(','))) for l in g.strip().splitlines()]}))"
    if node.get('local'):r=subprocess.run(['python3','-c',code],text=True,capture_output=True,check=True,timeout=15)
    else:r=remote(node,'python3 -c '+shlex.quote(code),timeout=20)
    return json.loads(r.stdout)
def start(node,task,slot,kind,clock):
    config,fold=task;unit=f'logika-frozen-{config}-f{fold}'
    limit=int(min(2400,clock['deadline_epoch']-time.time()-600));assert limit>0
    args=['systemd-run','--user','--quiet','--wait','--pipe','--collect','--unit='+unit,
          '-p','MemoryMax='+('22G' if kind=='gpu' else '8G'),'-p','MemorySwapMax=0','-p','CPUQuota=200%',
          '-p','RuntimeMaxSec='+str(limit),'-p','WorkingDirectory='+node['root'],
          '--setenv=CUDA_VISIBLE_DEVICES='+(str(slot) if kind=='gpu' else ''),'--setenv=HF_HOME='+node['HF_HOME'],
          node['python'],node['root']+'/source/model_worker.py','--config',config,'--fold',str(fold)]
    command=args if node.get('local') else node['ssh']+[shlex.join(args)]
    (R/'logs').mkdir(exist_ok=True);log=(R/'logs'/f'{config}_f{fold}_{node["name"]}.log').open('ab')
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    return {'process':process,'node':node,'task':task,'slot':slot,'kind':kind,'log':log,'unit':unit,'start':time.time()}
def collect(job):
    config,fold=job['task'];root=R/f'runs/{config}/fold{fold}'
    if not job['node'].get('local'):
        root.mkdir(parents=True,exist_ok=True);node=job['node'];sync(node,node['ssh'][-1]+':'+node['root']+f'/runs/{config}/fold{fold}/',str(root)+'/')
    return root
def audit(root,fold):
    done=json.loads((root/'RESULT.json').read_text());assert done['status']=='complete' and done['fitted_weights_saved'] is False
    assert sha(root/'prediction.npz')==done['prediction_sha256'];z=np.load(root/'prediction.npz');roles=np.load(R/f'frozen/fold{fold}/roles.npz')
    assert np.array_equal(z['rows'],np.r_[roles['correction'],roles['validation']]) and len(z['raw'])==20000 and np.isfinite(z['raw']).all()
    assert z['raw'].min()>=0 and z['raw'].max()<=1
def main():
    assert json.loads((R/'GATE.json').read_text())['status']=='passed_for_declared_development_protocol'
    if (R/'CLOCK.json').exists():clock=json.loads((R/'CLOCK.json').read_text())
    else:
        now=time.time();clock={'started_epoch':now,'started_UTC':utc(),'search_deadline_epoch':now+18*3600,'deadline_epoch':now+24*3600,'hours':24,
                             'feature_freeze_sha256':sha(R/'FEATURE_FREEZE.json'),'plan_sha256':sha(R/'PLAN.json')};dump(clock,R/'CLOCK.json')
    assert time.time()<clock['deadline_epoch']
    nodes=json.loads((R/'TOPOLOGY.json').read_text())['nodes'];state('deploying_frozen_features',clock=clock)
    for node in nodes:deploy(node)
    configs=json.loads((R/'CONFIGS.json').read_text());pending={'cpu':[],'gpu':[]};finished=[];failed=[];active=[];blocked=[]
    # Group all fivefold runs for each design together so early termination leaves useful complete cohorts.
    for config,spec in configs.items():
        for fold in range(5):
            root=R/f'runs/{config}/fold{fold}'
            if (root/'RESULT.json').exists():audit(root,fold);finished.append((config,fold))
            elif (root/'FAILURE.json').exists():failed.append({'config':config,'fold':fold,'prior_failure':True})
            else:pending[spec['resource']].append((config,fold))
    state('modelling',completed=len(finished),expected=340,active=0,failed=len(failed),pending=sum(map(len,pending.values())))
    last_state=0
    while any(pending.values()) or active:
        changed=False
        for job in list(active):
            rc=job['process'].poll()
            if rc is None:continue
            job['log'].close();root=collect(job)
            if rc==0:audit(root,job['task'][1]);finished.append(job['task'])
            else:failed.append({'config':job['task'][0],'fold':job['task'][1],'node':job['node']['name'],'returncode':rc,'path':str(root)})
            active.remove(job);changed=True
        stop=(R/'STOP').exists() or time.time()>=clock['search_deadline_epoch']
        if not stop:
            for node in nodes:
                try:info=resources(node)
                except (subprocess.SubprocessError,ValueError):continue
                jobs=[j for j in active if j['node']['name']==node['name']]
                used_gpu={j['slot'] for j in jobs if j['kind']=='gpu'}
                if pending['gpu'] and info['RAM_GiB']>=14:
                    for gpu,used,util in info['GPU']:
                        if not pending['gpu']:break
                        if used<256 and util<20 and gpu not in used_gpu:
                            job=start(node,pending['gpu'].pop(0),gpu,'gpu',clock);active.append(job);jobs.append(job);used_gpu.add(gpu);changed=True
                cpu_count=sum(j['kind']=='cpu' for j in jobs)
                # Preserve two hardware threads for host services. GPU jobs receive their own two-thread quota.
                busy_gpus={gpu for gpu,used,util in info['GPU'] if used>=256 or util>=20}|used_gpu
                capacity=max(1,min(5,(info['cores']-2-2*len(busy_gpus))//2))
                while pending['cpu'] and cpu_count<capacity and info['RAM_GiB']>=8:
                    task=pending['cpu'].pop(0);job=start(node,task,cpu_count,'cpu',clock);active.append(job);cpu_count+=1;changed=True
                    info['RAM_GiB']-=2
        if changed or time.time()-last_state>120:
            state('modelling',completed=len(finished),expected=340,failed=len(failed),active=[{'node':j['node']['name'],'kind':j['kind'],'task':j['task']}for j in active],
                  pending={k:len(v) for k,v in pending.items()},failures=failed);last_state=time.time()
        if stop and not active:break
        if any(pending.values()) or active:time.sleep(15)
    dump({'completed':finished,'failed':failed,'pending':pending,'UTC':utc()},R/'MODEL_RUN_AUDIT.json')
    state('selecting_and_evaluating',completed=len(finished),failed=len(failed))
    subprocess.run([sys.executable,str(R/'source/evaluate.py')],check=True,timeout=1800)
    result=json.loads((R/'evaluation/RESULT.json').read_text());state('evaluated',completed=len(finished),failed=len(failed),primary=result['primary'],
             target_reached=result['target_reached'],development_gain_supported=result['development_gain_supported'],submission_made=False)
if __name__=='__main__':
    try:main()
    except BaseException:state('failed',error=traceback.format_exc());raise
