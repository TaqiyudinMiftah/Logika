"""Bounded CPU/GPU scheduling for the new architecture campaign, on nodes2001/2003 only."""
from pathlib import Path
import json,shlex,subprocess,sys,time,traceback
import numpy as np
from common import R,dump,sha,utc


def state(stage,**kwargs):
    record={'stage':stage,'UTC':utc(),**kwargs};dump(record,R/'LIVE_STATE.json');print(json.dumps(record),flush=True)


def remote(node,cmd,timeout=30):return subprocess.run(node['ssh']+[cmd],capture_output=True,text=True,check=True,timeout=timeout)


def sync(node,src,dst):subprocess.run(['rsync','-az','-e',shlex.join(node['ssh'][:-1]),src,dst],check=True,timeout=240)


def resources(node):
    code="from pathlib import Path;import os,json,subprocess;m=dict(l.split(':',1)for l in Path('/proc/meminfo').read_text().splitlines());g=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);print(json.dumps({'cores':os.cpu_count(),'RAM_GiB':int(m['MemAvailable'].split()[0])/2**20,'GPU':[list(map(int,l.split(',')))for l in g.strip().splitlines()]}))"
    p=subprocess.run(['python3','-c',code],capture_output=True,text=True,check=True,timeout=20)if node.get('local')else remote(node,'python3 -c '+shlex.quote(code),20)
    return json.loads(p.stdout)


def deploy(node):
    if not node.get('local'):
        remote(node,'mkdir -p '+shlex.quote(node['root']))
        sync(node,str(R/'source')+'/',node['ssh'][-1]+':'+node['root']+'/source/')
        for name in ['CONFIGS.json','FEATURE_FREEZE.json','PATHS.json','VALIDATION_CONTRACT.json','CLOCK.json','GATE_PREP.json','GATE.json']:
            if (R/name).exists():sync(node,str(R/name),node['ssh'][-1]+':'+node['root']+'/'+name)
    code="import json,hashlib,importlib.metadata,inspect,tabm;from pathlib import Path;r=Path("+repr(node['root'])+");p=Path(json.loads((r/'PATHS.json').read_text())['frozen_root']);seal=json.loads((r/'FEATURE_FREEZE.json').read_text());assert hashlib.sha256((r/'FEATURE_FREEZE.json').read_bytes()).hexdigest()=='071c75e0f59cf92429bde9c76a64e27931e1eea03bbbbeac2e2589b5fee68dd6';[(lambda d,w:__import__('builtins').exec(\"assert hashlib.sha256((d/'features.parquet').read_bytes()).hexdigest()==w['features_sha256']\",{'d':d,'w':w,'hashlib':hashlib}))(p/f'fold{f}',json.loads((p/f'fold{f}/MANIFEST.json').read_text()))for f in range(5)];print(json.dumps({'torch':importlib.metadata.version('torch'),'tabm':importlib.metadata.version('tabm'),'tabm_source_sha256':hashlib.sha256(Path(inspect.getfile(tabm)).read_bytes()).hexdigest(),'all_frozen_inputs_verified':True}))"
    p=subprocess.run([node['python'],'-c',code],capture_output=True,text=True,check=True,timeout=60)if node.get('local')else remote(node,node['python']+' -c '+shlex.quote(code),60)
    env=json.loads(p.stdout);assert env['tabm']=='0.0.3';return env


def start(node,task,kind,slot,clock,pilot=False):
    name,fold=task;unit=f'logika-architecture-{name}-f{fold}'+('-pilot'if pilot else '')
    limit=min(600 if pilot else 2100,int(clock['deadline_epoch']-time.time()-600));assert limit>0
    args=['systemd-run','--user','--quiet','--wait','--pipe','--collect','--unit='+unit,'-p','WorkingDirectory='+node['root'],
          '-p','CPUQuota=200%','-p','MemoryMax='+('16G'if kind=='gpu'else '8G'),'-p','MemorySwapMax=0','-p','RuntimeMaxSec='+str(limit),
          '--setenv=CUDA_VISIBLE_DEVICES='+(str(slot)if kind=='gpu'else ''),node['python'],node['root']+'/source/worker.py','--config',name,'--fold',str(fold)]
    if pilot:args.append('--pilot')
    (R/'logs').mkdir(exist_ok=True);logpath=R/'logs'/f'{name}_f{fold}_{node["name"]}{"_pilot"if pilot else ""}.log'
    log=logpath.open('ab');process=subprocess.Popen(args if node.get('local')else node['ssh']+[shlex.join(args)],stdout=log,stderr=subprocess.STDOUT)
    job={'node':node,'task':task,'kind':kind,'slot':slot,'unit':unit,'pilot':pilot,'process':process,'log':log,'started_epoch':time.time()}
    path=R/('PILOT_DISPATCH.json'if pilot else 'DISPATCH_LEDGER.json');ledger=json.loads(path.read_text())if path.exists()else {}
    ledger[f'{name}/fold{fold}']={k:v for k,v in job.items()if k not in ['process','log']};dump(ledger,path)
    return job


def probe(job):
    node=job['node'];name,fold=job['task'];kind='pilots'if job.get('pilot')else 'runs'
    code="from pathlib import Path;import json,subprocess;p=Path("+repr(node['root']+f'/{kind}/{name}/fold{fold}')+");q=subprocess.run(['systemctl','--user','show',"+repr(job['unit'])+",'-p','ActiveState','--value'],capture_output=True,text=True);print(json.dumps({'active':q.stdout.strip() in ['active','activating'],'result':(p/'RESULT.json').exists(),'failure':(p/'FAILURE.json').exists()}))"
    p=subprocess.run(['python3','-c',code],capture_output=True,text=True,check=True,timeout=20)if node.get('local')else remote(node,'python3 -c '+shlex.quote(code),20)
    return json.loads(p.stdout)


def collect(job):
    name,fold=job['task'];kind='pilots'if job.get('pilot')else 'runs';root=R/f'{kind}/{name}/fold{fold}';node=job['node']
    if not node.get('local'):
        root.mkdir(parents=True,exist_ok=True);sync(node,node['ssh'][-1]+':'+node['root']+f'/{kind}/{name}/fold{fold}/',str(root)+'/')
    return root


def audit(root,fold):
    result=json.loads((root/'RESULT.json').read_text());start=json.loads((root/'START.json').read_text());assert result['status']=='complete'and result['fitted_weights_saved']is False
    assert sha(root/'prediction.npz')==result['prediction_sha256'];assert start['correction_labels_used_for_training']is False and start['outer_labels_loaded']is False
    roles=np.load(Path(json.loads((R/'PATHS.json').read_text())['frozen_root'])/f'fold{fold}/roles.npz');p=np.load(root/'prediction.npz')
    assert np.array_equal(p['rows'],np.r_[roles['correction'],roles['validation']]);assert p['raw'].shape==(20000,)and np.isfinite(p['raw']).all()and 0<=p['raw'].min()<=p['raw'].max()<=1


def main():
    clock=json.loads((R/'CLOCK.json').read_text());gate=json.loads((R/'GATE.json').read_text());assert gate['status']=='passed_for_declared_development_protocol'
    assert gate['contract_sha256']==sha(R/'VALIDATION_CONTRACT.json')==clock['contract_sha256'];assert sha(R/'CONFIGS.json')==clock['configs_sha256'];assert sha(R/'FEATURE_FREEZE.json')==clock['feature_freeze_sha256']
    configs=json.loads((R/'CONFIGS.json').read_text());nodes=json.loads((R/'TOPOLOGY.json').read_text())['nodes'];assert {n['name']for n in nodes}=={'node2001','node2003'}
    for node in nodes:deploy(node)
    completed=[];failed=[];active=[];errors={};pending=[];ledger=json.loads((R/'DISPATCH_LEDGER.json').read_text())if (R/'DISPATCH_LEDGER.json').exists()else {}
    for name in configs:
        for fold in range(5):
            root=R/f'runs/{name}/fold{fold}';key=f'{name}/fold{fold}'
            if (root/'RESULT.json').exists():audit(root,fold);completed.append((name,fold))
            elif (root/'FAILURE.json').exists():failed.append({'task':[name,fold],'reason':'prior recordedfailure'})
            elif key in ledger:job=ledger[key];job.update(process=None,log=None);active.append(job)
            else:pending.append((name,fold))
    last=0
    while pending or active:
        infos={};changed=False
        for node in nodes:
            try:infos[node['name']]=resources(node);errors.pop(node['name'],None)
            except (subprocess.SubprocessError,ValueError,OSError)as e:errors[node['name']]={'UTC':utc(),'error':str(e)}
        for job in list(active):
            if job['process']is not None and job['process'].poll()is None:continue
            if job['node']['name']not in infos:continue
            try:
                status=probe(job)
                if status['active']:continue
                if status['result']or status['failure']:root=collect(job)
                if status['result']:audit(root,job['task'][1]);completed.append(job['task'])
                else:
                    reason='worker_failure'if status['failure']else 'unit_exited_without_result'
                    failed.append({'task':job['task'],'node':job['node']['name'],'kind':job['kind'],'reason':reason,'scientific_conclusion':'none; inspectexecutionevidence'})
                if job['log']is not None:job['log'].close()
                active.remove(job);changed=True
            except (subprocess.SubprocessError,ValueError,OSError)as e:errors[job['node']['name']]={'UTC':utc(),'error':str(e)}
        stop=(R/'STOP').exists()or time.time()>=clock['search_deadline_epoch']
        if not stop and not (R/'PAUSE_ADMISSION').exists():
            for node in nodes:
                if node['name']not in infos:continue
                info=infos[node['name']];jobs=[j for j in active if j['node']['name']==node['name']];free=info['RAM_GiB']-sum(16 if j['kind']=='gpu'else 8 for j in jobs)
                occupied={j['slot']for j in jobs if j['kind']=='gpu'}
                for gpu,used,util in info['GPU']:
                    if pending and free>=16 and used<256 and util<20 and gpu not in occupied:
                        # GPU-only FT tasks first; the other architectures can also use CPU slots.
                        idx=next((i for i,t in enumerate(pending)if configs[t[0]]['resource']=='gpu'),0)
                        job=start(node,pending.pop(idx),'gpu',gpu,clock);active.append(job);jobs.append(job);occupied.add(gpu);free-=16;changed=True
                busy={gpu for gpu,used,util in info['GPU']if used>=256 or util>=20}|occupied;capacity=max(0,min(5,(info['cores']-2-2*len(busy))//2));cpus=sum(j['kind']=='cpu'for j in jobs)
                while free>=8 and cpus<capacity:
                    idx=next((i for i,t in enumerate(pending)if configs[t[0]]['resource']=='hybrid'),None)
                    if idx is None:break
                    active.append(start(node,pending.pop(idx),'cpu',cpus,clock));cpus+=1;free-=8;changed=True
        if changed or time.time()-last>120:
            state('modelling',completed=len(completed),expected=160,failed=len(failed),pending=len(pending),active=[{k:j[k]for k in ['task','kind','slot','started_epoch']}|{'node':j['node']['name']}for j in active],failures=failed,node_errors=errors);last=time.time()
        if stop and not active:break
        if stop and time.time()>min(clock['search_deadline_epoch']+2100,clock['deadline_epoch']-1800):break
        if pending or active:time.sleep(15)
    dump({'UTC':utc(),'completed':completed,'failed':failed,'pending':pending,'unresolved':[{'task':j['task'],'node':j['node']['name']}for j in active]},R/'MODEL_RUN_AUDIT.json')
    state('selecting_and_evaluating',completed=len(completed),failed=len(failed))
    subprocess.run([sys.executable,str(R/'source/evaluate.py')],check=True,timeout=1500)
    result=json.loads((R/'evaluation/RESULT.json').read_text());state('evaluated',completed=len(completed),failed=len(failed),primary=result['primary'],target_reached=result['target_reached'],development_gain_supported=result['development_gain_supported'],submission_made=False)


if __name__=='__main__':
    try:main()
    except BaseException:state('failed',error=traceback.format_exc());raise
