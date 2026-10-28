"""Deterministic ownership and durable per-PC catalogues, using runtime costs only."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import statistics
import time
from metaexp.common import read,write,sha,event

HOSTS=['pc1','pc2','pc3']

def reserve(coordinator,end):
    root=coordinator.root;path=root/'autonomous_shards.json'
    if path.exists():
        manifest=read(path)
        if manifest['deadline']!=coordinator.cfg['deadline'] or manifest['catalogue_sha256']!=sha(root/'catalogue.json'):raise ValueError('Shard clock or catalogue changed')
        return manifest
    records=coordinator.ledger.all();assigned={r['recipe_id'] for r in records};samples={}
    for row in records:
        p=root/'runs'/row['id']/'scores.json'
        if row['verified'] and row['mode']=='cv' and p.exists():
            score=read(p);samples.setdefault((score['model'],score['device']),[]).append(float(score.get('elapsed_seconds',60)))
    def cost(model,device):
        values=samples.get((model,device),[])[-128:]
        if values:return max(5,statistics.median(values)*1.3)
        return 90 if device=='cpu' and model in ['cat','xgb'] else 30
    capacities={(h,'cpu'):coordinator.cfg['machines'][h]['workers']-(h in ['pc2','pc3']) for h in HOSTS}
    capacities.update({('pc2','cuda'):1,('pc3','cuda'):1})
    if coordinator.cfg['machines']['pc1'].get('opencl'):capacities[('pc1','opencl')]=1
    loads={lane:0.0 for lane in capacities}
    for row in records:
        if row['state']!='assigned':continue
        task=json.loads(row['task']);lane=(row['host'],task.get('device','cpu'))
        if lane in loads:loads[lane]+=cost(task['recipe'].get('model','lgb'),lane[1])/capacities[lane]
    for recipe in coordinator.catalogue['recipes']:
        if recipe['id'] in assigned:continue
        candidates=[lane for lane in capacities if lane[1]=='cpu' or lane[1]=='cuda' and recipe['model'] in ['cat','xgb'] or lane[1]=='opencl' and recipe['model']=='lgb']
        lane=min(candidates,key=lambda pair:(loads[pair]+cost(recipe['model'],pair[1])/capacities[pair],pair))
        coordinator.submit(lane[0],recipe,end,force_device=lane[1],execution_timeout_seconds=900,
                           node_executor_revision=sha(root/'node_executor.py'),ownership_policy='autonomous_static_v1',ownership_policy_sha256=sha(Path(__file__)))
        loads[lane]+=cost(recipe['model'],lane[1])/capacities[lane];assigned.add(recipe['id'])
    tasks={h:[] for h in HOSTS}
    for row in coordinator.ledger.all():
        if row['state']=='assigned':tasks[row['host']].append(json.loads(row['task']))
    manifest={'created':time.time(),'started':coordinator.cfg['started'],'deadline':coordinator.cfg['deadline'],
              'search_deadline':end,'training_code_hash':coordinator.cfg['code_hash'],'catalogue_sha256':sha(root/'catalogue.json'),'planner_sha256':sha(Path(__file__)),
              'policy':'Every recipe has one durable owner. Each original bounded daemon consumes its entire local queue without consulting PC2.',
              'estimated_lane_work_seconds':{h+':'+d:v for (h,d),v in loads.items()},'hosts':tasks}
    write(path,manifest);return manifest

def deliver(coordinator,manifest):
    root=coordinator.root;receipt=root/'shard_delivery.json';state=read(receipt) if receipt.exists() else {}
    todo=[h for h in HOSTS if not state.get(h,{}).get('acknowledged')]
    def install(host):
        payload={'host':host,'campaign':coordinator.cfg['id'],'started':manifest['started'],'deadline':manifest['deadline'],
                 'training_code_hash':manifest['training_code_hash'],'catalogue_sha256':manifest['catalogue_sha256'],'tasks':manifest['hosts'][host]}
        return coordinator.tx.call(host,{'action':'install_shard','shard':payload},timeout=120)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(install,h):h for h in todo}
        for future in as_completed(jobs):
            host=jobs[future]
            try:
                reply=future.result();state[host]={'acknowledged':True,'time':time.time(),'tasks':len(manifest['hosts'][host]),'reply':reply}
                for task in manifest['hosts'][host]:
                    coordinator.tx.delivered.add((host,task['id']));coordinator.tx.buffer[host].pop(task['id'],None)
            except Exception as exc:state[host]={'acknowledged':False,'time':time.time(),'error':str(exc)}
    write(receipt,state)
    if todo:event(root/'scheduler_parallel.jsonl',event='autonomous_shards_delivery',receipts=state)
    return state
