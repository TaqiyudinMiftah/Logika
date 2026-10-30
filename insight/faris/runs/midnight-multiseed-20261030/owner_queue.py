"""Bounded owner queue; immutable pair ownership, no failed/unknown attempt retries."""
from pathlib import Path
import concurrent.futures,subprocess,json,time,sys,os,hashlib
R=Path(__file__).resolve().parent;C=json.loads((R/'CLOCK.json').read_text());plan=json.loads((R/'OWNER_PLAN.json').read_text())
def dump(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def pair(t):
 ident=('ref_'if t.get('reference')else'')+f"{t['kind']}__s{t['seed']}__f{t['fold']}";start=time.time()
 for phase in ['validation','production']:
  out=R/'runs'/ident/phase
  if out.exists():
   if (out/'RESULT.json').exists():continue
   return dict(task=t,status='existing_incomplete_not_retried')
  if time.time()>=C['admission_end_epoch']:return dict(task=t,status='admission_closed')
  command=[sys.executable,'-u',str(R/'worker.py'),'--kind',t['kind'],'--seed',str(t['seed']),'--fold',str(t['fold']),'--phase',phase]+(['--reference']if t.get('reference')else[])
  logs=R/'logs';logs.mkdir(exist_ok=True)
  with(logs/(ident+'_'+phase+'.log')).open('ab')as f:
   try:q=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,C['model_deadline_epoch']-time.time()-15))
   except subprocess.TimeoutExpired:
    out.mkdir(exist_ok=True,parents=True);dump(out/'INTERRUPTION.json',dict(reason='owned_model_deadline',epoch=time.time(),no_retry=True));return dict(task=t,status='deadline_interruption')
  if q.returncode:return dict(task=t,status='failed_no_retry')
 return dict(task=t,status='complete_pair',seconds=time.time()-start)
# Reference checks and first full-size pair per kind are explicit qualification exceptions.
results=[];groups={}
for t in plan['tasks']:groups.setdefault(t['kind'],[]).append(t)
qualified={}
for kind,tasks in groups.items():
 pilot=pair(tasks[0]);results.append(pilot);remaining=len(tasks)-1;feasible=pilot['status']=='complete_pair'and time.time()+pilot.get('seconds',1e9)*remaining*1.5/max(1,plan['slots'])+300<C['model_deadline_epoch'];qualified[kind]=feasible
 dump(R/('GATE_'+kind+'.json'),dict(status='PASS'if feasible else'FAIL_RUNTIME_OR_EXECUTION',pilot=pilot,remaining_pairs=remaining,scientific_negative=False))
pending=[t for k,v in groups.items()if qualified[k]for t in v[1:]]
while pending and time.time()<C['admission_end_epoch']:
 gate=R/'GATE.json'
 if gate.exists()and json.loads(gate.read_text()).get('status')=='PASS':break
 time.sleep(3)
if pending and time.time()<C['admission_end_epoch']:
 with concurrent.futures.ThreadPoolExecutor(max_workers=plan['slots'])as ex:
  futures=[ex.submit(pair,t)for t in pending]
  for f in concurrent.futures.as_completed(futures):results.append(f.result());dump(R/'OWNER_STATE.json',dict(epoch=time.time(),results=results))
dump(R/'OWNER_FINAL.json',dict(epoch=time.time(),owner=plan['owner'],results=results,planned_pairs=len(plan['tasks']),no_automatic_retry=True))
