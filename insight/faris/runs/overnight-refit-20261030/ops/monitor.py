"""Compact remote state and receipt bridge; no fitting or scientific decisions."""
from pathlib import Path
import json,subprocess,time,traceback
R=Path(__file__).resolve().parents[1];C=json.load(open(R/'CLOCK.json'));KEY='/mnt/home/faris/.ssh/id_ed25519_logika_backup119';HOST='faris@100.89.211.119';DEST='/home/faris/logika-overnight-refit-20261030'
def dump(p,v):t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,default=str)+'\n');t.replace(p)
while time.time()<C['deadline_epoch']:
 try:
  plan=json.load(open(R/'UNION_PLAN.json'));states={}
  for n in plan['component_studies']:
   p=R/n;v=json.load(open(p/'LIVE_STATE.json'))if(p/'LIVE_STATE.json').exists()else{'stage':'waiting_data_gate_or_previous_owner'}
   states[n]={k:v.get(k)for k in ['stage','UTC','completed_fits','failed_attempts','planned_fits']}
  ready=json.load(open(R/'READY_CANDIDATE.json'))if(R/'READY_CANDIDATE.json').exists()else None
  for name in ['SUBMITTED.json','SUBMISSION_NEEDS_ATTENTION.json','NO_SUBMISSION.json','FROZEN_SUBMISSION.json']:
   q=subprocess.run(['ssh','-o','ConnectTimeout=10','-i',KEY,HOST,'cat '+DEST+'/'+name],capture_output=True,text=True,timeout=20)
   if q.returncode==0:dump(R/name,json.loads(q.stdout))
  state=dict(stage='submitted'if(R/'SUBMITTED.json').exists()else'running',epoch=time.time(),primary_production_complete=sum((R/f'predictions/{k}/s{s}/fold{f}/RESULT.json').exists()for k in ['cann','kumo25','kumo70']for s in [2026,7731]for f in range(5)),primary_production_expected=30,components=states,current_candidate=ready,deadline_UTC=C['deadline_UTC'],submit_epoch=C['submit_epoch'],models_end_epoch=C['model_deadline_epoch'],old_clocks_unchanged=True,background_services_require_laptop=False,hourly_AI=True)
  dump(R/'LIVE_STATE.json',state);dump(R/'RESUME_STATE.json',state)
  if(R/'SUBMITTED.json').exists():break
 except Exception:(R/'MONITOR_FAILURE_PRIVATE.log').write_text(traceback.format_exc())
 time.sleep(60)
