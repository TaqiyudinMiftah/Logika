"""Compact read-only campaign health; invoke on canonical control host."""
from pathlib import Path
import json,subprocess,hashlib,time,datetime
R=Path('/mnt/home/faris/logika-twelvehour-next-20261029')
def load(p):return json.loads(p.read_text())
result=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),clock_sha256=hashlib.sha256((R/'CLOCK.json').read_bytes()).hexdigest(),components={})
for name in load(R/'UNION_PLAN.json')['component_studies']:
 d=R/name;v=load(d/'LIVE_STATE.json')if(d/'LIVE_STATE.json').exists()else{}
 s={k:v[k]for k in ['stage','UTC','completed_fits','failed_attempts','errors']if k in v};s.update(raw_results=len(list((d/'runs').glob('*/attempt1/RESULT.json'))),complete=(d/'COMPLETION.json').exists())
 if(d/'FINAL_EVALUATION.json').exists():
  e=load(d/'FINAL_EVALUATION.json');s.update(KS=e.get('primary_result',{}).get('metrics',{}).get('KS'),gain_supported=e.get('primary_development_gain_supported'))
 result['components'][name]=s
for n in ['CHECKPOINT42_STATE.json','UNION_SELECTION.json','UNION_ASSESSMENT.json','COMPLETION.json','ARTIFACT_CONSISTENCY_AUDIT.json']:
 if(R/n).exists():
  v=load(R/n);result[n]={k:v for k,v in v.items()if k not in ['files','manifest','passed','missing','failures','recomputed_cohorts']}
units={}
for unit in ['logika-next-master-bounded-20261029','logika-next-single-gpu2002-20261029','logika-next-softks-20261029','logika-next-checkpoint42-20261029']:
 q=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','MainPID','-p','Result'],capture_output=True,text=True);units[unit]=q.stdout.splitlines()
result['units']=units
print(json.dumps(result))
