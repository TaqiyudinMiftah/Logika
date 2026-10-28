"""Ordinary gated deployment. No AI, no score-dependent job selection."""
from pathlib import Path
import os,sys,time,json,hashlib,subprocess,shlex
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];A=R.parent;D=1792585871.823748
remote='/mnt/home/faris/faris/nested-correction-20261021';ret='/mnt/home/faris/faris/context-model-retention-20261021';target='faris@gpu-labkc.ub.ac.id';RET=A/'context-model-retention-20261021'
def write(n,x):
 t=R/(n+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(R/(n+'.json'))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def ssh(cmd):return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-p2003',target,cmd],capture_output=True,text=True)
while True:
 assert time.time()<D-10800 and not(R/'STOP').exists()
 if(R/'FAILURE.json').exists():raise RuntimeError('Feature preparation failed; do not admit models')
 inner=ssh('cat '+ret+'/PREFLIGHT_INNER.json')
 try:proof=json.loads(inner.stdout)if inner.returncode==0 else{}
 except ValueError:proof={}
 if proof.get('status')=='passed'and(RET/'CROSSHOST.json').exists()and(R/'DATA_READY.json').exists():break
 write('TRAINING_PREPARATION',dict(status='waiting_for_data_and_full_retention_gates',inner_retention=proof.get('status'),data_ready=(R/'DATA_READY.json').exists(),crosshost_ready=(RET/'CROSSHOST.json').exists(),original_deadline=D));time.sleep(10)
full=json.loads((RET/'PREFLIGHT.json').read_text());cross=json.loads((RET/'CROSSHOST.json').read_text());assert full['status']=='passed'and cross['status']=='passed'and full['all_pretrained_parameters_frozen']and proof['all_pretrained_parameters_frozen']
assert full['cold_materialization_replay_max_error']<2e-5 and proof['cold_materialization_replay_max_error']<2e-5 and cross['cold_crosshost_replay_max_error']<2e-5
assert full['context_rows']==56000 and proof['context_rows']==44800 and full['checkpoint_sha256']==proof['checkpoint_sha256']
write('RETENTION_GATE',dict(status='passed',full_context=full,inner_context=proof,crosshost=cross,canonical_model='All training context values/labels, columns, seed, pinned inference source and immutable shared pretrained weights. No optimized parameters are discarded. Attention caches are regenerated at inference; no cache checkpoint is produced or deleted.',receipt_clarification='The two engineering preflights read the full labeled training table before slicing contexts and dropping query targets. Their legacy outer_labels_loaded=false field is inaccurate; no query label was passed to models and no score was computed. Actual GPU research packets below contain no query labels.',source='https://huggingface.co/nvidia/Kumo-Tabular'))
assert json.loads((R/'INDEPENDENT_AUDIT.json').read_text())['status']=='passed'
checks=[]
for f in range(5):
 final=f*6+5;fc=pd.read_parquet(R/f'data/job{final}/context.parquet');fq=pd.read_parquet(R/f'data/job{final}/query.parquet');ids=set(fc.application_id);outer=set(fq.application_id);assert len(ids)==56000 and len(outer)==14000 and ids.isdisjoint(outer)and'target'not in fq
 seen=[]
 for j in range(f*6,f*6+6):
  d=R/f'data/job{j}';m=json.loads((d/'manifest.json').read_text())
  for n,h in m['files'].items():assert sha(d/n)==h
  c=pd.read_parquet(d/'context.parquet');q=pd.read_parquet(d/'query.parquet');assert'target'not in q and set(c.application_id).isdisjoint(q.application_id)
  if j!=final:
   assert len(c)==44800 and len(q)==11200 and set(c.application_id)|set(q.application_id)==ids and set(c.application_id).isdisjoint(outer);seen+=q.application_id.tolist()
  checks.append(dict(job=j,manifest_sha256=sha(d/'manifest.json')))
 assert len(seen)==56000 and len(set(seen))==56000 and set(seen)==ids
write('SCIENTIFIC_GATE',dict(status='passed_for_one_nested_recipe',outer_folds=5,inner_folds=5,checks=checks,each_meta_row_raw_prediction_excludes_its_inner_labels=True,all_priors_refit_inside_inner_contexts=True,query_labels_in_GPU_packets=False,historical_label_reuse=True,raw_snapshot_certified=False))
# Shared weights use independent hard links so historical-folder cleanup cannot remove them.
h=full['checkpoint_sha256'];local_weight=A/f'shared-weights/sha256/{h}/classifier.pt';local_weight.parent.mkdir(parents=True,exist_ok=True)
if not local_weight.exists():os.link((A/'ranking-pipeline-20261020/assets/classifier.pt').resolve(),local_weight)
assert sha(local_weight)==h
remote_weight='/mnt/home/faris/faris/shared-weights/sha256/'+h+'/classifier.pt'
command='from pathlib import Path; import os; p=Path('+repr(remote_weight)+'); p.parent.mkdir(parents=True,exist_ok=True); s=Path("/mnt/home/faris/faris/kumo-coverage-placement-20261021/assets/classifier.pt").resolve(); os.link(s,p) if not p.exists() else None'
remote_python='/mnt/home/faris/faris/kumo-coverage-placement-20261021/.venv/bin/python';r=ssh(shlex.join([remote_python,'-c',command]));assert r.returncode==0
host_jobs={'pc2':[j for j in range(30)if j%4<2],'pc3':[j for j in range(30)if j%4>=2]};assert len(set(host_jobs['pc2'])|set(host_jobs['pc3']))==30 and not set(host_jobs['pc2'])&set(host_jobs['pc3'])
source_hashes={z.name:sha(z)for z in(R/'source').glob('*.py')}
plan=dict(id=R.name,training_admitted=True,original_deadline=D,checkpoint_sha256=h,host_jobs=host_jobs,worker_source_sha256=source_hashes,canonical_models_retained=True,context_fits=30,distinct_nested_recipe_count=1,outer_folds=5,inner_folds=5,source_plan_sha256=sha(R/'plan.json'),hypothesis_archive='rounds/primary-011',submission_generated=False)
write('training-plan',plan)
assert ssh('test ! -e '+remote+'/TRAINING_STATUS.json').returncode==0,'Existing controller requires reconciliation'
assert ssh('mkdir -p '+remote+'/source '+remote+'/data').returncode==0
subprocess.run(['scp','-P2003',*map(str,(R/'source').glob('*.py')),target+':'+remote+'/source/'],check=True)
subprocess.run(['scp','-P2003',*[str(R/n)for n in ['training-plan.json','columns.json','head_columns.json','RETENTION_GATE.json','SCIENTIFIC_GATE.json']],target+':'+remote+'/'],check=True)
for j in host_jobs['pc3']:subprocess.run(['scp','-P2003','-r',str(R/f'data/job{j}'),target+':'+remote+'/data/'],check=True)
def launch(host,root,py,checkpoint):
 name='logika-nested-correction-'+host+'-20261021';command=['systemd-run','--user','--unit='+name,'--property=MemoryMax=32G','--property=MemorySwapMax=0','--property=CPUQuota=400%','--setenv=CAMPAIGN_HOST='+host,'--setenv=CHECKPOINT_PATH='+str(checkpoint),'--setenv=OMP_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','/bin/bash','-c','cd '+shlex.quote(root)+'; exec '+shlex.quote(py)+' source/training_entrypoint.py > controller.log 2>&1']
 if host=='pc2':subprocess.run(command,check=True)
 else:
  r=ssh(shlex.join(command));assert r.returncode==0,r.stderr
launch('pc3',remote,remote_python,remote_weight);launch('pc2',str(R),sys.executable,local_weight)
write('TRAINING_PREPARATION',dict(status='admitted',host_jobs=host_jobs,original_deadline=D))
subprocess.run([sys.executable,str(R/'source/collect.py')],check=True)
