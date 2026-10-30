from pathlib import Path
import subprocess,shlex,json,time,concurrent.futures,hashlib
R=Path('/mnt/home/faris/logika-final-three-20261030');OLD=Path('/mnt/home/faris/logika-midnight-multiseed-20261030')
# Correct a setup-only module collision before any scientific worker existed.
executor='''from pathlib import Path
import sys,json,time,subprocess,concurrent.futures,os
R=Path(__file__).resolve().parent;p=json.loads((R/('PLAN_'+sys.argv[1]+'.json')).read_text());C=json.loads((R/'CLOCK.json').read_text());results=[]
def fit(t,phase):
 ident=f"{t['kind']}__s{t['seed']}__f{t['fold']}";out=R/'runs'/ident/phase
 if out.exists():return dict(task=t,phase=phase,status='existing_attempt_not_retried')
 if time.time()>=C['admission_end_epoch']:return dict(task=t,phase=phase,status='not_admitted_clock')
 log=R/'logs'/(ident+'_'+phase+'.log');log.parent.mkdir(exist_ok=True);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(p.get('gpu',''));env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 with log.open('ab')as f:
  try:q=subprocess.run([p['python'],'-u',str(R/'worker.py'),'--kind',t['kind'],'--seed',str(t['seed']),'--fold',str(t['fold']),'--phase',phase],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,C['model_deadline_epoch']-time.time()))
  except subprocess.TimeoutExpired:return dict(task=t,phase=phase,status='owned_deadline_interruption')
 return dict(task=t,phase=phase,status='complete'if q.returncode==0 else'failed_not_retried')
def stage(tasks,phase):
 with concurrent.futures.ThreadPoolExecutor(max_workers=p['slots'])as ex:
  for f in concurrent.futures.as_completed([ex.submit(fit,t,phase)for t in tasks]):
   results.append(f.result());(R/('STATE_'+p['owner']+'.json')).write_text(json.dumps(dict(epoch=time.time(),results=results),indent=2)+'\\n')
stage(p['tasks'],'validation');(R/('VALIDATION_DONE_'+p['owner']+'.json')).write_text(json.dumps(dict(epoch=time.time(),results=results))+'\\n')
while time.time()<C['admission_end_epoch'] and not(R/'PRODUCTION_PLAN.json').exists():time.sleep(2)
if (R/'PRODUCTION_PLAN.json').exists():
 selected=json.loads((R/'PRODUCTION_PLAN.json').read_text())['seed'];stage([t for t in p['tasks']if t['seed']==selected],'production')
(R/('FINAL_'+p['owner']+'.json')).write_text(json.dumps(dict(epoch=time.time(),results=results),indent=2)+'\\n')
'''
gpu_tasks=[dict(kind='kumo70',seed=s,fold=f)for s in [3,4]for f in range(5)]
P='/mnt/home/faris/logika-model-arena-20261025/env311/bin/python'
plans={n:json.loads((R/('PLAN_'+n+'.json')).read_text())for n in ['gpu0','gpu1','cpu119','cpu42']}
for i,n in enumerate(['gpu0','gpu1']):plans[n]['tasks']=gpu_tasks[i::3]
plans['gpu2001']=dict(owner='gpu2001',slots=1,gpu=1,python=P,tasks=gpu_tasks[2::3])
policy=dict(epoch=time.time(),exact_experiments=['ecommerce_features_seed2026','seed3_fixed_pipeline','seed4_fixed_pipeline'],seed3and4='literal',validation_first_before_outer_scores=True,complete_OOF_required_for_validation_score=True,production_only_for_highest_complete_recipe=True,no_combined_bags=True,three_idle_GPUs='lab2003:0/1 and verified empty lab2001:1; foreign lab2001:0 untouched',queue_collision_pretraining_failure_preserved=True,old_campaign_clocks_unchanged=True)
(R/'FINAL_THREE_POLICY.json').write_text(json.dumps(policy,indent=2)+'\n')
def start(route):
 name,ssh,home,owners=route;dest=home+'/logika-final-three-20261030'
 code="from pathlib import Path;import json;R=Path("+repr(dest)+");p=R/'queue.py';p.rename(R/'queue_prestart_failure.txt') if p.exists() else None;(R/'owner_execute.py').write_text("+repr(executor)+");(R/'FINAL_THREE_POLICY.json').write_text("+repr(json.dumps(policy,indent=2))+ ");"+';'.join("(R/"+repr('PLAN_'+n+'.json')+").write_text("+repr(json.dumps(plans[n],indent=2))+")"for n in owners)
 subprocess.run(ssh+['python3 -c '+shlex.quote(code)],check=True)
 for n in owners:
  p=plans[n];cmd=['systemd-run','--user','--unit','logika-final-three-'+n+'-repaired-20261030','--property=MemoryMax='+('16G'if 'gpu'in n else'32G'),'--property=CPUQuota='+str(p['slots']*200)+'%','--property=StandardOutput=append:'+dest+'/owner_'+n+'_repaired.log','--property=StandardError=append:'+dest+'/owner_'+n+'_repaired.log',p['python'],'-u',dest+'/owner_execute.py',n]
  subprocess.run(ssh+[shlex.join(cmd)],check=True)
routes=[('lab2003',['ssh','-p','2003','faris@gpu-labkc.ub.ac.id'],'/mnt/home/faris',['gpu0','gpu1']),('host119',['ssh','-i','/mnt/home/faris/.ssh/id_ed25519_logika_backup119','faris@100.89.211.119'],'/home/faris',['cpu119']),('host42',['ssh','-i','/mnt/home/faris/.ssh/id_ed25519_logika_backup119','faris@100.98.169.42'],'/home/faris',['cpu42'])]
with concurrent.futures.ThreadPoolExecutor(max_workers=3)as ex:list(ex.map(start,routes))
if (R/'queue.py').exists():(R/'queue.py').rename(R/'queue_prestart_failure.txt')
(R/'owner_execute.py').write_text(executor);(R/'PLAN_gpu2001.json').write_text(json.dumps(plans['gpu2001'],indent=2)+'\n')
q=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True);row=[x for x in q.stdout.splitlines()if x.startswith('1,')][0];assert int(row.split(',')[1])<100 and int(row.split(',')[2])==0,row
(R/'GPU2001_NEW_IDLE_ADMISSION.json').write_text(json.dumps(dict(epoch=time.time(),actual=row,device=1,foreign_device0_excluded=True,new_explicit_user_experiment_scope=True))+'\n')
(R/'checkpoints').mkdir(exist_ok=True);(R/'checkpoints/large.pt').symlink_to('/mnt/home/faris/logika-model-arena-20261025/storage/huggingface/hub/models--nvidia--Kumo-Tabular/snapshots/b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f/large/classifier.pt')
cmd=['systemd-run','--user','--unit','logika-final-three-gpu2001-20261030','--property=MemoryMax=16G','--property=CPUQuota=200%','--property=StandardOutput=append:'+str(R/'owner_gpu2001.log'),'--property=StandardError=append:'+str(R/'owner_gpu2001.log'),P,'-u',str(R/'owner_execute.py'),'gpu2001'];subprocess.run(cmd,check=True)
print('Validation-first workers started; exactly three recipes',flush=True)
