"""Only audited completed studies may trigger frozen full-data fits; source preserved."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time,traceback
R=Path(__file__).resolve().parents[1];PY='/mnt/home/faris/logika-model-arena-20261025/env311/bin/python';EVAL='/mnt/home/faris/logika-sixhour-broad-20261028/env311/bin/python';clock=json.load(open(R/'CLOCK.json'))
while time.time()<clock['admission_end_epoch']-600:
 for namespace in ['helma-book-context10k']:
  study=R/namespace;target=R/'selected_fullfits'/namespace
  if(target/'DISPATCH.json').exists()or not(study/'COMPLETION.json').exists():continue
  try:
   assert json.load(open(study/'ARTIFACT_CONSISTENCY_AUDIT.json'))['status']=='PASS';v=json.load(open(study/'FINAL_EVALUATION.json'));score=v['primary_result']['metrics']['KS'];current=json.load(open(R/'READY_CANDIDATE.json'))
   if score<=current['local_KS']:
    target.mkdir(parents=True,exist_ok=True);(target/'DISPATCH.json').write_text(json.dumps(dict(status='no_refit_lower_local_KS',KS=score,threshold=current['local_KS']))+'\n');continue
   cohort=v['selected_cohort'];design,mode=cohort.rsplit('__',1);spec=json.load(open(study/'CONFIG.json'))['designs'][design];target.mkdir(parents=True,exist_ok=True);source=target/'source';shutil.copytree(study/'source',source,dirs_exist_ok=True)
   m=source/('cpu_models.py'if spec['resource']=='cpu'else'gpu_models.py');text=m.read_text();m.write_text(text.replace('50000','70000').replace('42500','59500').replace('7500','10500').replace('20000','30000'));shutil.copy2(study/'ENVIRONMENT.json',target/'ENVIRONMENT.json');shutil.copy2(study/'CHECKPOINTS.json',target/'CHECKPOINTS.json')
   selection=dict(namespace=namespace,cohort=cohort,design=design,mode=mode,spec=spec,KS=score,kind='selected_'+namespace,original_source_seal_sha256=hashlib.sha256((study/'SOURCE_SEAL.json').read_bytes()).hexdigest(),full_refit_adapter='Same architecture/loss; inner59500/stop10500 then fresh70000model; numerical feature maps fit59500train only.',frozen_before_production=True)
   (target/'SELECTION.json').write_text(json.dumps(selection,indent=2)+'\n');(target/'DISPATCH.json').write_text(json.dumps(dict(epoch=time.time(),status='current_full_refit_starting'))+'\n')
   if spec['resource']=='gpu':
    subprocess.run(['rsync','-az','-e','ssh -p2003',str(target)+'/', 'faris@gpu-labkc.ub.ac.id:'+str(target)+'/'],check=True,timeout=180)
   for seed in [2026,7731]:
    for fold in range(5):
     with(target/f's{seed}_f{fold}.log').open('ab')as f:
      if spec['resource']=='cpu':command=[PY,'-u',str(R/'source/refit_selected.py'),'--namespace',namespace,'--seed',str(seed),'--fold',str(fold)]
      else:command=['ssh','-p2003','faris@gpu-labkc.ub.ac.id','env','CUDA_VISIBLE_DEVICES=0','/mnt/home/faris/logika-submit-best-sub25k-20261029/env_gpu/bin/python','-u',str(R/'source/refit_selected.py'),'--namespace',namespace,'--seed',str(seed),'--fold',str(fold)]
      subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=min(1300,max(1,clock['model_deadline_epoch']-time.time())))
     if spec['resource']=='gpu':subprocess.run(['rsync','-az','-e','ssh -p2003','faris@gpu-labkc.ub.ac.id:'+str(R/f'predictions/{selection["kind"]}')+'/',str(R/f'predictions/{selection["kind"]}')+'/'],check=True,timeout=120)
   subprocess.run([EVAL,str(R/'source/export_selected.py'),namespace],check=True,timeout=300)
   (target/'COMPLETE.json').write_text(json.dumps(dict(epoch=time.time(),fresh_full_data_members=10,completed=True))+'\n')
  except Exception:
   target.mkdir(parents=True,exist_ok=True);(target/'FAILURE_PRIVATE.log').write_text(traceback.format_exc())
 time.sleep(30)
