from pathlib import Path
import shutil,json,time,subprocess,concurrent.futures,shlex,hashlib
H=Path.home();OLD=H/'logika-midnight-multiseed-20261030';R=H/'logika-final-three-20261030';R.mkdir(exist_ok=False)
for n in ['worker.py','compose_final.py','SOURCE_SEAL.json','PREPUBLICATION_SOURCE.json','sample_submission.csv']:
    shutil.copy2(OLD/n,R/n)
shutil.copytree(OLD/'source',R/'source')
for n in ['packets','original','evaluation_inputs']:(R/n).symlink_to(OLD/n,target_is_directory=True)
clock=dict(started_epoch=time.time(),admission_end_epoch=1793379060.,model_deadline_epoch=1793379195.,deadline_epoch=1793379240.,authorization='New explicit human: exactly ecommerce feature experiment and literal seeds3 and4, then stop/cancel all experiment workers. Twenty-minute final-submission window.',old_campaign_clock_preserved=True)
# 16:51 admissions,16:53:15 models,16:54 final; no extension of completed clocks.
assert time.time()<clock['admission_end_epoch']
(R/'CLOCK.json').write_text(json.dumps(clock,indent=2)+'\n')
config=json.loads((OLD/'CONFIG.json').read_text());config.update(id='final-three-20261030',new_seeds=[3,4],exact_experiments=['ecommerce_features_seed2026','seed3_fixed_pipeline','seed4_fixed_pipeline'],scientific_fit_slots=60,qualification_reference_fits=0,planned_fit_slots=70,primary='Exactly three recipes; single-seed3 and single-seed4 fixed pipelines plus one matched ecommerce LocalGLM addition. No combined seed bag or additional candidate search.',runtime_gate='Inherit completed exact-source full-size branch qualification on the same hosts. Prior full pair runtime180-187sec. Hard phase admission and model deadline; incomplete recipes excluded.',no_automatic_Kaggle=True)
(R/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n');(R/'GATE.json').write_text(json.dumps(dict(status='PASS',epoch=time.time(),inherited_qualification=hashlib.sha256((OLD/'GATE.json').read_bytes()).hexdigest(),same_frozen_sources=True,no_extra_reference_fits=True))+'\n')
queue='''from pathlib import Path
import sys,json,time,subprocess,concurrent.futures,os
R=Path(__file__).resolve().parent;p=json.loads((R/('PLAN_'+sys.argv[1]+'.json')).read_text());C=json.loads((R/'CLOCK.json').read_text());results=[]
def task(t):
 ident=f"{t['kind']}__s{t['seed']}__f{t['fold']}";start=time.time()
 for phase in ['validation','production']:
  out=R/'runs'/ident/phase
  if out.exists():return dict(task=t,status='existing_attempt_not_retried')
  if time.time()>=C['admission_end_epoch']:return dict(task=t,status='not_admitted_clock')
  cmd=[p['python'],'-u',str(R/'worker.py'),'--kind',t['kind'],'--seed',str(t['seed']),'--fold',str(t['fold']),'--phase',phase]
  log=R/'logs'/(ident+'_'+phase+'.log');log.parent.mkdir(exist_ok=True);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(p.get('gpu',''));env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
  with log.open('ab') as f:
   try:q=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,C['model_deadline_epoch']-time.time()))
   except subprocess.TimeoutExpired:return dict(task=t,status='owned_deadline_interruption',seconds=time.time()-start)
  if q.returncode:return dict(task=t,status='failed_not_retried',seconds=time.time()-start)
 return dict(task=t,status='complete_pair',seconds=time.time()-start)
with concurrent.futures.ThreadPoolExecutor(max_workers=p['slots']) as ex:
 fs=[ex.submit(task,t) for t in p['tasks']]
 for f in concurrent.futures.as_completed(fs):
  results.append(f.result());(R/('STATE_'+p['owner']+'.json')).write_text(json.dumps(dict(epoch=time.time(),results=results),indent=2)+'\\n')
(R/('FINAL_'+p['owner']+'.json')).write_text(json.dumps(dict(epoch=time.time(),owner=p['owner'],results=results),indent=2)+'\\n')
'''
(R/'queue.py').write_text(queue)
plans={
 'gpu0':dict(owner='gpu0',gpu=0,slots=1,python='/mnt/home/faris/logika-submit-best-sub25k-20261029/env_gpu/bin/python',tasks=[dict(kind='kumo70',seed=3,fold=f)for f in range(5)]),
 'gpu1':dict(owner='gpu1',gpu=1,slots=1,python='/mnt/home/faris/logika-submit-best-sub25k-20261029/env_gpu/bin/python',tasks=[dict(kind='kumo70',seed=4,fold=f)for f in range(5)]),
 'cpu119':dict(owner='cpu119',slots=6,python='/home/faris/logika-next-neural-cann-20261029/env311/bin/python',tasks=[dict(kind='localglm',seed=s,fold=f)for s in [3,4]for f in range(5)]),
 'cpu42':dict(owner='cpu42',slots=6,python='/home/faris/logika-next-neural-cann-20261029/env311/bin/python',tasks=[dict(kind='cann',seed=s,fold=f)for s in [3,4]for f in range(5)])}
for n,v in plans.items():(R/('PLAN_'+n+'.json')).write_text(json.dumps(v,indent=2)+'\n')
routes=[('lab2003',['ssh','-p','2003','faris@gpu-labkc.ub.ac.id'],'/mnt/home/faris',['gpu0','gpu1']),('host119',['ssh','-i','/mnt/home/faris/.ssh/id_ed25519_logika_backup119','faris@100.89.211.119'],'/home/faris',['cpu119']),('host42',['ssh','-i','/mnt/home/faris/.ssh/id_ed25519_logika_backup119','faris@100.98.169.42'],'/home/faris',['cpu42'])]
def deploy(route):
 name,ssh,home,owners=route;dest=home+'/logika-final-three-20261030'
 tar=subprocess.Popen(['tar','-C',str(R),'-cf','-','.'],stdout=subprocess.PIPE)
 commands='mkdir -p '+dest+' && tar -xf - -C '+dest+' && rm '+dest+'/packets '+dest+'/original '+dest+'/evaluation_inputs && ln -s '+home+'/logika-midnight-multiseed-20261030/packets '+dest+'/packets && ln -s '+home+'/logika-midnight-multiseed-20261030/original '+dest+'/original && ln -s '+home+'/logika-midnight-multiseed-20261030/evaluation_inputs '+dest+'/evaluation_inputs'
 q=subprocess.run(ssh+[commands],stdin=tar.stdout,capture_output=True);tar.stdout.close();tc=tar.wait();assert q.returncode==0,(name,q.stderr.decode());assert tc==0
 if name=='lab2003':subprocess.run(ssh+['mkdir -p '+dest+'/checkpoints && ln -s '+home+'/logika-midnight-multiseed-20261030/checkpoints/large.pt '+dest+'/checkpoints/large.pt'],check=True)
 for owner in owners:
  p=plans[owner];cmd=['systemd-run','--user','--unit','logika-final-three-'+owner+'-20261030','--property=MemoryMax='+('16G'if 'gpu'in owner else'32G'),'--property=CPUQuota='+str(p['slots']*200)+'%','--property=StandardOutput=append:'+dest+'/owner_'+owner+'.log','--property=StandardError=append:'+dest+'/owner_'+owner+'.log',p['python'],'-u',dest+'/queue.py',owner]
  subprocess.run(ssh+[shlex.join(cmd)],check=True)
 print(name,'launched',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3)as ex:list(ex.map(deploy,routes))
(R/'DISPATCH_COMPLETE.json').write_text(json.dumps(dict(epoch=time.time(),plans=plans,no_other_experiments_authorized=True))+'\n')
