"""Ordinary PC2 service: independent queues, verified collection, frozen terminal stages."""
import concurrent.futures,datetime,json,os,shlex,shutil,subprocess,sys,tarfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'));from utils import read,write,sha,ks,combine,delta,ident
import numpy as np
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8']

def rpc(request,output=None,payload=None):
 cmd=SSH+['-i',str(ROOT/'.credentials/pc1_ed25519'),'agung@100.89.211.119'];p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE if output is None else output,stderr=subprocess.PIPE)
 p.stdin.write((json.dumps(request)+'\n').encode())
 if payload is not None:
  with Path(payload).open('rb') as f:shutil.copyfileobj(f,p.stdin)
 p.stdin.close();out=p.stdout.read() if p.stdout else b'';err=p.stderr.read();rc=p.wait(timeout=180)
 if rc:raise RuntimeError('pc1 RPC '+err.decode()[-500:])
 return out

def remote(host,cmd,timeout=45):
 if host=='pc1':raise ValueError('Use forced rpc for PC1')
 if host=='pc2':return subprocess.run(cmd,shell=True,check=True,capture_output=True,text=True,timeout=timeout).stdout
 if host=='node380':
  from node_join import node_command
  return node_command(cmd,timeout)
 p=subprocess.run(SSH+['-p','2003','faris@gpu-labkc.ub.ac.id',cmd],check=True,capture_output=True,text=True,timeout=timeout);return p.stdout

def state(host):
 if host=='pc1':return json.loads(rpc({'action':'status'}))
 if host=='node380':
  try:
   from node_join import client
   c,k=client();payload=c.request('GET','api/contents/faris/exploration-12h-20261017/campaign/state.json?format=text&type=file');return json.loads(payload['content'])
  except Exception:return {'status':'unavailable','jobs':{}}
 cfg=read(ROOT/'deployment.json')['hosts'][host];p=cfg['root']+'/campaign/state.json'
 try:return json.loads(remote(host,'cat '+shlex.quote(p),timeout=20))
 except Exception:return {'status':'unavailable','jobs':{}}

def tasks(host,requests):
 if not requests:return
 if host=='pc1':rpc({'action':'tasks','tasks':requests});return
 path=read(ROOT/'deployment.json')['hosts'][host]['root']+'/stage_tasks.json'
 code="from pathlib import Path;import json;p=Path("+repr(path)+");old=json.loads(p.read_text()) if p.exists() else [];known={x['id'] for x in old};old += [x for x in "+repr(requests)+" if x['id'] not in known];t=p.with_suffix('.tmp');t.write_text(json.dumps(old));t.replace(p)"
 remote(host,'python3 -c '+shlex.quote(code))

def collect(host,s):
 cfg=read(ROOT/'deployment.json')['hosts'][host];dest=ROOT/'archive'/host;dest.mkdir(parents=True,exist_ok=True)
 collected=read(dest/'receipts.json') if (dest/'receipts.json').exists() else {}
 pending=[j for j in s.get('jobs',{}).values() if j['status'] in ['complete','failed','interrupted','timeout','invalid'] and j.get('directory') and j['job_id']+'_'+str(j['attempt']) not in collected]
 pending.sort(key=lambda j:(j['status']=='complete',j.get('score',-1)),reverse=True);pending=pending[:128]
 projected=sum(j.get('artifact_mb',j['resources']['disk_mb']) for j in pending)*1048576
 if projected+int(subprocess.check_output(['du','-sb',str(ROOT)],text=True).split()[0])>300*1024**3:raise RuntimeError('300GiB campaign archive ceiling; retain source copies')
 if pending and host=='pc1':
  tmp=dest/'batch-transfer.tar.gz'
  with tmp.open('wb') as f:rpc({'action':'get','paths':['campaign/'+j['directory'] for j in pending]},output=f)
  with tarfile.open(tmp) as t:
   for m in t.getmembers():assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
   t.extractall(dest,filter='data')
  tmp.unlink()
 elif pending and host=='pc3':
  filelist=dest/'batch-files.txt';filelist.write_text(''.join('campaign/'+j['directory']+'/\n' for j in pending))
  subprocess.run(['rsync','-ar','--files-from='+str(filelist),'-e',shlex.join(SSH+['-p','2003']),'faris@gpu-labkc.ub.ac.id:'+cfg['root']+'/',str(dest)+'/'],check=True,timeout=180,stdout=subprocess.DEVNULL)
 allowed={j['job_id']+'_'+str(j['attempt']) for j in pending}
 for j in s.get('jobs',{}).values():
  if j['status'] not in ['complete','failed','interrupted','timeout','invalid'] or not j.get('directory'):continue
  rel='campaign/'+j['directory'];key=j['job_id']+'_'+str(j['attempt'])
  if key in collected or key not in allowed:continue
  folder=dest/rel;folder.mkdir(parents=True,exist_ok=True)
  if host=='pc2':
   source=Path(cfg['root'])/rel
   # PC2 archive and queue share the same filesystem: hardlink immutable files.
   for f in source.rglob('*'):
    if f.is_file():
     p=folder/f.relative_to(source);p.parent.mkdir(parents=True,exist_ok=True)
     if not p.exists():os.link(f,p)
  elif host in ['pc1','pc3']:pass
  else:
   from node_join import download_directory
   download_directory(cfg['root']+'/'+rel,folder)
  if j['status']=='complete':
   out=read(folder/'outcome.json')
   for name,h in out['artifacts'].items():assert sha(folder/name)==h,(host,key,name)
   z=np.load(folder/'validation.npz');bank=Path(read(ROOT/'deployment.json')['hosts']['pc2']['bank']);dev=np.load(bank/'development_rows.npy');labels=np.load(bank/'development_y.npy')
   if j['protocol']=='confirm':
    assert (ROOT/'reports/frozen_finalists.json').exists();expected=np.load(bank/'confirmation_rows.npy');assert np.array_equal(z['rows'],expected);labels=np.load(bank/'confirmation_y.npy')
   else:
    idx={int(v):i for i,v in enumerate(dev)};ix=np.array([idx[int(v)] for v in z['rows']]);labels=labels[ix]
    if j['protocol']!='temporal':assert np.array_equal(z['rows'],dev)
   assert abs(ks(labels,z['prediction'])-out['evaluation']['score'])<1e-12
   # Content sharing preserves identical immutable weights without multiplying disk use.
   for name,checksum in out['artifacts'].items():
    path=folder/name
    if name in out['models'] and path.stat().st_size>16*1024*1024:
     store=ROOT/'archive/objects';store.mkdir(exist_ok=True);canonical=store/checksum
     if not canonical.exists():os.link(path,canonical)
     elif path.stat().st_ino!=canonical.stat().st_ino:
      assert sha(canonical)==checksum;temporary=path.with_name(path.name+'.dedup');os.link(canonical,temporary);temporary.replace(path)

  collected[key]={'time':time.time(),'job':j,'verified':True};write(dest/'receipts.json',collected)
  if j['status']=='complete' and host in ['pc1','pc2','pc3']:
   files={name:checksum for name,checksum in out['artifacts'].items() if name in out['models'] and name.endswith(('.joblib','.pt','.pkl'))}
   request={'action':'prune','directory':rel,'archive':str(folder),'files':files}
   try:
    if host=='pc1':rpc(request)
    elif host=='pc2':
     from archive_ops import prune
     prune(request)
    else:subprocess.run(SSH+['-p','2003','faris@gpu-labkc.ub.ac.id',cfg['root']+'/.venv/bin/python '+cfg['root']+'/archive_ops.py'],input=json.dumps(request),text=True,check=True,capture_output=True,timeout=60)
   except Exception as e:
    with (ROOT/'logs/collection-errors.jsonl').open('a') as handle:handle.write(json.dumps({'host':host,'job':key,'archive_safe_but_prune_failed':str(e)})+'\n')
 # Preserve host-specific seals, code snapshots and records as well as model folders.
 if s.get('jobs'):
  metadata=['campaign/state.json','campaign/seal.json','campaign/config.json','campaign/environment.json','campaign/records','campaign/reports','campaign/resources.jsonl']
  if not (dest/'campaign/snapshots').exists():metadata += ['campaign/snapshots','campaign/runtime']
  if host=='pc1':
   tmp=dest/'metadata.tar.gz'
   with tmp.open('wb') as f:rpc({'action':'get','paths':metadata},output=f)
   with tarfile.open(tmp) as t:
    for m in t.getmembers():assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
    t.extractall(dest,filter='data')
   tmp.unlink()
  elif host=='pc2':
   for name in metadata:
    source=Path(cfg['root'])/name
    if not source.exists():continue
    if source.is_dir():shutil.copytree(source,dest/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    else:(dest/name).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest/name)
  elif host=='pc3':
   subprocess.run(['rsync','-ar','--exclude=runs','--exclude=search.sqlite3*','--exclude=*.pkl','--exclude=__pycache__','-e',shlex.join(SSH+['-p','2003']),'faris@gpu-labkc.ub.ac.id:'+cfg['root']+'/campaign/',str(dest/'campaign')+'/'],check=True,timeout=90,stdout=subprocess.DEVNULL)
 return collected

def records():
 found=[]
 for host in ['pc1','pc2','pc3','node380']:
  path=ROOT/'archive'/host/'receipts.json'
  if not path.exists():continue
  for receipt in read(path).values():
   j=receipt['job'];folder=ROOT/'archive'/host/'campaign'/j['directory']
   if j['status']=='complete':
    out=read(folder/'outcome.json');found.append({'id':host+'_'+j['job_id'],'host':host,'job':j,'folder':str(folder),'evaluation':out['evaluation']})
 return found

def predictions(r):
 z=np.load(Path(r['folder'])/'validation.npz');return z['prediction']

def ensembles(rows,limit=4096):
 bank=Path(read(ROOT/'deployment.json')['hosts']['pc2']['bank']);y=np.load(bank/'development_y.npy');fold=np.load(bank/'folds.npy');champ=np.load(ROOT/'evidence/champion_oof.npz')['existing_feature_ensemble']
 screen=[r for r in rows if r['job']['protocol']=='screen'];screen.sort(key=lambda r:r['evaluation']['score'],reverse=True)
 selected=[];families=set()
 for r in screen:
  f=r['job']['family']
  if f not in families:selected.append(r);families.add(f)
 for r in screen:
  if r not in selected:selected.append(r)
  if len(selected)>=32:break
 arrays={'champion':champ};mapping={'champion':{'kind':'champion'}}
 for r in selected:arrays[r['id']]=predictions(r);mapping[r['id']]=r
 candidates=[];seen=set()
 def add(spec):
  key=ident(spec)
  if key in seen or len(candidates)>=limit:return
  seen.add(key);p=combine(spec,arrays);d=delta(y,p,champ,fold);candidates.append({'id':'ens_'+key[:16],'spec':spec,**d})
 for key in list(arrays)[1:]:
  for weight in [.05,.10,.20,.35,.50]:
   for method in ['mean','logit','geometric','rank']:add({'method':method,'parents':['champion',key],'weights':[1-weight,weight]})
 rng=np.random.default_rng(20261017);keys=list(arrays)
 for count in [3,4,5,7,9]:
  if len(keys)<count:continue
  for _ in range(300):
   parents=sorted(rng.choice(keys,count,replace=False).tolist())
   for method in ['mean','trimmed','logit','geometric']:add({'method':method,'parents':parents})
 candidates.sort(key=lambda r:r['ks'],reverse=True);write(ROOT/'reports/scoreboard_fixed.json',candidates);write(ROOT/'ensemble_sources.json',mapping)
 # Save durable predictions for the selected shortlist, not all candidate matrices.
 for c in candidates[:12]:np.save(ROOT/'reports'/(c['id']+'.npy'),combine(c['spec'],arrays))
 return candidates,mapping

def nominate(rows,candidates,mapping):
 bank=Path(read(ROOT/'deployment.json')['hosts']['pc2']['bank']);y=np.load(bank/'development_y.npy');fold=np.load(bank/'folds.npy');champ=np.load(ROOT/'evidence/champion_oof.npz')['existing_feature_ensemble'];items=[]
 for r in rows:
  if r['job']['protocol']=='screen':
   d=delta(y,predictions(r),champ,fold)
   if d['eligible'] and d['delta']>0:items.append({'id':r['id'],'spec':{'method':'mean','parents':[r['id']]},**d})
 for c in candidates:
  if c['eligible'] and c['delta']>0:items.append(c)
 items.sort(key=lambda r:(r['delta'],min(r['fold_delta'])),reverse=True)
 # Keep six nominees and every required component for genuine retraining.
 return items[:6]

def champion_host(config_id):
 c=read(ROOT/'source/configurations.json')[config_id];recipe=read(ROOT/'evidence/champion.json')['components']['components'][c['view']]['recipe']
 return 'pc1' if recipe['model']=='lgb' else 'pc2'

def original_champion_record(r):
 return r['job']['family']=='champion' and r['host']==champion_host(r['job']['params']['config'])

def schedule_nominees(nominees,mapping,protocols):
 if not nominees:return {}
 requested={h:[] for h in ['pc1','pc2','pc3','node380']};needed={k for n in nominees for k in n['spec']['parents']};needed.discard('champion')
 for key in needed:
  r=mapping.get(key)
  if not r:continue
  for protocol in protocols:
   if protocol=='temporal' and r['job']['family']=='nested':continue # Nested temporal training is not implemented; never mislabel original CV.
   request={'id':ident([key,protocol])[:20],'family':r['job']['family'],'params':r['job']['params'],'protocol':protocol,'seed':4917 if protocol=='reseed' else 2026,'deadline':read(ROOT/'clock.json')['start']+({'confirm':11.5,'repeat':10.5,'reseed':10.5,'temporal':10.5}[protocol])*3600}
   requested[r['host']].append(request)
 # Champion matched evaluations use the actual five original component recipes.
 conf=read(ROOT/'source/configurations.json');champ_ids=[k for k,v in conf.items() if v['model']=='champion']
 for k in champ_ids:
  for protocol in protocols:requested[champion_host(k)].append({'id':ident(['champion',k,protocol])[:20],'family':'champion','params':{'config':k},'protocol':protocol,'seed':4917 if protocol=='reseed' else 2026,'deadline':read(ROOT/'clock.json')['start']+({'confirm':11.5,'repeat':10.5,'reseed':10.5,'temporal':10.5}[protocol])*3600})
 for host,requests in requested.items():tasks(host,requests)
 return requested

def final_scores(rows,nominees,mapping,confirmation=False):
 bank=Path(read(ROOT/'deployment.json')['hosts']['pc2']['bank']);y=np.load(bank/('confirmation_y.npy' if confirmation else 'development_y.npy'));fold=np.load(bank/'folds.npy');out=[];cfgs=read(ROOT/'source/configurations.json')
 def arrays(protocol):
  a={};champ=[]
  for r in rows:
   j=r['job']
   if j['protocol']!=protocol:continue
   if j['family']=='champion':
    if original_champion_record(r):champ.append(predictions(r))
   for key,original in mapping.items():
    if key=='champion':continue
    if r['host']==original['host'] and j['family']==original['job']['family'] and j['params']==original['job']['params']:a[key]=predictions(r)
  if len(champ)==5:a['champion']=np.sort(np.column_stack(champ),axis=1)[:,1:-1].mean(1)
  return a
 for n in nominees:
  scores=[];deltas=[];fold_deltas=list(n['fold_delta']);available=True
  protocols=['confirm'] if confirmation else ['reseed','repeat']
  for protocol in protocols:
   a=arrays(protocol)
   if 'champion' not in a or any(k not in a for k in n['spec']['parents']):available=False;continue
   p=combine(n['spec'],a);b=a['champion'];scores.append(ks(y,p));deltas.append(ks(y,p)-ks(y,b))
   if not confirmation:
    assignments=np.asarray(read(ROOT/'evidence/repeat.json')['folds']) if protocol=='repeat' else fold
    fold_deltas += [ks(y[assignments==f],p[assignments==f])-ks(y[assignments==f],b[assignments==f]) for f in range(3)]
   np.save(ROOT/'reports'/(n['id']+'_'+protocol+'.npy'),p);np.save(ROOT/'reports'/('champion_'+protocol+'.npy'),b)
  out.append({**n,'worst_matched_fold_delta':min(fold_deltas),'complete_comparisons':available and len(scores)==len(protocols),'additional_scores':scores,'additional_deltas':deltas,'mean_matched_delta':float(np.mean([n['delta']]+deltas)) if not confirmation else (deltas[0] if deltas else None)})
 out.sort(key=lambda r:(r['complete_comparisons'],r['mean_matched_delta'] if r['mean_matched_delta'] is not None else -1,r['worst_matched_fold_delta']),reverse=True)
 return out

def catalogue_counts(configurations,states,completed,equivalent=()):
 eligible=set(configurations)-set(equivalent);done=set(completed)&eligible;started=set();failed=set();active=set();proposed=set()
 for st in states.values():
  for j in st.get('jobs',{}).values():
   if j.get('protocol')!='screen' or j.get('family')=='nested':continue
   cid=j['params']['config'];proposed.add(cid)
   if j.get('started'):started.add(cid)
   if j['status']=='running':active.add(cid)
   if j['status'] in ['failed','invalid','unschedulable']:failed.add(cid)
 partial=(started-done)-failed
 return {'completed':len(done),'failed':len((failed-done)&eligible),'partial_or_running':len(partial&eligible),'unstarted':len(eligible-started-done-failed),'equivalent_excluded':len(set(equivalent)),'registered':len(configurations),'running':len((active-done)&eligible),'unstarted_ids':sorted(eligible-started-done-failed),'partial_or_running_ids':sorted(partial&eligible),'failed_ids':sorted((failed-done)&eligible),'registered_is_not_execution':True}

def report_all(states):
 rows=records();counts={h:{s:sum(j['status']==s for j in st.get('jobs',{}).values()) for s in ['complete','failed','interrupted','timeout','invalid','budget_skipped','unschedulable','pending','running','unstarted']} for h,st in states.items()}
 single=[{k:r[k] for k in ['id','host','evaluation']}|{'family':r['job']['family'],'params':r['job']['params'],'protocol':r['job']['protocol']} for r in rows];single.sort(key=lambda r:r['evaluation']['score'],reverse=True)

 controls=[r for r in rows if r['job']['family']=='champion' and r['job']['protocol']=='screen' and original_champion_record(r)]
 if len(controls)==5:
  reproduced=np.sort(np.column_stack([predictions(r) for r in controls]),axis=1)[:,1:-1].mean(1);reference=np.load(ROOT/'evidence/champion_oof.npz')['existing_feature_ensemble'];maximum=float(np.max(np.abs(reproduced-reference)));write(ROOT/'reports/champion_reproduction.json',{'complete':True,'maximum_probability_difference':maximum,'ks':ks(np.load(Path(read(ROOT/'deployment.json')['hosts']['pc2']['bank'])/'development_y.npy'),reproduced),'passes':maximum<1e-10})
  if maximum>=1e-10:write(ROOT/'operator_stop.json',{'time':time.time(),'reason':'Champion reproduction mismatch; fail closed before interpreting new scores'})
 write(ROOT/'reports/scoreboard_actuarial.json',[r for r in single if r['family'] in ['act_logit','act_cloglog','woe_scorecard','cred_scorecard','spline','ebm']])
 write(ROOT/'reports/scoreboard_nested.json',[r for r in single if r['family']=='nested'])
 configurations=read(ROOT/'source/configurations.json');effects=[];anchors={};unmatched=[]
 for r in rows:
  if r['job']['protocol']!='screen' or r['job']['family']=='nested':continue
  c=configurations[r['job']['params']['config']]
  if c.get('anchor') and c['view']=='v3':anchors[(c['model'],r['job']['family'])]=r
 bank=Path(read(ROOT/'deployment.json')['hosts']['pc2']['bank']);y=np.load(bank/'development_y.npy');fold=np.load(bank/'folds.npy')
 for r in rows:
  if r['job']['protocol']!='screen' or r['job']['family']=='nested':continue
  c=configurations[r['job']['params']['config']];baseline=anchors.get((c['model'],r['job']['family']))
  if c.get('anchor') and c['view']!='v3' and not baseline:unmatched.append({'id':r['id'],'model':c['model'],'execution_family':r['job']['family'],'reason':'No completed v3 anchor on the same execution backend; no feature-effect attribution'})
  if not c.get('anchor') or c['view']=='v3' or not baseline:continue
  bc=configurations[baseline['job']['params']['config']]
  if {k:v for k,v in c.items() if k!='view'}!={k:v for k,v in bc.items() if k!='view'}:continue
  effects.append({'treatment':r['id'],'control':baseline['id'],'view':c['view'],'model':c['model'],**delta(y,predictions(r),predictions(baseline),fold)})
 for effect in effects:effect['finding']='positive exploratory' if effect['delta']>0 and sum(d>0 for d in effect['fold_delta'])>=2 else ('negative exploratory' if effect['delta']<0 and sum(d<0 for d in effect['fold_delta'])>=2 else 'inconclusive')
 write(ROOT/'reports/feature_effects.json',effects)
 write(ROOT/'reports/unmatched_feature_contrasts.json',unmatched)
 resolved=read(ROOT/'evidence/resolved-deduplication.json') if (ROOT/'evidence/resolved-deduplication.json').exists() else {}
 classified=catalogue_counts(configurations,states,{r['job']['params']['config'] for r in rows if r['job']['protocol']=='screen' and r['job']['family']!='nested'},resolved.get('excluded_equivalent',[]))
 write(ROOT/'reports/catalogue_status.json',classified)
 write(ROOT/'reports/coverage.json',{'catalogue_status':{k:v for k,v in classified.items() if not k.endswith('_ids')},'registered_training':4096,'completed_screening_configurations':len({r['job']['params']['config'] for r in rows if r['job']['protocol']=='screen' and r['job']['family']!='nested'}),'fixed_ensemble_cap':4096,'nested_cap':12,'not_completed_is_not_a_negative_result':True})
 write(ROOT/'reports/scoreboard_individual.json',single);write(ROOT/'reports/summary.json',{'clock':read(ROOT/'clock.json'),'counts':counts,'verified_results':len(rows),'unique_screening_configurations':len({r['job']['params']['config'] for r in rows if r['job']['protocol']=='screen'}),'public_score_of_champion_user_reported':.56188,'no_new_submission':True,'gpu_allocation_seconds':sum((r['job'].get('finished',time.time())-r['job']['started'])*r['job']['resources'].get('gpus',0) for r in rows),'amd_allocation_seconds':sum(r['job'].get('finished',time.time())-r['job']['started'] for r in rows if r['job']['resources'].get('opencl')),'generated':time.time()})
 return rows


def terminal_uncertainty(results):
 bank=Path(read(ROOT/'deployment.json')['hosts']['pc2']['bank']);y=np.load(bank/'confirmation_y.npy');rng=np.random.default_rng(20261021);groups=[np.flatnonzero(y==c) for c in [0,1]];reference_path=ROOT/'reports/champion_confirm.npy'
 if not reference_path.exists():return
 reference=np.load(reference_path);valid=[]
 for result in results:
  path=ROOT/'reports'/(result['id']+'_confirm.npy')
  if not path.exists():continue
  p=np.load(path);differences=[]
  for _ in range(2000):
   ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);differences.append(ks(y[ix],p[ix])-ks(y[ix],reference[ix]))
  a=np.array(differences);result['paired_bootstrap_95ci']=np.quantile(a,[.025,.975]).tolist();result['conditional_bootstrap_p']=min(1.,2*min((sum(a<=0)+1)/2001,(sum(a>=0)+1)/2001));result['uncertainty_caveat']='Conditional on frozen selected predictions; historical selection reuse remains';valid.append(result)
 ordered=sorted(valid,key=lambda r:r['conditional_bootstrap_p']);previous=0
 for i,r in enumerate(ordered):previous=max(previous,min(1.,(len(ordered)-i)*r['conditional_bootstrap_p']));r['holm_adjusted_p']=previous

def main():
 clock=read(ROOT/'clock.json');deadline=clock['deadline'];node_pool=concurrent.futures.ThreadPoolExecutor(max_workers=1);node_collection=None;states={};phase=read(ROOT/'phase.json')['phase'] if (ROOT/'phase.json').exists() else 'search';last_collection=0;nominees=(read(ROOT/'reports/frozen_finalists.json')['nominees'] if (ROOT/'reports/frozen_finalists.json').exists() else (read(ROOT/'reports/nominees.json') if (ROOT/'reports/nominees.json').exists() else []));mapping=read(ROOT/'ensemble_sources.json') if (ROOT/'ensemble_sources.json').exists() else {};(ROOT/'reports').mkdir(exist_ok=True);write(ROOT/'supervisor_owner.json',{'pid':os.getpid(),'started':time.time()})
 while time.time()<deadline:
  if (ROOT/'operator_stop.json').exists():
   for h in ['pc1','pc2','pc3','node380']:
    try:
     if h=='pc1':rpc({'action':'stop'})
     else:
      path=read(ROOT/'deployment.json')['hosts'][h]['root']+'/campaign/stop.json';remote(h,'python3 -c '+shlex.quote('from pathlib import Path;p=Path('+repr(path)+');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(chr(123)+chr(125))'))
    except Exception:pass
   break
  now=time.time();elapsed=(now-clock['start'])/3600
  used=int(subprocess.check_output(['du','-sb',str(ROOT)],text=True).split()[0]) if now-last_collection>120 else 0
  if shutil.disk_usage(ROOT).free<5*1024**3 or used>280*1024**3:
   write(ROOT/'admission-paused.json',{'reason':'Archive headroom below 5GiB or campaign above 280GiB; reserve remaining 20GiB for owned active checkpoints','time':now})
   for h in ['pc1','pc2','pc3','node380']:
    try:
     if h=='pc1':rpc({'action':'pause_admission'})
     else:
      p=read(ROOT/'deployment.json')['hosts'][h]['root']+'/campaign/pause_admission.json';remote(h,'python3 -c '+shlex.quote('from pathlib import Path;Path('+repr(p)+').write_text(\'{}\')'))
    except Exception:pass
  if now-last_collection>120:
   for host in ['pc1','pc2','pc3','node380']:
    try:
     st=state(host);states[host]=st
     if st.get('status')!='unavailable':
      if host=='node380':
       if node_collection is None or node_collection.done():
        if node_collection is not None:
         try:node_collection.result()
         except Exception as e:
          with (ROOT/'logs/collection-errors.jsonl').open('a') as f:f.write(json.dumps({'host':host,'error':str(e)[-500:]})+'\n')
        node_collection=node_pool.submit(collect,host,st)
      else:collect(host,st)
    except Exception as e:
     with (ROOT/'logs/collection-errors.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'host':host,'error':str(e)[-1000:]})+'\n')
   last_collection=time.time();rows=report_all(states)
  if elapsed<8 and (ROOT/'extras/manifest.json').exists() and not (ROOT/'node-admitted.json').exists() and (not (ROOT/'node-attempt.json').exists() or now-read(ROOT/'node-attempt.json')['time']>=600):
   try:
    from node_join import maybe_join
    write(ROOT/'node-attempt.json',{'time':now})
    maybe_join(ROOT,clock)
   except Exception:pass
  if elapsed>=7.5 and not (ROOT/'nested_requested.json').exists():
   rs=records();configs=read(ROOT/'source/configurations.json');eligible=[r for r in rs if r['job']['protocol']=='screen' and r['job']['family'] in ['lgb','lgb_cpu','elastic','spline','rff'] and r['host']=='pc2'];eligible.sort(key=lambda r:r['evaluation']['score'],reverse=True);champids=[k for k,p in configs.items() if p['model']=='champion'];requests=[]
   for i in range(min(12,len(eligible)*3)):
    component=eligible[(i//3)%len(eligible)]['job']['params']['config'];parents=[champids[0],champids[1],component];requests.append({'id':'nested_'+str(i),'family':'nested','params':{'config':'nested_'+str(i),'components':parents},'protocol':'screen','seed':2026,'deadline':clock['start']+9*3600})
   tasks('pc2',requests);write(ROOT/'nested_requested.json',requests)
  if elapsed>=9 and phase=='search':
   candidates,mapping=ensembles(records());nominees=nominate(records(),candidates,mapping);write(ROOT/'reports/nominees.json',nominees);schedule_nominees(nominees,mapping,['reseed','repeat','temporal']);phase='stability';write(ROOT/'phase.json',{'phase':phase,'nominees':[n['id'] for n in nominees]})
  if elapsed>=10.5 and phase=='stability':
   ranking=final_scores(records(),nominees,mapping);write(ROOT/'reports/stability.json',ranking);nominees=[n for n in ranking if n['complete_comparisons'] and n['mean_matched_delta']>0][:3];write(ROOT/'reports/frozen_finalists.json',{'time':time.time(),'nominees':nominees,'primary':nominees[0]['id'] if nominees else None});schedule_nominees(nominees,mapping,['confirm']);phase='confirmation';write(ROOT/'phase.json',{'phase':phase})
  if elapsed>=11.5 and phase=='confirmation':
   # One terminal assessment; results never feed proposal generation.
   results=final_scores(records(),nominees,mapping,True);terminal_uncertainty(results);write(ROOT/'reports/confirmation.json',results);phase='report';write(ROOT/'phase.json',{'phase':phase})
  write(ROOT/'supervisor_heartbeat.json',{'time':time.time(),'phase':phase,'deadline':deadline});time.sleep(15)
 for host in ['pc1','pc2','pc3','node380']:
  try:
   st=state(host);states[host]=st
   if host!='node380':collect(host,st)
  except Exception:pass
 node_pool.shutdown(wait=True)
 rows=report_all(states)
 bank=Path(read(ROOT/'deployment.json')['hosts']['pc2']['bank']);meta_manifest=read(bank/'manifest.json');extra_manifest=read(ROOT/'extras/manifest.json');configurations=read(ROOT/'source/configurations.json');ranking=[]
 for r in sorted([r for r in rows if r['job']['protocol']=='screen'],key=lambda r:r['evaluation']['score'],reverse=True)[:20]:
  recipe=configurations.get(r['job']['params']['config'],r['job']['params']);ranking.append({'id':r['id'],'family':r['job']['family'],'recipe':recipe,'score':r['evaluation']['score'],'archive':r['folder'],'models_index':str(Path(r['folder'])/'models.json'),'reproduction':'Use control.py replay '+r['host']+' --job '+r['job']['job_id']+' --output NEW_DIRECTORY; explicit new training request required after handoff'})
 write(ROOT/'reports/a100_handoff.json',{'ranked_recipes':ranking,'frozen_finalists':read(ROOT/'reports/frozen_finalists.json') if (ROOT/'reports/frozen_finalists.json').exists() else None,'base_bank_id':meta_manifest['bank_id'],'new_feature_lineage':extra_manifest['info'],'all_original_feature_lineage':meta_manifest['info'],'selected_columns_and_transformers':'Each retained attempt contains fold preprocessing.json and serialized preprocessing.joblib; encoder/reducer state is retained with exact feature names','caveat':'Ranking is research evidence; transfer to a larger foundation model is unverified'})
 screen=[r for r in rows if r['job']['protocol']=='screen']
 if screen:
  best=max(screen,key=lambda r:r['evaluation']['score']);y=np.load(bank/'development_y.npy');p=predictions(best);b=np.load(ROOT/'evidence/champion_oof.npz')['existing_feature_ensemble'];metadata=__import__('pandas').read_parquet(bank/'metadata.parquet');dev=np.load(bank/'development_rows.npy');cohorts=metadata.decision_datetime.iloc[dev].astype(str).str[:7].to_numpy();cohort=[]
  for value in sorted(set(cohorts)):
   ix=cohorts==value
   if len(np.unique(y[ix]))<2:continue
   cohort.append({'decision_month':value,'rows':int(ix.sum()),'champion_ks':ks(y[ix],b[ix]),'candidate_ks':ks(y[ix],p[ix]),'champion_brier':float(np.mean((b[ix]-y[ix])**2)),'candidate_brier':float(np.mean((p[ix]-y[ix])**2))})
  write(ROOT/'reports/cohort_errors.json',{'candidate':best['id'],'cohorts':cohort,'uses_development_labels_only':True})
 lines=['# Twelve-hour exploration report','','Champion public score: user-reported 0.56188. No new submission generated.','','See separate individual, fixed, stability and confirmation JSON evidence.','',json.dumps({k:v for k,v in read(ROOT/'reports/catalogue_status.json').items() if not k.endswith('_ids')},indent=2),json.dumps(read(ROOT/'reports/summary.json')['counts'],indent=2),'','All scores are exploratory because historical labels and adaptively selected features were reused. Missing, failed, partial or unstarted work is not a negative quality result.']
 if not nominees:lines+=['','No candidate completed the declared improvement/stability gates. The champion is retained.']
 (ROOT/'reports/final_report.md').write_text('\n'.join(lines)+'\n');write(ROOT/'finished.json',{'time':time.time(),'state':'stopped' if (ROOT/'operator_stop.json').exists() else 'completed','deadline':deadline})

if __name__=='__main__':main()
