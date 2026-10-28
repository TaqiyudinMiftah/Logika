"""Bounded archived-expert fusion. No original campaign is modified; no weights saved."""
from pathlib import Path
import argparse, concurrent.futures, datetime, fcntl, hashlib, json, os, subprocess, sys, time, traceback
os.environ.setdefault('OMP_NUM_THREADS','2'); os.environ.setdefault('OPENBLAS_NUM_THREADS','2'); os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve, roc_auc_score, log_loss
R=Path(__file__).resolve().parents[1]
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False,default=str)+'\n');t.replace(p)
def save(p,**v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with t.open('wb') as f: np.savez_compressed(f,**v)
 t.replace(p)
def ids(v): return hashlib.sha256(np.asarray(v,dtype='<i8').tobytes()).hexdigest()
def check():
 c=load(R/'CLOCK.json');s=load(R/'SOURCE_SEAL.json');assert all(sha(R/p)==v for p,v in s.items());assert c['seal_sha256']==sha(R/'SOURCE_SEAL.json');assert c['plan_sha256']==sha(R/'PLAN.json');assert c['input_manifest_sha256']==sha(R/'INPUT_MANIFEST.json');assert all(sha(R/'evaluation_inputs'/n)==v for n,v in c['evaluation_input_sha256'].items() if (R/'evaluation_inputs'/n).exists());return c

def metrics(y,p):
 y=np.asarray(y,int);p=np.asarray(p,float);assert y.shape==p.shape and np.isfinite(p).all() and len(np.unique(y))==2 and np.all((p>=0)&(p<=1))
 f,t,_=roc_curve(y,p,drop_intermediate=False);k=float(np.max(abs(t-f)));assert abs(k-ks_2samp(p[y==1],p[y==0],method='asymp').statistic)<1e-12
 return dict(KS=k,AUC=float(roc_auc_score(y,p)),logloss=float(log_loss(y,p)),rows=len(y))
def bank_group(name):
 if name.startswith('kumo_'): return 'kumo'
 if name.startswith(('lgb_','cpu_capacity')): return 'lightgbm'
 if name.startswith(('cat_','cpu_order')): return 'catboost'
 if name.startswith('xgb_'): return 'xgboost'
 if name.startswith(('glm_','cpu_actuarial')): return 'glm'
 if name.startswith('resnet_'): return 'resnet'
 return 'tabm_mlp'

def prepare():
 """Read query predictions and permitted held-meta labels; outer target stays evaluator-only."""
 plan=load(R/'PLAN.json');old=Path(plan['archive_root']);board=load(old/'SCOREBOARD.json');designs=sorted(board['complete_designs']);assert len(designs)==104
 assert sha(old/'SCOREBOARD.json')==plan['archive_scoreboard_sha256'];assert sha(old/'ARTIFACT_CONSISTENCY_AUDIT.json')==plan['archive_audit_sha256']
 retries=load(old/'APPROVED_RETRIES.json');manifest={};names=['anchor']+designs
 for fold in range(5):
  rolespath=old/f'packets/fold{fold}/roles.npz';anchorpath=old/f'packets/fold{fold}/anchor.npz';roles=np.load(rolespath);anchor=np.load(anchorpath)
  context,meta,outer=[np.asarray(roles[k],int) for k in ('context','correction','validation')]
  assert tuple(map(len,(context,meta,outer)))==(50000,6000,14000);assert len(np.unique(np.r_[context,meta,outer]))==70000;assert not any(k in roles.files for k in ['y_outer','y_validation'])
  y=np.asarray(roles['y_correction'],int);assignment=np.asarray(anchor['inner_fold'],int);assert y.shape==assignment.shape==(6000,);assert set(assignment)=={0,1,2}
  for seed in plan['seeds']:
   columns=[np.r_[anchor['correction_oof'],anchor['validation']]];inputs={str(rolespath):sha(rolespath),str(anchorpath):sha(anchorpath)}
   for design in designs:
    task=f'{design}__s{seed}__f{fold}';a=int(retries.get(task,1));d=old/f'runs/{task}/attempt{a}';res=load(d/'RESULT.json');start=load(d/'START.json');p=np.load(d/'prediction.npz')
    assert res['status']=='complete' and sha(d/'prediction.npz')==res['prediction_sha256'] and sha(d/'START.json')==res['start_sha256'];assert start['task']==dict(design=design,seed=seed,fold=fold);assert start['outer_labels_loaded'] is False and start['seed']==seed+fold;assert start['correction_labels_used_for_training'] is False and start['query_labels_used_for_early_stopping'] is False;assert start['context_rows_sha256']==ids(context) and start['prediction_rows_sha256']==ids(np.r_[meta,outer]);assert (start['context_rows'],start['correction_rows'],start['validation_rows'])==(50000,6000,14000);assert start['source_sha256']==sha(old/'source/worker.py') and start['roles_sha256']==sha(rolespath);assert all(start[k]==sha(old/n) for k,n in [('source_seal_sha256','SOURCE_SEAL.json'),('config_sha256','CONFIG.json'),('contract_sha256','VALIDATION_CONTRACT.json'),('clock_sha256','CLOCK.json')])
    assert np.array_equal(p['rows'],np.r_[meta,outer]);assert p['raw'].shape==(20000,) and np.isfinite(p['raw']).all();columns.append(p['raw']);inputs[str(d/'RESULT.json')]=sha(d/'RESULT.json')
   x=np.asarray(columns,float).T;assert x.shape==(20000,105) and np.isfinite(x).all() and np.all((x>=0)&(x<=1))
   dest=R/f'packets/s{seed}_f{fold}.npz';save(dest,probability=x,y_meta=y,assignment=assignment,meta_rows=meta,outer_rows=outer)
   manifest[dest.name]=dict(sha256=sha(dest),inputs=inputs,context_rows_sha256=ids(context),meta_rows_sha256=ids(meta),outer_rows_sha256=ids(outer),context_count=50000,meta_count=6000,outer_count=14000)
 dump(R/'INPUT_MANIFEST.json',dict(UTC=utc(),packets=manifest,names=names,groups=['anchor']+[bank_group(d) for d in designs],outer_labels_in_worker_packets=False,base_fit_count=0))
 # Evaluator-only historical development targets and exact old references.
 for filename in ['baseline.npz','target.npy']:
  src=old/'evaluation_inputs'/filename;dest=R/'evaluation_inputs'/filename;dest.parent.mkdir(exist_ok=True);dest.write_bytes(src.read_bytes());assert sha(dest)==sha(src)
 selected=old/'evaluation/cohorts/kumo_large_global99_e2__stack/prediction.npz';dest=R/'evaluation_inputs/selected_previous.npz';dest.write_bytes(selected.read_bytes())
 dump(R/'INPUTS_READY.json',dict(UTC=utc(),input_manifest_sha256=sha(R/'INPUT_MANIFEST.json'),outer_targets_evaluator_only=True,no_new_data=True))

def gate():
 check();m=load(R/'INPUT_MANIFEST.json');y=np.load(R/'evaluation_inputs/target.npy');b=np.load(R/'evaluation_inputs/baseline.npz');p=b['prediction'];folds=b['fold'];old=np.load(R/'evaluation_inputs/selected_previous.npz')['prediction']
 assert y.shape==p.shape==folds.shape==(70000,);assert abs(metrics(y,p)['KS']-.5306361042400267)<1e-12;assert abs(metrics(y,old)['KS']-.5312029153339166)<1e-12
 assert metrics(np.array([0,0,1,1]),np.array([.1,.2,.8,.9]))['KS']==1.;assert metrics(np.array([0,1,0,1]),np.array([.5,.5,.5,.5]))['KS']==0.
 supports=[]
 for name,info in m['packets'].items():
  a=np.load(R/'packets'/name);assert sha(R/'packets'/name)==info['sha256'];mr,ou=a['meta_rows'],a['outer_rows'];assert not np.intersect1d(mr,ou).size;assert len(np.unique(np.r_[mr,ou]))==20000;assert np.array_equal(y[mr],a['y_meta']);assert a['probability'].shape==(20000,105)
  for block in range(3):
   z=a['y_meta'][a['assignment']==block];assert min(np.bincount(z,minlength=2))>=50
  supports.append(dict(packet=name,positives=int(a['y_meta'].sum()),rows=6000))
 dump(R/'GATE.json',dict(status='PASS',UTC=utc(),checks=['known_metric_cases','independent_scipy_metric','exact_anchor_replay','exact_previous_winner_replay','packet_hashes','context_meta_outer_disjointness','meta_label_alignment','three_block_class_support','historical_source_audit_references','worker_packets_no_outer_targets'],baseline=metrics(y,p),prior_selected=metrics(y,old),supports=supports,independent_audit_available=False,limitation='All70000rows adaptive reused development; baseline replay is an implementation gate, not independent confirmation.',source_seal_sha256=sha(R/'SOURCE_SEAL.json')))

def features(p,names,groups,spec,train):
 z=logit(np.clip(p,1e-6,1-1e-6));anchor=z[:,0]
 if spec['bank']=='all104': x=z[:,1:]
 elif spec['bank']=='kumo': x=z[:,np.array([g=='kumo' for g in groups])]
 else:
  x=np.column_stack([np.mean(z[:,np.array([v==g for v in groups])],axis=1) for g in sorted(set(groups)-{'anchor'})])
 # Every channel is a residual from matched anchor, so zero coefficients reproduce it.
 x=x-anchor[:,None]
 if spec['conditional']:
  knots=np.quantile(anchor[train],[.25,.5,.75]);scale=max(float(np.std(anchor[train])),.1)
  bases=np.column_stack([np.ones(len(anchor))]+[np.clip((anchor-k)/scale,-2,2) for k in knots]);x=np.concatenate([x*bases[:,i,None] for i in range(4)],axis=1)
 mean=x[train].mean(axis=0);scale=np.maximum(x[train].std(axis=0),.05);return (x-mean)/scale,anchor,dict(columns=x.shape[1],preprocessing_train_rows=len(train),normalization='inner-meta training rows only',conditional_knots=spec['conditional'])

def fit_predict(packet,spec,train,query,seed,deadline):
 m=load(R/'INPUT_MANIFEST.json');p=packet['probability'];x,anchor,audit=features(p,m['names'],m['groups'],spec,train);y=packet['y_meta'][train].astype(float);xt=x[train];at=anchor[train];penalty=spec['penalty'];dim=x.shape[1]
 if spec['objective']=='pairwise':
  rng=np.random.default_rng(seed);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);ii=rng.choice(pos,20000,replace=True);jj=rng.choice(neg,20000,replace=True);dx=xt[ii]-xt[jj];da=at[ii]-at[jj]
  def objective(w):
   assert time.time()<deadline-20,'Model deadline reached';s=da+dx@w;e=expit(s)-1;return float(np.mean(np.logaddexp(0,-s))+.5*penalty*(w@w)),dx.T@e/len(e)+penalty*w
 else:
  def objective(w):
   assert time.time()<deadline-20,'Model deadline reached';s=at+xt@w;e=expit(s)-y;return float(np.mean(np.logaddexp(0,s)-y*s)+.5*penalty*(w@w)),xt.T@e/len(e)+penalty*w
 bounds=[(0,2)]*dim if spec['objective']=='nonnegative' else [(-2,2)]*dim
 fit=minimize(objective,np.zeros(dim),jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=160,ftol=1e-9,gtol=1e-6));assert np.isfinite(fit.x).all();assert fit.success or fit.nit==160,str(fit.message)
 # Partial optimizer convergence is explicit; it cannot silently count as success.
 assert fit.success,dict(message=str(fit.message),iterations=int(fit.nit))
 audit.update(iterations=int(fit.nit),objective=float(fit.fun),converged=bool(fit.success),learned_coefficients_saved=False,training_rows_sha256=ids(packet['meta_rows'][train]),prediction_rows=len(query),objective_name=spec['objective'])
 return expit(anchor[query]+x[query]@fit.x),audit

def worker(task_id):
 c=check();assert load(R/'GATE.json')['status']=='PASS';plan=load(R/'PLAN.json');task=plan['tasks'][task_id];dest=R/'runs'/task_id;assert not dest.exists(),'Existing attempts are never overwritten';dest.mkdir(parents=True);t=time.time();spec=plan['designs'][task['design']]
 try:
  assert t<c['admission_end_epoch'],'Admissions closed';packetpath=R/f"packets/s{task['seed']}_f{task['fold']}.npz";assert sha(packetpath)==load(R/'INPUT_MANIFEST.json')['packets'][packetpath.name]['sha256'];packet=np.load(packetpath);start=dict(UTC=utc(),task=task,spec=spec,source_sha256=sha(__file__),seal_sha256=sha(R/'SOURCE_SEAL.json'),packet_sha256=sha(R/f"packets/s{task['seed']}_f{task['fold']}.npz"),outer_labels_loaded=False,weights_saved=False,meta_training_rows=6000,seed_effect='matching archived base-model seed and pairwise pair sampling')
  dump(dest/'START.json',start);inner=np.full(6000,np.nan);audits=[]
  for block in range(3):
   tr=np.flatnonzero(packet['assignment']!=block);q=np.flatnonzero(packet['assignment']==block);v,a=fit_predict(packet,spec,tr,q,task['seed']+block+task['fold'],c['model_end_epoch']);inner[q]=v;audits.append(dict(block=block,**a))
  assert np.isfinite(inner).all();ym=packet['y_meta'];am=packet['probability'][:6000,0];weights=[0,.025,.05,.1,.2,.4,.6,1.];scores=[metrics(ym,(1-w)*am+w*inner)['KS'] for w in weights];weight=weights[int(np.argmax(scores))]
  q=np.arange(6000,20000);pred,fullaudit=fit_predict(packet,spec,np.arange(6000),q,task['seed']+task['fold'],c['model_end_epoch']);anchor=packet['probability'][6000:,0];result=(1-weight)*anchor+weight*pred
  save(dest/'prediction.npz',rows=packet['outer_rows'],prediction=result,standalone=pred,inner_meta=inner)
  dump(dest/'FROZEN.json',dict(UTC=utc(),task=task,spec=spec,start_sha256=sha(dest/'START.json'),prediction_sha256=sha(dest/'prediction.npz'),weight=weight,weight_training_KS=scores,inner_audits=audits,full_audit=fullaudit,outer_labels_opened=False,source_sha256=sha(__file__),weights_saved=False))
  dump(dest/'RESULT.json',dict(status='complete',UTC=utc(),task=task,wall_seconds=time.time()-t,prediction_sha256=sha(dest/'prediction.npz'),frozen_sha256=sha(dest/'FROZEN.json'),weights_saved=False,outer_labels_loaded=False))
 except Exception:
  dump(dest/'FAILURE.json',dict(status='failed',UTC=utc(),task=task,wall_seconds=time.time()-t,traceback=traceback.format_exc(),evidence_preserved=True));raise

def queue(folds):
 c=check();plan=load(R/'PLAN.json');out=R/f"QUEUE_{''.join(map(str,folds))}_FINAL.json";pending=[k for k,t in plan['tasks'].items() if t['fold'] in folds];running={};done=[]
 while pending or running:
  while pending and len(running)<2 and time.time()<c['admission_end_epoch']:
   k=pending.pop(0)
   if (R/'runs'/k).exists(): continue
   log=R/'logs'/f'{k}.log';log.parent.mkdir(exist_ok=True);f=log.open('w');p=subprocess.Popen([sys.executable,__file__,'worker','--task',k],stdout=f,stderr=subprocess.STDOUT);f.close();running[k]=p
  for k,p in list(running.items()):
   rc=p.poll()
   if rc is not None:done.append(dict(task=k,returncode=rc));del running[k]
  dump(R/f"QUEUE_{''.join(map(str,folds))}_STATE.json",dict(UTC=utc(),running={k:p.pid for k,p in running.items()},pending=len(pending),finished=len(done)))
  if time.time()>=c['model_end_epoch']:
   for p in running.values():p.terminate()
   break
  if time.time()>=c['admission_end_epoch'] and not running:break
  time.sleep(2)
 dump(out,dict(UTC=utc(),status='queue_terminal',finished=done,pending=pending,interrupted=list(running),original_clock_sha256=sha(R/'CLOCK.json')))

def evaluate(final=False):
 c=check();plan=load(R/'PLAN.json');baseline=np.load(R/'evaluation_inputs/baseline.npz');b=baseline['prediction'];folds=baseline['fold'];old=np.load(R/'evaluation_inputs/selected_previous.npz')['prediction'];ready={};incomplete={}
 # Freeze all complete candidates before reading any outer targets for this evaluation.
 for design in plan['designs']:
  predictions=[];inputs={};missing=[]
  for seed in plan['seeds']:
   p=np.full(70000,np.nan);coverage=np.zeros(70000,int)
   for fold in range(5):
    name=f'{design}__s{seed}__f{fold}';d=R/'runs'/name
    if not (d/'RESULT.json').exists(): missing.append(name);continue
    result=load(d/'RESULT.json');assert sha(d/'prediction.npz')==result['prediction_sha256'];assert sha(d/'FROZEN.json')==result['frozen_sha256'];a=np.load(d/'prediction.npz');rows=a['rows'];assert np.array_equal(np.sort(rows),np.flatnonzero(folds==fold));p[rows]=a['prediction'];coverage[rows]+=1;inputs[name]=sha(d/'RESULT.json')
   if not missing:assert np.all(coverage==1) and np.isfinite(p).all();predictions.append(p)
  if missing:incomplete[design]=missing;continue
  assert len(predictions)==2;mean=np.mean(predictions,axis=0);d=R/'evaluation'/design
  if not (d/'FROZEN.json').exists():
   save(d/'prediction.npz',prediction=mean,seeds=np.asarray(predictions),rows=np.arange(70000));dump(d/'FROZEN.json',dict(UTC=utc(),design=design,inputs=inputs,prediction_sha256=sha(d/'prediction.npz'),outer_labels_opened_for_this_score=False))
  else:assert load(d/'FROZEN.json')['inputs']==inputs and np.array_equal(np.load(d/'prediction.npz')['prediction'],mean)
  ready[design]=(mean,predictions)
 y=np.load(R/'evaluation_inputs/target.npy');reference=metrics(y,b);previous=metrics(y,old);board={}
 for design,(p,ps) in ready.items():
  score=metrics(y,p);board[design]=dict(metrics=score,delta_anchor=score['KS']-reference['KS'],delta_previous=score['KS']-previous['KS'],seed_delta_previous=[metrics(y,q)['KS']-previous['KS'] for q in ps],fold_delta_previous=[metrics(y[folds==f],p[folds==f])['KS']-metrics(y[folds==f],old[folds==f])['KS'] for f in range(5)],prediction_sha256=sha(R/'evaluation'/design/'prediction.npz'))
 selected=min(board,key=lambda k:(-board[k]['metrics']['KS'],k)) if board else None;dump(R/'SCOREBOARD.json',dict(UTC=utc(),complete_designs=len(board),planned_designs=len(plan['designs']),results=board,baseline=reference,previous=previous,best=selected,incomplete=incomplete,independent_confirmation=False))
 if final:
  selection_path=R/'FINAL_SELECTION.json'
  if selection_path.exists():assert load(selection_path)['selected']==selected and load(selection_path)['scoreboard_sha256']==sha(R/'SCOREBOARD.json'),'Final selection is immutable'
  else:dump(selection_path,dict(UTC=utc(),selected=selected,scoreboard_sha256=sha(R/'SCOREBOARD.json'),rule='largest complete two-seed five-fold developmentKS, lexicographic ties',selection_frozen_before_interval=True))
  if selected:
   p=ready[selected][0];rng=np.random.default_rng(20261029);strata=[np.flatnonzero((folds==f)&(y==v)) for f in range(5) for v in [0,1]];orders=[np.argsort(a,kind='stable') for a in [p,old]];ends=[np.flatnonzero(np.r_[a[o][1:]!=a[o][:-1],True]) for a,o in zip([p,old],orders)];delta=[]
   for i in range(2000):
    assert time.time()<c['deadline_epoch']-30
    sample=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata]);counts=np.bincount(sample,minlength=70000);q=counts*(y/y.sum()-(1-y)/(1-y).sum());k=[np.max(abs(np.cumsum(q[o])[e])) for o,e in zip(orders,ends)];delta.append(float(k[0]-k[1]))
   ci=np.quantile(delta,[.025,.975]).tolist();a=board[selected];support=a['delta_previous']>=.0005 and ci[0]>0 and sum(v>0 for v in a['fold_delta_previous'])>=4 and all(v>0 for v in a['seed_delta_previous']);dump(R/'FINAL_EVALUATION.json',dict(status='complete_development_assessment',UTC=utc(),selected=selected,result=a,conditional_delta95_vs_previous=ci,conditional_replicates=2000,supported_development_gain=support,point_target054=a['metrics']['KS']>=.54,independent_confirmation=False,limitations='All70k reused adaptive development; conditional interval excludes fitting and selection uncertainty.',no_refit=True,no_submission=True,weights_saved=False,final_selection_sha256=sha(R/'FINAL_SELECTION.json')))
  else:dump(R/'FINAL_EVALUATION.json',dict(status='no_complete_cohort',UTC=utc(),independent_confirmation=False))
 return len(board)

def audit_terminal():
 c=check();plan=load(R/'PLAN.json');inputs=load(R/'INPUT_MANIFEST.json');passed=[];failures=[]
 for task_id,task in plan['tasks'].items():
  d=R/'runs'/task_id
  if (d/'RESULT.json').exists():
   result=load(d/'RESULT.json');start=load(d/'START.json');frozen=load(d/'FROZEN.json');a=np.load(d/'prediction.npz');packet=np.load(R/f"packets/s{task['seed']}_f{task['fold']}.npz")
   assert start['task']==task and start['outer_labels_loaded'] is False and start['meta_training_rows']==6000;assert sha(d/'START.json')==frozen['start_sha256'] and sha(d/'FROZEN.json')==result['frozen_sha256'] and sha(d/'prediction.npz')==result['prediction_sha256'];assert np.array_equal(a['rows'],packet['outer_rows']) and a['prediction'].shape==(14000,) and np.isfinite(a['prediction']).all();assert frozen['outer_labels_opened'] is False
   for block,audit in enumerate(frozen['inner_audits']):assert audit['training_rows_sha256']==ids(packet['meta_rows'][packet['assignment']!=block]) and audit['converged'] is True
   assert frozen['full_audit']['training_rows_sha256']==ids(packet['meta_rows']);passed.append(dict(task=task_id,result_sha256=sha(d/'RESULT.json'),prediction_sha256=sha(d/'prediction.npz')))
  elif (d/'FAILURE.json').exists():failures.append(dict(task=task_id,sha256=sha(d/'FAILURE.json')))
 board=load(R/'SCOREBOARD.json');y=np.load(R/'evaluation_inputs/target.npy');reference=np.load(R/'evaluation_inputs/selected_previous.npz')['prediction'];replayed={}
 for design,result in board['results'].items():
  a=np.load(R/'evaluation'/design/'prediction.npz');score=metrics(y,a['prediction']);assert abs(score['KS']-result['metrics']['KS'])<1e-12 and abs(score['KS']-metrics(y,reference)['KS']-result['delta_previous'])<1e-12;assert np.array_equal(a['prediction'],a['seeds'].mean(axis=0));replayed[design]=score['KS']
 final=load(R/'FINAL_EVALUATION.json');selected=load(R/'FINAL_SELECTION.json');assert selected['selected']==board['best'];assert selected['scoreboard_sha256']==sha(R/'SCOREBOARD.json');assert all(not list(R.glob(pattern)) for pattern in ['**/*.pt','**/*.pth','**/*.pkl','**/*.joblib','**/*.safetensors'])
 dump(R/'ARTIFACT_CONSISTENCY_AUDIT.json',dict(status='PASS',UTC=utc(),completed_fits=len(passed),failure_records=len(failures),passed=passed,failures=failures,recomputed_cohorts=replayed,selected=selected['selected'],final_evaluation_sha256=sha(R/'FINAL_EVALUATION.json'),clock_sha256=sha(R/'CLOCK.json'),source_seal_sha256=sha(R/'SOURCE_SEAL.json'),independent_statistical_confirmation=False,not_independent='Arithmetic/provenance audit only; all70krowsreuseddevelopment',no_weights=True))

def launch(resume=False):
 lock=(R/'coordinator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if not (R/'CLOCK.json').exists():
  assert not resume,'No original clock';assert (R/'PREPUBLICATION.json').exists(),'Root must publish preregistration before fitting';receipt=load(R/'PREPUBLICATION.json');assert len(receipt['remote_commit'])==40 and receipt['plan_sha256']==sha(R/'PLAN.json') and receipt['source_seal_sha256']==sha(R/'SOURCE_SEAL.json');prepare();start=time.time();dump(R/'CLOCK.json',dict(started_epoch=start,started_UTC=utc(),admission_end_epoch=start+3.25*3600,model_end_epoch=start+(3+40/60)*3600,deadline_epoch=start+4*3600,hours=4,plan_sha256=sha(R/'PLAN.json'),seal_sha256=sha(R/'SOURCE_SEAL.json'),clock_starts_before_first_new_fit=True,input_manifest_sha256=sha(R/'INPUT_MANIFEST.json'),evaluation_input_sha256={n:sha(R/'evaluation_inputs'/n) for n in ['baseline.npz','target.npy','selected_previous.npz']}))
 c=check();assert time.time()<c['deadline_epoch'];
 if (R/'COMPLETION.json').exists():return
 gate()
 if not (R/'RUNTIME_GATE.json').exists():
  # Two predefined ordinary fits, reused in queue, qualify CPU execution before broad admission.
  times=[]
  for task in load(R/'PLAN.json')['pilot_tasks']:
   d=R/'runs'/task
   if not (d/'RESULT.json').exists():worker(task)
   times.append(load(d/'RESULT.json')['wall_seconds'])
  passed=max(times)<90 and time.time()+max(times)*len(load(R/'PLAN.json')['tasks'])/4*1.5<c['model_end_epoch']-300
  dump(R/'RUNTIME_GATE.json',dict(status='PASS' if passed else 'FAIL_RESOURCE_FEASIBILITY',UTC=utc(),pilot_seconds=times,pilot_bound_seconds=90,conservative_assumption='maxpilot * total_tasks /4slots *1.5',no_parameter_changes=True))
 if load(R/'RUNTIME_GATE.json')['status']!='PASS':evaluate(True);audit_terminal();dump(R/'COMPLETION.json',dict(UTC=utc(),status='runtime_admission_failed',weights_saved=False));return
 # Transfer only source/metadata, private packets and evidence; old source remains read-only.
 remote='faris@gpu-labkc.ub.ac.id';root=str(R);rsh=['ssh','-p','2003',remote]
 subprocess.run(rsh+['mkdir','-p',root],check=True)
 for name in ['source','packets','PLAN.json','CLOCK.json','SOURCE_SEAL.json','INPUT_MANIFEST.json','GATE.json']:
  subprocess.run(['rsync','-a','-e','ssh -p2003',str(R/name),f'{remote}:{root}/'],check=True)
 # Equal small tasks: folds0-2node2001, folds3-4node2003; CPU cap is a service-level root launch policy.
 unit='logika-fourhour-worker-20261029';py='/mnt/home/faris/logika-sixhour-broad-20261028/env311/bin/python'
 command=f"systemd-run --user --unit={unit} --property=MemoryMax=12G --property=CPUQuota=400% --property=RuntimeMaxSec={max(60,int(c['model_end_epoch']-time.time()))} {py} {root}/source/campaign.py queue --folds 3,4"
 subprocess.run(rsh+[command],check=True)
 local=subprocess.Popen([sys.executable,__file__,'queue','--folds','0,1,2'])
 while time.time()<c['model_end_epoch']-10:
  subprocess.run(['rsync','-a','--ignore-existing','-e','ssh -p2003',f'{remote}:{root}/runs/',str(R/'runs')+'/'],check=False)
  subprocess.run(['rsync','-a','-e','ssh -p2003',f'{remote}:{root}/QUEUE_34_FINAL.json',str(R)+'/'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
  evaluate(False);dump(R/'LIVE_STATE.json',dict(UTC=utc(),local_queue_pid=local.pid,local_returncode=local.poll(),completed_fits=len(list((R/'runs').glob('*/RESULT.json'))),failed_fits=len(list((R/'runs').glob('*/FAILURE.json'))),clock_sha256=sha(R/'CLOCK.json')))
  if local.poll() is not None and (R/'QUEUE_34_FINAL.json').exists():break
  time.sleep(20)
 if local.poll() is None:local.terminate();local.wait(timeout=30)
 subprocess.run(['rsync','-a','-e','ssh -p2003',f'{remote}:{root}/runs/',str(R/'runs')+'/'],check=True)
 evaluate(True);audit_terminal();dump(R/'COMPLETION.json',dict(UTC=utc(),status='completed_or_budget_terminal',completed_fits=len(list((R/'runs').glob('*/RESULT.json'))),failed_fits=len(list((R/'runs').glob('*/FAILURE.json'))),original_deadline_epoch=c['deadline_epoch'],no_refit=True,no_submission=True,weights_saved=False))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','gate','worker','queue','evaluate','launch','resume']);a.add_argument('--task');a.add_argument('--folds',default='0,1,2,3,4');a.add_argument('--final',action='store_true');v=a.parse_args()
 if v.command=='worker':worker(v.task)
 elif v.command=='queue':queue([int(x) for x in v.folds.split(',')])
 elif v.command=='evaluate':evaluate(v.final)
 elif v.command in ['launch','resume']:launch(v.command=='resume')
 else:globals()[v.command]()
