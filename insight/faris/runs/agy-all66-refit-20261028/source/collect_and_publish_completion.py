"""Collect the verified requested CSV and publish its reproducible terminal record.

No Kaggle upload, recipe selection, force push, or fitted-weight serialization.
A genuine error stops with durable evidence rather than an uncontrolled retry.
"""
from pathlib import Path
import csv,datetime,hashlib,json,subprocess,sys,time,tarfile,shutil
import numpy as np
R=Path(__file__).resolve().parents[1];P=R/'private';W=R.parents[2];REPO=W/'.history-publish-20261028';REMOTE='/mnt/home/faris/logika-antigravity-refit-20261028';RUN=REPO/'insight/faris/runs/agy-all66-refit-20261028'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def state(stage,**kwargs):
 v={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':stage,**kwargs};(R/'COMPLETION_STATE.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v),flush=True)
def command(args,*,timeout=180,capture=False):
 r=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 if r.returncode:raise RuntimeError({'command':args[:4],'exit_code':r.returncode,'diagnostic':r.stderr[-1500:]})
 return r.stdout
try:
 state('waiting_for_verified_remote_CSV');deadline=time.monotonic()+21000
 while True:
  code="from pathlib import Path;import json;p=Path('"+REMOTE+"/FINALIZE_STATE.json');print(p.read_text()if p.exists()else '{}')"
  import shlex
  raw=command(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','faris@100.68.29.126',shlex.join(['python3','-c',code])],timeout=40);v=json.loads(raw)
  if v.get('stage')=='CSV_COMPLETE_VERIFIED':break
  assert time.monotonic()<deadline,'FinalCSVwaitexceeded';time.sleep(45)
 verification=v['verification'];assert verification['status']=='passed'and verification['rows']==30000
 artifact=R/'submission_ALL66_exploratory_shift070_full_refit.csv'
 command(['scp','faris@100.68.29.126:'+verification['file'],str(artifact)],timeout=180);assert sha(artifact)==verification['sha256']
 with artifact.open(newline='')as stream:
  reader=csv.DictReader(stream);assert reader.fieldnames==['application_id','default_probability'];rows=list(reader)
 assert len(rows)==30000 and len({r['application_id']for r in rows})==30000
 predictions=np.array([float(r['default_probability'])for r in rows]);assert np.isfinite(predictions).all()and(predictions>=0).all()and(predictions<=1).all()
 target=W/artifact.name;shutil.copy2(artifact,target)
 state('collect_compact_provenance',CSV_sha256=sha(artifact))
 # Bundle only compact evidence, prediction outputs, sources and receipts.
 code="""from pathlib import Path
import tarfile,json,hashlib
r=Path(ROOT);out=r/'FINAL_COMPACT_EVIDENCE.tar.gz';paths=[]
for prefix in ['predictions','assembly','source']:
 for p in(r/prefix).rglob('*'):
  if p.is_file()and'__pycache__'not in p.parts and p.suffix in{'.py','.json','.npz','.txt','.csv'}:paths.append(p)
for p in r.glob('features/**/*.json'):paths.append(p)
for p in r.glob('packets/**/meta.json'):paths.append(p)
for p in r.glob('pilots/**/*.json'):paths.append(p)
for name in ['CONFIG.json','PROTOCOL.json','REFIT_POLICY.json','ENGINE_CONFIGS.json','ENGINE_ENVIRONMENT.json','A100_RESULTS_RECEIPT.json','FINALIZE_STATE.json','features/RAW_PARITY.json','features70/COMPLETE.json']:
 p=r/name
 if p.exists():paths.append(p)
with tarfile.open(out,'w:gz')as tar:
 for p in sorted(set(paths)):tar.add(p,arcname=str(p.relative_to(r)),recursive=False)
h=hashlib.sha256(out.read_bytes()).hexdigest();print(json.dumps({'file':str(out),'sha256':h,'bytes':out.stat().st_size}))
""".replace('ROOT',repr(REMOTE))
 receipt=json.loads(command(['ssh','-o','BatchMode=yes','faris@100.68.29.126',shlex.join(['python3','-c',code])],timeout=180));bundle=P/'FINAL_COMPACT_EVIDENCE.tar.gz';command(['scp','faris@100.68.29.126:'+receipt['file'],str(bundle)],timeout=180);assert sha(bundle)==receipt['sha256']
 evidence=P/'FINAL_COMPACT_EVIDENCE';evidence.mkdir(exist_ok=True)
 with tarfile.open(bundle)as tar:tar.extractall(evidence,filter='data')
 components=list((evidence/'predictions').rglob('COMPLETE.json'));assert len(components)==45
 for p in components:assert sha(p.with_name('prediction.npz'))==json.loads(p.read_text())['prediction_sha256']
 for fold in range(5):
  p=evidence/f'assembly/production/fold{fold}/complete.json';assert json.loads(p.read_text())['status']=='passed'
 result={'status':'VERIFIED_REQUESTED_EXPLORATORY_CSV_CREATED','CSV':{'local_path':str(artifact),'workspace_copy':str(target),'rows':30000,'unique_ids':30000,'sha256':sha(artifact),'sample_order_exact':True,'finite_bounded_predictions':True},'completed_component_fits':45,'production_folds':5,'retained':'Source,params,seeds,preprocessing,compactpredictionsandprovenance; nofittedweightsorKVcaches','original_development_replay_KS':0.5340125023110701,'development_score_caveat':'HistoricalALL66routingandlambda0.70selectedonreusedOOF; thisisnotanindependentgainestimate','new_full_refit_independent_KS':None,'local_target_054_attainment_claimed':False,'new_public_Kaggle_score':None,'public_incumbent':{'id':'V25','KS':0.56822},'A100_replay_gate':json.loads((R/'A100_REPLAY_GATE.json').read_text()),'runtime_stopped_after_verified_collection':True}
 (R/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
 # Wait for initial historical import push receipt before committing the outcome.
 archive=P.parents[1]/'github-history-20261028/private/ARCHIVE_REMOTE_RECEIPT.json'
 while not archive.exists():
  assert time.monotonic()<deadline,'Historicalarchivepushreceiptmissing';time.sleep(10)
 assert not command(['git','-C',str(REPO),'status','--porcelain']).strip(),'Sharedpublicationcheckoutchanged; preservemanualworkandstop'
 command(['git','-C',str(REPO),'fetch','origin','main'],timeout=180)
 command(['git','-C',str(REPO),'merge','--ff-only','origin/main'],timeout=180)
 sys.path.insert(0,str(REPO/'tools'));from publish_experiment_history import public_content
 for path in(R/'source').rglob('*'):
  if path.is_file()and path.suffix in{'.py','.txt','.json'}and'__pycache__'not in path.parts:
   content,_=public_content(path);assert content is not None;dest=RUN/'source'/path.relative_to(R/'source');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
 manifest={str(path.relative_to(RUN)):sha(path)for path in(RUN/'source').rglob('*')if path.is_file()}
 (RUN/'SOURCE_SNAPSHOT.json').write_text(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'Completedrequestedrefit','files_sha256':manifest},indent=2)+'\n')
 for prefix in ['predictions','features','packets','pilots']:
  for path in(evidence/prefix).rglob('*.json'):
   content,_=public_content(path)
   if content is None:continue
   dest=RUN/'evidence'/path.relative_to(evidence);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
 (RUN/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');(RUN/'CSV_VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
 for fold in range(5):
  path=evidence/f'assembly/production/fold{fold}/complete.json';dest=RUN/f'evidence/assembly_fold{fold}.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
 command(['python3',str(REPO/'tools/log_experiment.py'),'finish','--id','agy-all66-refit-20261028','--status','completed','--results',str(R/'RESULTS.json'),'--private-artifact',str(artifact),'--private-artifact',str(bundle),'--notes','RequestedexploratoryCSVverified; independent0.54/publicgainnotestablished.'],timeout=120)
 (RUN/'README.md').write_text('# ALL66 exploratory refit completed\n\nThe requested full rebuild produced a verified 30,000-row CSV in sample order.\nAll 45 Kumo component fits, five spline refits and five fold assemblies completed.\nThe compact predictions and case-level CSV remain private with hashes in RUN.json.\n\nThe historical development replay KS is 0.5340125023110701. Routing and the shift\nwere selected on reused OOF results; no independent 0.54 or new public gain is claimed.\nSee RESULTS.json, CSV_VERIFICATION.json, REFIT_POLICY.json and source/ for provenance.\n')
 scanner=W/'a100/faris/github-history-20261028/private/scanner/gitleaks';command([str(scanner),'dir',str(RUN),'--config',str(REPO/'.gitleaks.toml'),'--ignore-gitleaks-allow','--redact=100','--no-banner','--no-color'],timeout=180)
 command(['git','-C',str(REPO),'add',str(RUN)],timeout=120);command(['git','-C',str(REPO),'commit','-m','Record verified ALL66 exploratory refit CSV and completed component evidence'],timeout=180)
 command(['git','-C',str(REPO),'fetch','origin','main'],timeout=180);command(['git','-C',str(REPO),'rebase','origin/main'],timeout=180)
 command(['git','-C',str(REPO),'push','origin','HEAD:refs/heads/main'],timeout=600)
 head=command(['git','-C',str(REPO),'rev-parse','HEAD']).strip();remote=command(['git','-C',str(REPO),'ls-remote','origin','refs/heads/main']).split()[0];assert head==remote
 state('COMPLETE_CSV_COLLECTED_AND_EXPERIMENT_LOG_PUSHED',CSV=str(artifact),workspace_copy=str(target),CSV_sha256=sha(artifact),remote_commit=remote,model_improvement_claimed=False)
except Exception as error:
 state('FAILED_PRESERVE_EVIDENCE_NO_AUTOMATIC_RETRY',error=str(error));raise
