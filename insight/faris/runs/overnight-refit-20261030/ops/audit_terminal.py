"""Read-only terminal provenance/replay; never fit, select, downgrade or extend work."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='2'
from pathlib import Path
import argparse,hashlib,importlib.util,json,subprocess,sys,time,traceback
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.load(open(p))
def dump(name,v):
 p=R/name;p.parent.mkdir(exist_ok=True,parents=True);q=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');q.write_text(json.dumps(v,indent=2,default=str)+'\n');q.replace(p)
def pending():
 plan=load(R/'UNION_PLAN.json');ready=load(R/'READY_CANDIDATE.json');wait=[]
 for n in plan['component_studies']:
  p=R/n
  if not(p/'COMPLETION.json').exists():wait.append(n);continue
  e=load(p/'FINAL_EVALUATION.json');q=e.get('primary_result')or{};score=(q.get('metrics')or{}).get('KS')
  if score is not None and score>ready['local_KS']:
   d=R/'selected_fullfits'/n
   if not(d/'COMPLETE.json').exists()and not(d/'FAILURE_PRIVATE.log').exists():wait.append('fresh_refit:'+n)
 for n in plan.get('meta_composition_studies',[]):
  if not(R/n/'COMPLETION.json').exists()and not(R/n/'FAILURE.json').exists():wait.append('meta:'+n)
 return wait
def audit(terminal=False):
 assert sha(R/'CLOCK.json')=='576f1da25b5636c93a8d93d36c3f993cdfdf300cdade61563799362e195507d0'
 ready=load(R/'READY_CANDIDATE.json');assert ready['campaign']=='overnight-refit-20261030'and ready['fresh_training']
 csv=Path(ready['path']);before=sha(csv);assert before==ready['sha256'];a=load(Path(ready['audit_path']));assert a['status']=='PASS'and a['sha256']==before and a['fresh_training']
 saved=pd.read_csv(csv);assert list(saved)==['application_id','default_probability']and len(saved)==30000 and saved.application_id.nunique()==30000
 sample=pd.read_csv('/mnt/home/faris/logika-faris-exploration/exploration-12h-20261017/raw/sample_submission(3).csv');assert saved.application_id.astype(str).tolist()==sample.application_id.astype(str).tolist()
 assert np.isfinite(saved.default_probability).all()and saved.default_probability.between(0,1).all()
 # Rerun frozen meta recipe against saved fresh raw outputs, suppressing ALL writes.
 captures=[];original_to_csv=pd.DataFrame.to_csv;identity=ready['identity'];selected='/'in identity
 path=(R/'final-fusion/source/meta_fusion.py')if identity.startswith('finalfusion_')else R/'source'/('export_selected.py'if selected else'fusion.py');spec=importlib.util.spec_from_file_location('readonly_delivery_replay',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.dump=lambda *args,**kw:None
 def capture(frame,target,*args,**kwargs):captures.append((str(target),frame.copy()))
 pd.DataFrame.to_csv=capture
 try:
  if selected:module.main(identity.split('/',1)[0])
  else:
   assert(R/'FROZEN_DEVELOPMENT_SELECTION.json').exists();module.export()
 finally:pd.DataFrame.to_csv=original_to_csv
 frame=next(v for p,v in captures if Path(p)==csv);assert len(frame)==30000;error=float(np.max(abs(frame.default_probability.to_numpy()-saved.default_probability.to_numpy())));assert error<1e-15 and sha(csv)==before
 checked=[];primary=0;optional=0
 for p in sorted((R/'predictions').glob('*/s*/fold*/RESULT.json')):
  v=load(p);d=p.parent;assert v['campaign']=='overnight-refit-20261030'and v['fresh_training'];assert sha(d/'START.json')==v['start_sha256']and sha(d/'prediction.npz')==v['prediction_sha256']
  with np.load(d/'prediction.npz')as z:assert z['raw'].shape==(30000,)and np.isfinite(z['raw']).all()and np.all((z['raw']>=0)&(z['raw']<=1))
  kind=p.relative_to(R/'predictions').parts[0];primary+=kind in ['cann','kumo25','kumo70'];optional+=kind.startswith('selected_');checked.append(dict(relative=str(p.relative_to(R)),result_sha256=sha(p),prediction_sha256=v['prediction_sha256']))
 assert primary==30
 components={};totals=dict(completed_fits=0,failed_attempts=0,interruptions=0)
 for n in load(R/'UNION_PLAN.json')['component_studies']:
  d=R/n;seal=load(d/'SOURCE_SEAL.json')
  for rel,h in seal.items():assert sha(d/rel)==h,'Scientific source changed:'+str(d/rel)
  counts={k:len(list((d/'runs').glob('*/attempt1/'+f)))for k,f in [('completed_fits','RESULT.json'),('failed_attempts','FAILURE.json'),('interruptions','INTERRUPTION.json')]}
  for k,v in counts.items():totals[k]+=v
  v=dict(counts,terminal=(d/'COMPLETION.json').exists(),source_seal_sha256=sha(d/'SOURCE_SEAL.json'),clock_sha256=sha(d/'CLOCK.json'))
  if v['terminal']:
   aa=load(d/'ARTIFACT_CONSISTENCY_AUDIT.json');assert aa['status']=='PASS';e=load(d/'FINAL_EVALUATION.json');v.update(audit_sha256=sha(d/'ARTIFACT_CONSISTENCY_AUDIT.json'),evaluation_sha256=sha(d/'FINAL_EVALUATION.json'),selected_cohort=e.get('selected_cohort'),eligible_local_KS=((e.get('primary_result')or{}).get('metrics')or{}).get('KS'))
  components[n]=v
 meta_studies={}
 for n in load(R/'UNION_PLAN.json').get('meta_composition_studies',[]):
  d=R/n
  for rel,h in load(d/'SOURCE_SEAL.json').items():assert sha(d/rel)==h
  if(d/'COMPLETION.json').exists():
   aa=load(d/'ARTIFACT_CONSISTENCY_AUDIT.json');assert aa['status']=='PASS';meta_studies[n]=dict(status='PASS',meta_designs=aa['meta_designs'],meta_member_fits=aa['meta_member_fits'],new_base_fits=0,audit_sha256=sha(d/'ARTIFACT_CONSISTENCY_AUDIT.json'),selected=aa['selected'])
  elif(d/'FAILURE.json').exists():meta_studies[n]=dict(status='failed_execution_preserved',failure_sha256=sha(d/'FAILURE.json'),no_predictive_conclusion=True)
  else:meta_studies[n]=dict(status='pending')
 # CurrentREADY must be the largest genuinely available audited current prediction.
 eligible_scores=[ready['local_KS']]
 for f in [R/'CURRENT_SUBMISSION_AUDIT.json',R/'CURRENT_FINAL_FUSION_AUDIT.json',*R.glob('AUDIT_CURRENT_*.json')]:
  if not f.exists():continue
  v=load(f)
  if v.get('status')=='PASS'and v.get('fresh_training')and v.get('campaign')=='overnight-refit-20261030':
   q=v.get('candidate')or v.get('selection')or{}
   if 'KS'in q:eligible_scores.append(q['KS'])
 assert ready['local_KS']>=max(eligible_scores),'An audited higher fresh candidate exists; reconcile READY before terminal delivery'
 assert load(R/'READY_CANDIDATE.json')==ready,'READY changed during read-only audit; repeat with new frozen candidate'
 result=dict(status='PASS',epoch=time.time(),scope='Current fresh prediction provenance plus read-only exact frozen meta/mean/CSV replay; component sealed audits inherited',ready=ready,CSV_rows=30000,exact_expected_ID_order=True,replay_max_probability_error=error,primary_fresh_members=primary,optional_fresh_members=optional,production_records=checked,components=components,meta_composition_studies=meta_studies,research_totals=totals,terminal=terminal,independent_confirmation=False,validation_limitation='All70000 adaptively reused; local point selection explicitly authorized, no verified public/private gain.',new_refit_validation_KS=None)
 dump('ARTIFACT_CONSISTENCY_AUDIT.json'if terminal else'ROOT_LIVE_AUDIT.json',result)
 if terminal:
  dump('COMPLETION.json',dict(status='completed_computation_audit',epoch=time.time(),artifact_audit_sha256=sha(R/'ARTIFACT_CONSISTENCY_AUDIT.json'),ready=ready,research_totals=totals,planned_attempt_slots=load(R/'UNION_PLAN.json')['planned_fit_attempts'],submission_status='scheduled_separately_06:25WIB; require server SUBMITTED receipt',not_all_planned_slots_claimed_complete=True))
 return result
def main():
 a=argparse.ArgumentParser();a.add_argument('--check-only',action='store_true');args=a.parse_args();clock=load(R/'CLOCK.json')
 if args.check_only:result=audit();print(json.dumps({k:result[k]for k in ['status','primary_fresh_members','optional_fresh_members','replay_max_probability_error']}));return
 # Reserve the original window for prospectively registered work; no early root seal.
 while time.time()<clock['model_deadline_epoch']+60:time.sleep(min(30,clock['model_deadline_epoch']+60-time.time()))
 terminal_wait_end=min(clock['deadline_epoch']-900,clock['model_deadline_epoch']+720)
 while time.time()<terminal_wait_end and pending():time.sleep(15)
 # Finalize only after actual remaining training stopped; all child clocks cap it.
 waits=pending()
 assert not any(not(R/n/'COMPLETION.json').exists()for n in load(R/'UNION_PLAN.json')['component_studies']), 'Child terminal evidence missing; preserve partial status'
 result=audit(terminal=True);dump('TERMINAL_AUDIT_DISPATCH.json',dict(status='PASS',epoch=time.time(),remaining_promotions=waits,audit_sha256=sha(R/'ARTIFACT_CONSISTENCY_AUDIT.json')))
if __name__=='__main__':
 try:main()
 except BaseException:
  p=R/'private/TERMINAL_AUDIT_FAILURE.log';p.parent.mkdir(exist_ok=True);p.write_text(traceback.format_exc());dump('TERMINAL_AUDIT_FAILURE.json',dict(status='execution_review_required',epoch=time.time(),trace_sha256=sha(p)));raise
