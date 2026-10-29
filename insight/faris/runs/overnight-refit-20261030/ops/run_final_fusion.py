"""Run prospectively sealed meta recipes, audit before any READY replacement."""
from pathlib import Path
import os
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='2'
import hashlib,importlib.util,json,sys,time,traceback
import numpy as np,pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve
R=Path(__file__).resolve().parents[1];D=R/'final-fusion';sys.path.insert(0,str(D/'source'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,v):
 p=D/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,default=str)+'\n');q.replace(p)
try:
 C=json.load(open(R/'CLOCK.json'));assert time.time()<C['admission_end_epoch'];assert sha(R/'CLOCK.json')=='576f1da25b5636c93a8d93d36c3f993cdfdf300cdade61563799362e195507d0'
 receipt=json.load(open(D/'PREPUBLICATION.json'));assert receipt['config_sha256']==sha(D/'CONFIG.json')and receipt['source_seal_sha256']==sha(D/'SOURCE_SEAL.json')
 seal=json.load(open(D/'SOURCE_SEAL.json'))
 for p,h in seal.items():assert sha(D/p)==h
 config=json.load(open(D/'CONFIG.json'));spec=importlib.util.spec_from_file_location('sealed_final_fusion',D/'source/meta_fusion.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 original_dump=m.dump;proposals=[]
 def guarded(path,value):
  if Path(path)==R/'READY_CANDIDATE.json':proposals.append(value)
  else:original_dump(path,value)
 m.dump=guarded
 frozen=m.evaluate();assert len(frozen['candidates'])==len(config['designs'])==28
 y=np.load(m.C/'evaluation_inputs/target.npy');replay={};records=[]
 for design in config['designs']:
  seed_predictions=[]
  for seed in config['seeds']:
   q=np.full(70000,np.nan);coverage=np.zeros(70000,int)
   for fold in range(5):
    d=D/'meta_members'/design['id']/f's{seed}'/f'fold{fold}';v=json.load(open(d/'META_FROZEN.json'));assert v['design']==design and v['source_sha256']==sha(D/'source/meta_fusion.py')and v['prediction_sha256']==sha(d/'prediction.npz');assert not v['outer_labels_used_for_fitting']
    z=np.load(d/'prediction.npz');rows=z['rows'];assert len(rows)==14000 and z['prediction'].shape==(14000,)and np.isfinite(z['prediction']).all();q[rows]=z['prediction'];coverage[rows]+=1;records.append(dict(relative=str((d/'META_FROZEN.json').relative_to(D)),sha256=sha(d/'META_FROZEN.json'),prediction_sha256=v['prediction_sha256']))
   assert np.all(coverage==1);seed_predictions.append(q)
  expected=np.mean(seed_predictions,axis=0);folder=D/'development'/design['id'];v=json.load(open(folder/'FROZEN.json'));assert v['prediction_sha256']==sha(folder/'prediction.npz')and v['source_sha256']==sha(D/'source/meta_fusion.py');p=np.load(folder/'prediction.npz')['prediction'];assert np.array_equal(expected,p)
  ks=float(ks_2samp(p[y==1],p[y==0]).statistic);fpr,tpr,_=roc_curve(y,p,drop_intermediate=False);assert abs(ks-float(np.max(abs(tpr-fpr))))<1e-12
  selected=next(z for z in frozen['candidates']if z['id']==design['id']);assert abs(ks-selected['KS'])<1e-12;replay[design['id']]=ks
 winner=min(frozen['candidates'],key=lambda x:(-x['KS'],x['id']));assert winner==frozen['selection']
 assert time.time()<C['model_deadline_epoch'];m.export();out=R/f'submission_CURRENT_{winner["id"]}.csv';first_hash=sha(out);a=json.load(open(R/'CURRENT_FINAL_FUSION_AUDIT.json'));assert a['sha256']==first_hash
 # Replay frozen production meta recipe without touching ANY file.
 captured=[];old_to_csv=pd.DataFrame.to_csv;m.dump=lambda *args,**kw:None
 def capture(frame,path,*args,**kw):captured.append((str(path),frame.copy()))
 pd.DataFrame.to_csv=capture
 try:m.export()
 finally:pd.DataFrame.to_csv=old_to_csv
 prediction=next(frame for path,frame in captured if Path(path)==out);saved=pd.read_csv(out);error=float(np.max(abs(saved.default_probability-prediction.default_probability)));assert error<1e-15 and sha(out)==first_hash
 audit=dict(status='PASS',epoch=time.time(),meta_designs=28,meta_member_fits=len(records),base_fits_reused=40,new_base_fits=0,source_seal_sha256=sha(D/'SOURCE_SEAL.json'),source_sha256=sha(D/'source/meta_fusion.py'),config_sha256=sha(D/'CONFIG.json'),recomputed_cohorts=replay,member_hash_records=records,selected=winner,CSV_sha256=first_hash,production_replay_max_abs_error=error,independent_confirmation=False,limit='All70k adaptively reused; no supported/public/privategain inferred from this point selection.')
 dump('ARTIFACT_CONSISTENCY_AUDIT.json',audit)
 # Publish only after the complete deterministic/cohort/full-fit audit.
 proposed=next((v for v in reversed(proposals)if v['identity']==winner['id']),None);current=json.load(open(R/'READY_CANDIDATE.json'));promoted=False
 if proposed is not None and proposed['local_KS']>current['local_KS']:
  assert proposed['sha256']==first_hash;original_dump(R/'READY_CANDIDATE.json',proposed);promoted=True
 done=dict(stage='complete',epoch=time.time(),meta_designs=28,meta_member_fits=len(records),new_base_fits=0,reused_fresh_base_members=40,best_local_KS=winner['KS'],identity=winner['id'],promoted=promoted,artifact_audit_sha256=sha(D/'ARTIFACT_CONSISTENCY_AUDIT.json'),source_seal_sha256=sha(D/'SOURCE_SEAL.json'),conditional_development_only=True)
 dump('COMPLETION.json',done);original_dump(R/'FINAL_FUSION_STATE.json',done);print(json.dumps(done))
except BaseException:
 p=D/'FAILURE_PRIVATE.log';p.write_text(traceback.format_exc());dump('FAILURE.json',dict(status='execution_review_required',epoch=time.time(),trace_sha256=sha(p),no_implicit_retry=True));raise
