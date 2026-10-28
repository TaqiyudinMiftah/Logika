"""Fit outer-specific heads using only that outer training partition's labels."""
from pathlib import Path
import json,time,hashlib,numpy as np,pandas as pd,joblib
import feature_head,baseline as b
R=Path(__file__).resolve().parents[1];A=R.parent

def read(p):return json.loads(Path(p).read_text())
def validate(j):
 d=R/f'data/job{j}';m=read(d/'manifest.json')
 for n,h in m['files'].items():assert b.sha(d/n)==h
 r=R/f'runs/job{j}';c=read(r/'complete.json');assert c['status']=='complete'and c['job']==j
 for n,h in c['files'].items():assert b.sha(r/n)==h
 q=pd.read_parquet(d/'query.parquet');z=np.load(r/'raw.npz');ix=np.load(d/'indices.npz')
 assert'target'not in q and np.array_equal(z['rows'],ix['validation'])and np.array_equal(q.application_id.to_numpy(str),z['application_id'])
 assert read(r/'boundary.json')['full_cold_model_replay_max_error']<2e-5
 return q,z,m

def fit(f):
 assert time.time()<read(R/'training-plan.json')['original_deadline']and not(R/'STOP').exists()
 out=R/f'heads/fold{f}';assert not out.exists();out.mkdir(parents=True)
 # This is the only label-bearing input opened here: all56000outer-training rows.
 final=f*6+5;ctx=pd.read_parquet(R/f'data/job{final}/context.parquet');labels=ctx.set_index('application_id').target
 assert len(labels)==56000 and labels.index.is_unique
 frames=[];raw=[];rows=[]
 for g in range(5):
  q,z,m=validate(f*6+g);assert m['role']['outer_fold']==f and m['role']['kind']=='inner'and len(q)==11200
  assert set(q.application_id)<=set(labels.index)
  frames.append(q);raw.append(z['prediction']);rows.append(z['rows'])
 train=pd.concat(frames,ignore_index=True);rp=np.concatenate(raw);ri=np.concatenate(rows)
 assert len(train)==56000 and train.application_id.is_unique and set(train.application_id)==set(labels.index)and len(set(ri))==56000
 train['target']=labels.reindex(train.application_id).to_numpy(np.int64);head=feature_head.fit(train,rp,read(R/'head_columns.json'),.003,.5)
 joblib.dump(head,out/'correction.joblib',compress=3)
 query,z,m=validate(final);assert m['role']['outer_fold']==f and m['role']['kind']=='final'and len(query)==14000 and set(query.application_id).isdisjoint(labels.index)
 d=A/f'kumo-coverage-placement-20261021/conditions/fold{f}/control90';complete=read(d/'complete.json');assert b.sha(d/'correction.joblib')==complete['small_files']['correction.joblib'];old=joblib.load(d/'correction.joblib')
 shared=A/f'kumo-coverage-placement-20261021/shared/fold{f}';sm=read(shared/'complete.json');assert b.sha(shared/'validation.npz')==sm['files']['validation.npz'];sp=np.load(shared/'validation.npz');assert np.array_equal(sp['rows'],z['rows'])
 main=.75*b.corrected_probability(head,query,z['prediction'])[0]+.25*sp['prediction'];control=.75*b.corrected_probability(old,query,z['prediction'])[0]+.25*sp['prediction']
 replay=.75*b.corrected_probability(joblib.load(out/'correction.joblib'),query,z['prediction'])[0]+.25*sp['prediction'];assert np.max(abs(replay-main))<1e-14
 replay_main=.75*b.corrected_probability(head,query,z['replayed_prediction'])[0]+.25*sp['prediction'];replay_control=.75*b.corrected_probability(old,query,z['replayed_prediction'])[0]+.25*sp['prediction']
 np.savez_compressed(out/'validation.npz',rows=z['rows'],application_id=query.application_id.to_numpy(str),crossfit56000=main,heldout6000=control,replayed_crossfit56000=replay_main,replayed_heldout6000=replay_control)
 np.savez_compressed(out/'meta_inputs.npz',rows=ri,prediction=rp,application_id=train.application_id.to_numpy(str))
 b.dump(dict(fold=f,meta_training_rows=56000,inner_context_rows=44800,inner_folds=5,final_context_rows=56000,every_meta_row_predicted_without_its_inner_label=True,outer_labels_available_to_meta_fit=False,spline_outside_learned_head=True,meta_training_ids_sha256=hashlib.sha256('\n'.join(train.application_id).encode()).hexdigest(),conditional_comparison='Same finalKumo raw probability and sharedspline, different honest correction fitting procedure; prior/context distribution changes remain part of procedure.',head_gradient_max=head['gradient_max_abs']),out/'boundary.json')
 b.dump(dict(status='complete',fold=f,files={z.name:b.sha(z)for z in out.iterdir()if z.is_file()}),out/'complete.json')
if __name__=='__main__':
 for f in range(5):fit(f)
 b.dump(dict(status='complete',folds=5,outer_labels_opened=False),R/'HEADS_COMPLETE.json')
