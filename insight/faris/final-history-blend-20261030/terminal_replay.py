"""Read-only independent terminal evidence replay. No candidate/selection changes."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import hashlib,json,time,collections
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp
R=Path('/mnt/home/faris/logika-twelvehour-cross-20261030')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
roles=[np.load(R/f'packets/fold{f}/roles.npz')for f in range(5)]
for z in roles:
 assert len(z['context'])==50000 and len(z['correction'])==6000 and len(z['validation'])==14000
 assert len(set(np.r_[z['context'],z['correction'],z['validation']]))==70000
raw=[];designs=collections.defaultdict(set)
for p in R.glob('runs/*/attempt*/RESULT.json'):
 d=json.loads(p.read_text());s=json.loads((p.parent/'START.json').read_text());assert d['status']=='complete'
 assert sha(p.parent/'START.json')==d['start_sha256'] and sha(p.parent/'prediction.npz')==d['prediction_sha256']
 assert d['clock_sha256']==sha(R/'CLOCK.json') and s['source_seal_sha256']==sha(R/'SOURCE_SEAL.json')
 assert s['seed']==s['task']['seed']+s['task']['fold']
 assert not s['outer_labels_loaded'] and not d['outer_labels_loaded'] and not d['correction_labels_used_for_training'] and not d['weights_saved']
 z=np.load(p.parent/'prediction.npz');f=s['fold'];assert np.array_equal(z['rows'],np.r_[roles[f]['correction'],roles[f]['validation']]);assert np.isfinite(z['raw']).all()
 assert d['audit'].get('batch_replay_max_abs_probability_error',0)<=5e-6
 designs[s['design_id']].add((s['task']['seed'],f));raw.append(dict(path=str(p.relative_to(R)),result_sha256=sha(p),prediction_sha256=d['prediction_sha256']))
print('raw verified',len(raw),flush=True)
y=np.load(R/'evaluation_inputs/target.npy');ks=lambda p:float(ks_2samp(p[y==1],p[y==0]).statistic)
assert abs(ks(np.load(R/'evaluation_inputs/baseline.npz')['prediction'])-.5306361042400267)<1e-12
assert abs(ks(np.load(R/'evaluation_inputs/previous.npz')['prediction'])-.5316414268134972)<1e-12
records=[]
for i,p in enumerate((R/'evaluation/cohorts').iterdir()):
 if not (p/'RESULT.json').exists():continue
 f=json.loads((p/'FROZEN.json').read_text());d=json.loads((p/'RESULT.json').read_text());assert sha(p/'prediction.npz')==f['prediction_sha256']
 identity=f['identity'];prediction=np.zeros(70000);coverage=np.zeros(70000,int)
 for q,h in identity['composed_results'].items():
  q=R/q;assert sha(q)==h;c=json.loads(q.read_text());assert sha(q.parent/'prediction.npz')==c['prediction_sha256'];z=np.load(q.parent/'prediction.npz');prediction[z['rows']]+=z[identity['mode']]/2;coverage[z['rows']]+=1
 assert np.all(coverage==2);saved=np.load(p/'prediction.npz')['prediction'];assert np.max(abs(prediction-saved))<1e-14
 score=ks(saved);assert abs(score-d['metrics']['KS'])<1e-12;records.append(dict(id=p.name,KS=score,prediction_sha256=sha(p/'prediction.npz')))
 if i%200==0:print('cohort replay',i,flush=True)
records.sort(key=lambda d:(-d['KS'],d['id']));selection=json.loads((R/'FINAL_SELECTION.json').read_text());assert selection['identity']['cohort']==records[0]['id']
full=sum(len(v)==10 for v in designs.values());v=dict(status='PASS',UTC=time.strftime('%FT%TZ',time.gmtime()),completed_raw_fits=len(raw),complete2seed5fold_designs=full,cohort_scores_and_mean_replayed=len(records),best=records[0],clock_sha256=sha(R/'CLOCK.json'),source_seal_sha256=sha(R/'SOURCE_SEAL.json'),raw_hash_rows_seed_training_boundaries='PASS',exact_SciPy_KS_and_baseline_previous_replay='PASS',raw_manifest_sha256=hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest(),no_model_or_selection_changes=True,untouched_audit=False)
dump(R/'INDEPENDENT_TERMINAL_REPLAY.json',v);print(json.dumps(v),flush=True)
