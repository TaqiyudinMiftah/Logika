"""Read-only independent final-cohort arithmetic and provenance checks."""
from pathlib import Path
import argparse,hashlib,json,datetime
import numpy as np
from scipy.stats import ks_2samp

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main(root):
 r=Path(root);config=read(r/'CONFIG.json');clock=read(r/'CLOCK.json');selection=read(r/'FINAL_SELECTION.json');e=read(r/'FINAL_EVALUATION.json');audit=read(r/'ARTIFACT_CONSISTENCY_AUDIT.json');completion=read(r/'COMPLETION.json')
 assert audit['status']=='PASS'and audit['completed_fits']==1920 and audit['failure_records']==audit['not_completed']==0
 assert len(audit['recomputed_cohorts'])==768 and completion['audit_sha256']==digest(r/'ARTIFACT_CONSISTENCY_AUDIT.json')
 assert digest(r/'CONFIG.json')==clock['config_sha256']and digest(r/'VALIDATION_CONTRACT.json')==clock['contract_sha256']
 assert read(r/'SOURCE_SEAL.json')=={str(p.relative_to(r)):digest(p)for p in(r/'source').glob('*.py')}
 name=e['selected_cohort'];assert selection['identity']['cohort']==name
 design,mode=name.rsplit('__',1);y=np.load(r/'evaluation_inputs/target.npy');cohort=r/f'evaluation/cohorts/{name}';a=np.load(cohort/'prediction.npz')
 assert digest(cohort/'prediction.npz')==e['primary_result']['prediction_sha256']and np.array_equal(a['rows'],np.arange(70000))
 seeds=[]
 for seed in config['seeds']:
  vector=np.full(70000,np.nan);coverage=np.zeros(70000,int)
  for fold in range(5):
   c=r/f'composed/{design}/s{seed}/fold{fold}';v=np.load(c/'prediction.npz');result=read(c/'RESULT.json')
   assert digest(c/'prediction.npz')==result['prediction_sha256']and digest(c/'FROZEN.json')==result['frozen_sha256']
   roles=np.load(r/f'packets/fold{fold}/roles.npz');assert np.array_equal(v['rows'],roles['validation']);assert len(np.unique(np.r_[roles['context'],roles['correction'],roles['validation']]))==70000
   raw=r/f'runs/{design}__s{seed}__f{fold}/attempt1';start=read(raw/'START.json');out=read(raw/'RESULT.json');q=np.load(raw/'prediction.npz')
   assert start['seed']==seed+fold and not start['outer_labels_loaded']and not start['correction_labels_used_for_training']
   assert np.array_equal(q['rows'],np.r_[roles['correction'],roles['validation']])and digest(raw/'START.json')==out['start_sha256']and digest(raw/'prediction.npz')==out['prediction_sha256']
   assert out['audit']['full_refit_fresh_initialization']and not out['audit']['preprocessing_refit_on_context']and out['audit']['batch_replay_max_abs_probability_error']<=5e-6
   vector[v['rows']]=v[mode];coverage[v['rows']]+=1
  assert np.all(coverage==1)and np.isfinite(vector).all();seeds.append(vector)
 assert np.array_equal(np.mean(seeds,axis=0),a['prediction'])
 ks=lambda p:float(ks_2samp(p[y==0],p[y==1],method='asymp').statistic)
 point=ks(a['prediction']);assert abs(point-e['primary_result']['metrics']['KS'])<1e-12
 base=ks(np.load(r/'evaluation_inputs/baseline.npz')['prediction']);prior=ks(np.load(r/'evaluation_inputs/previous.npz')['prediction']);assert abs(base-.5306361042400267)<1e-12 and abs(prior-.5312163876106586)<1e-12
 # Descriptive matched contrasts only; these correlated outcomes are not tests.
 board=read(r/'SCOREBOARD.json')['candidates'];contrasts={}
 for mode in ['standalone','fixed_010','meta_adaptive','stack']:
  pairs=[]
  for key,d in config['designs'].items():
   if not d['learn_selection']:continue
   peer=key.replace('_learn_','_fixed_');pairs.append(board[key+'__'+mode]['metrics']['KS']-board[peer+'__'+mode]['metrics']['KS'])
  assert len(pairs)==96;contrasts[mode]=dict(paired_designs=len(pairs),median_learned_minus_fixed_KS=float(np.median(pairs)),positive_pairs=sum(v>0 for v in pairs))
 value=dict(status='PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_cohort=name,KS=point,baseline_KS=base,prior_KS=prior,complete_fits=1920,source_audit_replayed_modes=768,independently_reconstructed_selected_fit_count=10,seed_count=2,fold_count=5,rows=70000,uncertainty_from_frozen_predeclared_assessment=e['previous_winner_uncertainty'],gain_supported=e['primary_development_gain_supported'],descriptive_selector_contrasts=contrasts,independent_statistical_confirmation=False,scope='Artifact and arithmetic verification; adaptively reused development, no new fit or selection',clock_sha256=digest(r/'CLOCK.json'),audit_sha256=digest(r/'ARTIFACT_CONSISTENCY_AUDIT.json'))
 print(json.dumps(value))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
