import numpy as np,joblib
import run
checks=[]
for f in range(5):
 d=run.STAGED/f'fold{f}';manifest=run.read(d/'manifest.json');ix=dict(np.load(d/'indices.npz'));pc=run.source(f,'classification',ix,manifest)
 for n,h in manifest['files'].items():assert run.b.sha(d/n)==h
 q=run.pd.read_parquet(d/'query.parquet');assert 'target'not in q
 k=run.MAIN/f'conditions/fold{f}/control90';c=run.read(k/'complete.json')
 for n in ['raw_predictions.npz','correction.joblib']:assert run.b.sha(k/n)==c['small_files'][n]
 raw=np.load(k/'raw_predictions.npz');assert np.array_equal(raw['rows'],np.r_[ix['correction'],ix['validation']]);base=joblib.load(k/'correction.joblib')
 x,o=run.b.design(base,q.iloc[:6000],raw['prediction'][:6000]);labels=np.load(d/'correction_labels.npy');v,g=run.b.objective(base['coefficients'],x,o,labels,np.ones(6000),.003)
 assert abs(v-base['objective'])<1e-10 and np.max(abs(g))<1e-4
 assert len(set(np.r_[ix['context'],ix['correction'],ix['validation']]))==70000
 checks.append(dict(fold=f,head_rows=6000,outer_rows=14000,baseline_objective_reproduced=True,all_component_roles_and_hashes_match=True))
run.write(run.R/'SCIENTIFIC_GATE.json',dict(status='passed_for_bounded_heldout_fusion',checks=checks,historical_label_reuse=True,outer_labels_loaded=False,scope='Input provenance and fold-local meta-fit boundary, not generalization certification'))
print('ALL5ROLE_AND_BASELINE_CHECKS_PASSED',flush=True)
