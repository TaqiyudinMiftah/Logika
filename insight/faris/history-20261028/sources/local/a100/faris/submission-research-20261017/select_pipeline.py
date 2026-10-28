import json,numpy as np,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent;e=r/'evidence';m=json.loads((e/'manifest.json').read_text());m['components']['depth_linear']=json.loads((e/'depth_linear/manifest.json').read_text());m['components']['kumo']=json.loads((e/'kumo/manifest.json').read_text())
bank=r.parent/'depth-20261017/stage/bank';y=np.load(bank/'development_y.npy').astype(np.int64);folds=np.load(bank/'folds.npy');rows=np.load(bank/'development_rows.npy')
def ks(y,p):
 y=np.asarray(y,dtype=np.int64);ix=np.argsort(p,kind='stable');v=p[ix];yy=y[ix];last=np.r_[v[1:]!=v[:-1],True];d=np.cumsum(yy)/np.sum(yy)-np.cumsum(1-yy)/np.sum(1-yy);return float(np.max(np.abs(d[last])))
a={}
for k in m['components']:
 if (e/k/'prediction.npy').exists():p=np.load(e/k/'prediction.npy')
 else:
  z=np.load(e/k/'validation.npz');assert np.array_equal(z['rows'],rows) and np.array_equal(z['fold'],folds);p=z['prediction']
 assert len(p)==len(y) and np.isfinite(p).all();a[k]=p
 original=json.loads((e/k/('scores.json' if (e/k/'scores.json').exists() else 'outcome.json')).read_text());expected=original.get('pooled_oof',original.get('evaluation',{})).get('ks',original.get('evaluation',{}).get('score'))
 assert abs(ks(y,p)-expected)<1e-10,(k,expected,ks(y,p))
parents=m['ensemble']['parents'];base=np.mean(np.sort(np.column_stack([a[k] for k in parents]),axis=1)[:,1:-1],axis=1);depth=(a['depth_lgb']+a['depth_linear'])/2
candidates={'existing_feature_ensemble':base,'kumo':a['kumo'],'equal_kumo_features':.5*base+.5*a['kumo'],'equal_three_blocks':(base+a['kumo']+depth)/3,'equal_kumo_depth':.5*a['kumo']+.5*depth}
out=[]
for name,p in candidates.items():out.append({'name':name,'pooled_ks':ks(y,p),'per_fold':[ks(y[folds==f],p[folds==f]) for f in range(3)]})
print(json.dumps(out,indent=2));np.savez_compressed(e/'candidate_oof.npz',**candidates,rows=rows,fold=folds)
(r/'component_manifest.json').write_text(json.dumps(m,indent=2));(r/'candidate_comparison.json').write_text(json.dumps(out,indent=2))
# Freeze one fixed combination; no learned weights and no validation-threshold fitting.
best=max(out,key=lambda v:v['pooled_ks']);fixed={'existing_feature_ensemble':{'features':1},'kumo':{'kumo':1},'equal_kumo_features':{'features':.5,'kumo':.5},'equal_three_blocks':{'features':1/3,'kumo':1/3,'depth':1/3},'equal_kumo_depth':{'kumo':.5,'depth':.5}}[best['name']]
recipe={'id':'research-submission-20261017-v1','combination':fixed,'selection':best,'candidate_count':len(out),'metric':'tie-aware KS','components':m,'training_seed':2026,'holdout_fallback_rule':'Use known reference-safe frozen wave1 primary only if terminal confirmation KS is more than 0.01 below the existing v3 reference; no retuning on confirmation.','caveat':'OOF reused for historical/adaptive selection. Confirmation partition historically used and wave1 results already viewed. No unbiased generalization or leaderboard claim.'}
(r/'frozen_recipe.json').write_text(json.dumps(recipe,indent=2));(r/'frozen_recipe.sha256').write_text(hashlib.sha256((r/'frozen_recipe.json').read_bytes()).hexdigest()+'\n')
