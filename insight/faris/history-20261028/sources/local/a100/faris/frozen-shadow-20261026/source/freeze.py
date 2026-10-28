"""Publish immutable fold tables once; subsequent work changes models only."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from common import R,root_path,dump,sha,utc
def main():
    assert not (R/'FEATURE_FREEZE.json').exists()
    root=root_path('packet_root');anchor=json.loads((root/'ANCHOR_SOURCE.json').read_text());base=anchor['feature_columns'];head=anchor['head_columns']
    assert len(base)==90 and len(head)==18
    added=['shadow_interval12']+[f'joint_z{j}' for j in range(8)]+['joint_posterior_variance']
    master=list(dict.fromkeys(base+head+['shadow_interval12','shadow_known12']+added));assert 'target' not in master
    views={'champion90':base,'shadow91':base+['shadow_interval12'],'latent99':base+added[1:],'full100':base+added,'master':master}
    dest=R/'frozen';dest.mkdir(exist_ok=False);folds=[]
    for fold in range(5):
        p=dest/f'fold{fold}';p.mkdir();q=pd.read_parquet(root/f'packets/fold{fold}/current.parquet');roles=np.load(root/f'packets/fold{fold}/roles.npz')
        s=pd.read_parquet(R/f'learned/fold{fold}/shadow.parquet');z=pd.read_parquet(R/f'learned/fold{fold}/latent.parquet')
        for c in s:q[c]=s[c].to_numpy()
        for c in z:q[c]=z[c].to_numpy()
        q=q[master];assert len(q)==70000 and q.columns.is_unique
        q.to_parquet(p/'features.parquet',index=False)
        np.savez_compressed(p/'roles.npz',**dict(roles))
        old=np.load(root/f'packets/fold{fold}/baseline.npz');np.savez_compressed(p/'baseline.npz',**dict(old))
        dump({'views':views,'head_columns':head,'fold':fold,'features_sha256':sha(p/'features.parquet'),'rows':70000,
              'roles_sha256':sha(p/'roles.npz'),'baseline_sha256':sha(p/'baseline.npz'),
              'shadow_recipe_sha256':sha(R/f'learned/fold{fold}/SHADOW.json'),'latent_recipe_sha256':sha(R/f'learned/fold{fold}/LATENT.json')},p/'MANIFEST.json')
        folds.append({'fold':fold,'manifest_sha256':sha(p/'MANIFEST.json')})
    seal={'status':'frozen','UTC':utc(),'views':views,'master_columns':len(master),'folds':folds,
          'post_freeze_policy':'No feature additions, removals or feature-value regeneration for search. Vary model family, hyperparameters, context, loss and seed only. Fixed ablation views are declared now.',
          'supervised_representation_boundary':'Each latent model uses only context selected-loan labels; shadow teachers use borrower-disjoint historical labels from context customers. No correction/outer targets fit learned representations.',
          'source_sha256':{p.name:sha(p) for p in (R/'source').glob('*.py')},'independent_audit':False}
    dump(seal,R/'FEATURE_FREEZE.json');print(json.dumps({'status':'frozen','master_columns':len(master),'views':{k:len(v) for k,v in views.items()}}),flush=True)
if __name__=='__main__':main()
