"""Requalify architecture and preprocessing boundaries on the sealed development protocol."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,sys,importlib.metadata,inspect
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from common import R,sha,dump,utc,ks
from architectures import Preprocessor,make_model,training_loss


def main():
    torch.set_num_threads(2);torch.manual_seed(261027)
    paths=json.loads((R/'PATHS.json').read_text());prior=Path(paths['prior_root']);seal=json.loads((R/'FEATURE_FREEZE.json').read_text());configs=json.loads((R/'CONFIGS.json').read_text())
    assert sha(R/'FEATURE_FREEZE.json')=='071c75e0f59cf92429bde9c76a64e27931e1eea03bbbbeac2e2589b5fee68dd6'
    old=json.loads((prior/'TERMINAL_AUDIT.json').read_text());assert old['status']=='passed'and old['verified_fits']==340
    assert old['feature_freeze_sha256']==sha(R/'FEATURE_FREEZE.json')
    assert ks([0,0,1,1],[0,0,1,1])==1 and ks([0,1,0,1],[.5]*4)==0
    assert ks([0,0,1,1],[1,1,0,0])==1
    fold_evidence=[]
    for item in seal['folds']:
        f=item['fold'];p=Path(paths['frozen_root'])/f'fold{f}';m=json.loads((p/'MANIFEST.json').read_text())
        assert sha(p/'MANIFEST.json')==item['manifest_sha256']
        for filename,key in [('features.parquet','features_sha256'),('roles.npz','roles_sha256'),('baseline.npz','baseline_sha256')]:assert sha(p/filename)==m[key]
        roles=np.load(p/'roles.npz');assert len(roles['context'])==50000 and len(roles['correction'])==6000 and len(roles['validation'])==14000
        assert not np.intersect1d(roles['context'],np.r_[roles['correction'],roles['validation']]).size
        yc=roles['y_context'];yr=roles['y_correction'];assert min(np.bincount(yc.astype(int)))>1000 and min(np.bincount(yr.astype(int)))>100
        fold_evidence.append({'fold':f,'context_class_counts':np.bincount(yc.astype(int)).tolist(),'correction_class_counts':np.bincount(yr.astype(int)).tolist(),'manifest_hash_passed':True,'roles_disjoint':True})
    # Synthetic counterfactual: arbitrary held-out rows cannot change fitted preprocessing.
    frame=pd.DataFrame({'number':np.arange(100,dtype=float),'missing':np.r_[np.full(20,np.nan),np.arange(80,dtype=float)],'category':['a','b']*50})
    train=np.arange(60);p=Preprocessor().fit(frame,train);changed=frame.copy();changed.loc[60:,'number']=1e100;changed.loc[60:,'category']='unseen'
    q=Preprocessor().fit(changed,train);assert np.array_equal(p.median,q.median)and np.array_equal(p.quantile.quantiles_,q.quantile.quantiles_)and p.vocabs==q.vocabs
    z,c=p.transform(changed);assert (c[60:]==0).all();assert np.isfinite(z).all()
    tests=[]
    for kind in sorted({v['kind']for v in configs.values()}):
        spec=next(v for v in configs.values()if v['kind']==kind);torch.manual_seed(261027)
        model=make_model(208,[5,9,3,7],spec);x=torch.randn(16,208);cat=torch.stack([torch.randint(n,(16,))for n in [5,9,3,7]],dim=1);y=torch.tensor([0.,1.]*8)
        before=next(model.parameters()).detach().clone();opt=torch.optim.AdamW(model.parameters(),lr=.001)
        model.train();values=[]
        for _ in range(3):
            logits=model(x,cat);loss=training_loss(logits,y,.1);assert torch.isfinite(loss);opt.zero_grad();loss.backward()
            grads=[v.grad for v in model.parameters()if v.grad is not None];assert grads and all(torch.isfinite(g).all()for g in grads);opt.step();values.append(float(loss.detach()))
        assert not torch.equal(before,next(model.parameters()).detach());model.eval()
        with torch.inference_mode():a=model(x,cat);b=model(x,cat)
        assert torch.equal(a,b)and a.shape==(16,16 if kind=='tabm'else 1)
        tests.append({'kind':kind,'loss_values':values,'gradient_finite':True,'parameters_updated':True,'eval_repeat_exact':True,'parameters':sum(v.numel()for v in model.parameters())})
    old_scores=json.loads((prior/'evaluation/RESULT.json').read_text())
    evidence={'status':'prepared_checks_passed_runtime_pilot_pending','UTC':utc(),'contract_sha256':sha(R/'VALIDATION_CONTRACT.json'),
              'feature_freeze_sha256':sha(R/'FEATURE_FREEZE.json'),'configs_sha256':sha(R/'CONFIGS.json'),'source_sha256':sha(__file__),
              'architecture_source_sha256':sha(R/'source/architectures.py'),'fold_checks':fold_evidence,
              'preprocessing_counterfactual_passed':True,'unknown_categories_safe':True,'metric_known_cases_passed':True,
              'architecture_gradient_checks':tests,'baseline_reused_verified_KS':old_scores['baseline_KS'],
              'simple_reference_reused_verified_KS':old_scores['diagnostic_metrics']['logistic_champion90_C0.01']['KS'],
              'constant_prediction_KS':0.,'null_AUC':.5,'baseline_prediction_replay_max_abs':old['baseline_prediction_replay_max_abs'],
              'uncertainty_policy':'5folds,2seeds perdesign;paired class-stratifiedbootstrap2000 conditionaldevelopmentreplicates',
              'torch_version':importlib.metadata.version('torch'),'tabm_version':importlib.metadata.version('tabm'),
              'limits':'All70kouterexamples reused. No untouchedaudit. Historical snapshot completeness inherited assumption. Synthetic architecture tests establish execution/boundaries, not expected score. Runtime pilot additionally required before broad admission.'}
    dump(evidence,R/'GATE_PREP.json');print(json.dumps(evidence,indent=2),flush=True)


if __name__=='__main__':main()
