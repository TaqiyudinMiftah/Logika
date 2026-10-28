"""Audit feature/borrower boundaries and reproduce the existing downstream baseline."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import json,sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
from common import R,root_path,sha,dump,ks,utc
from shadow_features import interval_loss
def main():
    seal=json.loads((R/'FEATURE_FREEZE.json').read_text());assert seal['status']=='frozen'
    assert ks([0,0,1,1],[0,0,1,1])==1 and ks([0,0,1,1],[.5,.5,.5,.5])==0
    scale=torch.tensor([100.]);shape=torch.tensor([1.]);lower=torch.tensor([90.]);upper=torch.tensor([180.]);weight=torch.tensor([1.])
    loss=float(interval_loss(scale,shape,lower,upper,torch.tensor([True]),weight));expected=-np.log(np.exp(-.9)-np.exp(-1.8));assert abs(loss-expected)<1e-6
    assert abs(float(interval_loss(scale,shape,lower,upper,torch.tensor([False]),weight))-.9)<1e-6
    current=pd.read_parquet(R/'prepared/current.parquet',columns=['customer_id']);h=pd.read_parquet(R/'prepared/historical.parquet',columns=['customer_id'])
    sys.path.insert(0,'/mnt/home/faris/logika-faris-exploration/transfer-20261025/source')
    import baseline as b
    import feature_head
    checks=[]
    for fold in range(5):
        root=R/f'frozen/fold{fold}';m=json.loads((root/'MANIFEST.json').read_text());p=root/'features.parquet';assert sha(p)==m['features_sha256']
        x=pd.read_parquet(p);q=pd.read_parquet(root_path('packet_root')/f'packets/fold{fold}/current.parquet');roles=np.load(root/'roles.npz');base=np.load(root/'baseline.npz')
        assert list(x)[:90]==m['views']['champion90'] and x.iloc[:,:90].equals(q[m['views']['champion90']])
        assert all(len(m['views'][v])<=100 for v in ['champion90','shadow91','latent99','full100'])
        assert all(np.isfinite(x[c]).all() for c in ['shadow_interval12','shadow_known12']+[f'joint_z{i}' for i in range(8)]+['joint_posterior_variance'])
        for a,c in KFold(3,shuffle=True,random_state=261026).split(roles['context']):
            assert not set(current.customer_id.iloc[roles['context'][a]])&set(current.customer_id.iloc[roles['context'][c]])
        assert not set(current.customer_id.iloc[roles['context']])&set(current.customer_id.iloc[np.r_[roles['correction'],roles['validation']]])
        rows=np.r_[roles['correction'],roles['validation']];table=x.iloc[rows].reset_index(drop=True)
        head=feature_head.fit(table.iloc[:6000].assign(target=roles['y_correction']),base['raw'][:6000],m['head_columns'],.003,.5)
        actual=.75*b.corrected_probability(head,table.iloc[6000:],base['raw'][6000:])[0]+.25*base['spline']
        error=float(np.max(abs(actual-base['corrected'])));assert error<1e-10,error
        checks.append({'fold':fold,'baseline_replay_max_abs':error,'all90_unchanged':True,'master_columns':len(x.columns),'selected_target_outer_overlap':0})
    gate={'status':'passed_for_declared_development_protocol','UTC':utc(),'checks':checks,
          'known_KS_cases':True,'interval_likelihood_known_case':True,'right_censor_likelihood_known_case':True,
          'feature_freeze_sha256':sha(R/'FEATURE_FREEZE.json'),'plan_sha256':sha(R/'PLAN.json'),
          'shadow_data_gate_sha256':sha(R/'SHADOW_DATA_GATE.json'),'source_sha256':sha(__file__),
          'limits':'No untouched audit; upstream repayment snapshot semantics remain an assumption. Shadow12labels use conservative witnessed bounds; no exact event times claimed.'}
    dump(gate,R/'GATE.json');print(json.dumps({'status':gate['status'],'folds':5,'baseline_max_error':max(c['baseline_replay_max_abs'] for c in checks)}),flush=True)
if __name__=='__main__':main()
