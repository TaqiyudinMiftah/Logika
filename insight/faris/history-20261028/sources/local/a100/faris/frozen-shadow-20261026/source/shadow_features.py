"""Borrower-cross-fitted historical teachers; no selected-loan labels used."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='3'
import argparse,collections,gc,json,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from common import R,root_path,sha,dump,utc

class Weibull(nn.Module):
    def __init__(self,d):
        super().__init__();self.net=nn.Sequential(nn.Linear(d,32),nn.SiLU(),nn.Linear(32,2))
    def forward(self,x):
        z=self.net(x);return torch.exp(z[:,0].clamp(-3,4))*365,torch.nn.functional.softplus(z[:,1])+.2
def interval_loss(scale,shape,lower,upper,event,weight):
    a=(lower.clamp(min=.1)/scale).pow(shape).clamp(max=70)
    b=(upper.clamp(min=.1)/scale).pow(shape).clamp(max=75)
    logdifference=-a+torch.log((-torch.expm1(-(b-a).clamp(min=1e-6))).clamp(min=1e-20))
    loss=torch.where(event,-logdifference,a)
    return (loss*weight).sum()/weight.sum()
def preprocess(h,current,fitrows,columns):
    raw=h[columns].to_numpy(float);query=current[columns].to_numpy(float)
    raw=np.sign(raw)*np.log1p(abs(raw));query=np.sign(query)*np.log1p(abs(query))
    median=np.nanmedian(raw[fitrows],axis=0);median=np.nan_to_num(median,nan=0)
    sd=np.nanstd(raw[fitrows],axis=0);sd=np.maximum(np.nan_to_num(sd,nan=1),.05)
    def apply(x):
        missing=~np.isfinite(x);value=np.where(missing,median,x)
        return np.column_stack([np.clip((value-median)/sd,-8,8),missing.astype(float)]).astype('float32')
    return apply(raw),apply(query)
def fit_predict(h,current,fitrows,predictrows,columns,seed):
    assert len(fitrows)>1000
    torch.manual_seed(seed);torch.set_num_threads(3)
    x,q=preprocess(h,current,fitrows,columns)
    subset=h.iloc[fitrows];counts=subset.customer_id.value_counts();weight=1/subset.customer_id.map(counts).to_numpy(float)
    weight=weight/weight.mean();rng=np.random.default_rng(seed)
    model=Weibull(x.shape[1]);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    tx=torch.from_numpy(x[fitrows]);lo=torch.tensor(subset.lower_days.to_numpy(),dtype=torch.float32)
    up=torch.tensor(np.where(subset.event,subset.upper_days,subset.lower_days),dtype=torch.float32)
    ev=torch.tensor(subset.event.to_numpy(bool));w=torch.tensor(weight,dtype=torch.float32)
    losses=[]
    for epoch in range(35):
        total=0
        for start in range(0,len(tx),2048):
            # Independent permutation each epoch, deterministic under the declared seed.
            if start==0:order=rng.permutation(len(tx))
            ix=order[start:start+2048];scale,shape=model(tx[ix])
            loss=interval_loss(scale,shape,lo[ix],up[ix],ev[ix],w[ix]);assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();total+=float(loss.detach())
        losses.append(total/max(1,int(np.ceil(len(tx)/2048))))
    model.eval()
    with torch.inference_mode():
        scale,shape=model(torch.from_numpy(q[predictrows]));survival_risk=(-torch.expm1(-(365.25/scale).pow(shape))).numpy()
    eligible=fitrows[h.iloc[fitrows].target12.to_numpy()>=0]
    assert h.iloc[eligible].target12.nunique()==2
    classifier=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,
                                  loss_function='Logloss',random_seed=seed,thread_count=3,verbose=False,allow_writing_files=False)
    targets=h.iloc[eligible].target12.to_numpy(int)
    c=h.iloc[eligible].customer_id.value_counts();weights=1/h.iloc[eligible].customer_id.map(c).to_numpy(float)
    classifier.fit(x[eligible],targets,sample_weight=weights/weights.mean())
    class_risk=classifier.predict_proba(q[predictrows])[:,1]
    return survival_risk,class_risk,{'seed':seed,'historical_fit_rows':len(fitrows),'known12_fit_rows':len(eligible),
                                  'customers':subset.customer_id.nunique(),'interval_NLL_first_last':[losses[0],losses[-1]],
                                  'classifier':classifier.get_params(),'fitted_weights_saved':False}
def main(fold):
    start=time.time();h=pd.read_parquet(R/'prepared/historical.parquet');current=pd.read_parquet(R/'prepared/current.parquet')
    roles=np.load(root_path('packet_root')/f'packets/fold{fold}/roles.npz');context=roles['context']
    customers=current.customer_id.iloc[context].to_numpy();assert len(set(customers))==50000
    features=json.loads((R/'SHADOW_DATA_GATE.json').read_text())['teacher_features']
    allowed=h.schedule_complete|h.event
    out=np.full((70000,2),np.nan);recipes=[]
    for split,(trainidx,valididx) in enumerate(KFold(3,shuffle=True,random_state=261026).split(context)):
        train_customers=set(customers[trainidx]);validation_customers=set(customers[valididx]);assert not train_customers&validation_customers
        fitrows=np.flatnonzero(h.customer_id.isin(train_customers)&allowed);predictrows=context[valididx]
        p,c,recipe=fit_predict(h,current,fitrows,predictrows,features,261026+fold*10+split)
        out[predictrows]=np.column_stack([p,c]);recipe['excluded_query_customers']=len(validation_customers);recipe['split']=split;recipes.append(recipe)
    fitrows=np.flatnonzero(h.customer_id.isin(set(customers))&allowed)
    query=np.r_[roles['correction'],roles['validation']]
    assert not set(current.customer_id.iloc[query])&set(h.customer_id.iloc[fitrows])
    p,c,recipe=fit_predict(h,current,fitrows,query,features,261026+fold*10+3)
    out[query]=np.column_stack([p,c]);recipe['split']='full_context';recipes.append(recipe)
    assert np.isfinite(out).all() and out.min()>=0 and out.max()<=1
    dest=R/f'learned/fold{fold}';dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'shadow_interval12':out[:,0],'shadow_known12':out[:,1]}).to_parquet(dest/'shadow.parquet',index=False)
    dump({'status':'complete','fold':fold,'recipes':recipes,'seconds':time.time()-start,'selected_target_labels_used':False,
          'historical_labels_from_outer_query_customers_used':False,'crossfit':'3 borrower folds within50000context; complete context teacher predicts correction and outer',
          'prediction_sha256':sha(dest/'shadow.parquet'),'source_sha256':sha(__file__),'fitted_weights_saved':False},dest/'SHADOW.json')
    print(json.dumps({'fold':fold,'seconds':time.time()-start,'status':'complete'}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);main(p.parse_args().fold)
