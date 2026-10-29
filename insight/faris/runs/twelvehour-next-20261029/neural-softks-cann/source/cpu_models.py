"""Hybrid BCE and smooth signed-KS reward in a fixed-offset CANN.
A logistic-CDF adaptation motivated by direct-KS literature, not exact DMKS/KSNet.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import gc,time
import numpy as np
from scipy.special import logit
from sklearn.model_selection import train_test_split
import torch
from cpu_offset import FrozenInput,_glm,_ids,_hash,_ks,_check

def dense(encoder,frame,rows):
    x=encoder.glm(frame,rows).toarray().astype(np.float32)
    assert np.isfinite(x).all()
    return torch.from_numpy(x)

class Network(torch.nn.Module):
    def __init__(self,features,width,threshold):
        super().__init__()
        self.body=torch.nn.Sequential(torch.nn.Linear(features,width),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(width,width),torch.nn.GELU(),torch.nn.Dropout(.1))
        self.head=torch.nn.Linear(width,1)
        torch.nn.init.zeros_(self.head.weight);torch.nn.init.zeros_(self.head.bias)
        self.threshold=torch.nn.Parameter(torch.tensor(float(threshold),dtype=torch.float32))
    def forward(self,x):return self.head(self.body(x)).squeeze(-1)

def model(features,spec,seed,threshold):
    torch.manual_seed(int(seed));return Network(features,int(spec['width']),threshold)

def probabilities(net,x,offset,batch=1024):
    net.eval();output=[]
    with torch.no_grad():
        for start in range(0,len(x),batch):output.append(torch.sigmoid(offset[start:start+batch]+net(x[start:start+batch])).numpy().astype(np.float64))
    return np.concatenate(output)

def ks_reward(logits,labels,threshold,temperature,prevalence):
    # Inverse population-class proportions give an unbiased minibatch estimate
    # of E[sigmoid|positive]-E[sigmoid|negative], including single-class batches.
    gate=torch.sigmoid((logits-threshold)/temperature)
    return torch.mean(gate*(labels/prevalence-(1-labels)/(1-prevalence)))

def initial_threshold(labels,offset):
    from sklearn.metrics import roc_curve
    fpr,tpr,thresholds=roc_curve(np.asarray(labels,np.int64),np.asarray(offset,np.float64),drop_intermediate=False)
    valid=np.isfinite(thresholds)
    return float(thresholds[valid][np.argmax((tpr-fpr)[valid])])

def epochs(net,x,y,offset,spec,seed,end,number,stop=None):
    optimizer=torch.optim.AdamW([{'params':list(net.body.parameters())+list(net.head.parameters()),'weight_decay':.01},{'params':[net.threshold],'weight_decay':0.}],lr=float(spec['learning_rate']))
    prevalence=float(y.mean());assert 0.<prevalence<1.
    generator=torch.Generator(device='cpu').manual_seed(int(seed)+271828)
    penalty=float(spec['residual_penalty']);best_epoch=0;best_score=None;stale=0;done=0
    if stop is not None:
        xs,ys,ofs=stop;best_score=_ks(ys,probabilities(net,xs,ofs))
    for epoch in range(1,number+1):
        _check(end);net.train();permutation=torch.randperm(len(x),generator=generator)
        for index in permutation.split(512):
            _check(end);optimizer.zero_grad(set_to_none=True);residual=net(x[index])
            logits=offset[index]+residual
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y[index])+penalty*torch.mean(residual.square())
            if float(spec['ks_weight'])>0:loss=loss-float(spec['ks_weight'])*ks_reward(logits,y[index],net.threshold,float(spec['ks_temperature']),prevalence)
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        done=epoch
        if stop is not None:
            score=_ks(ys,probabilities(net,xs,ofs))
            if score>best_score+1e-12:best_score=score;best_epoch=epoch;stale=0
            else:stale+=1
            if stale>=12:break
    del optimizer
    return dict(epochs_completed=done,selected_epochs=best_epoch if stop is not None else done,best_stopping_KS=best_score)

def train_predict(frame,ctx,y,query,spec,seed,device='cpu'):
    started=time.time();assert device=='cpu'and os.environ['CUDA_VISIBLE_DEVICES']==''
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    end=min(float(spec['deadline_epoch']),started+float(spec['max_seconds']))
    ctx=np.asarray(ctx,np.int64);query=np.asarray(query,np.int64);y=np.asarray(y,np.int64)
    assert (len(ctx),len(y),len(query))==(50000,50000,20000)and not np.intersect1d(ctx,query).size
    inner,stop=train_test_split(np.arange(50000),test_size=.15,stratify=y,random_state=int(spec['inner_split_seed']))
    encoder=FrozenInput().fit(frame,ctx[inner]);recipe=encoder.recipe();mode=spec['offset_mode'];C=float(spec['glm_C']);glm_audits={}
    assert mode in {'glm','intercept'}and int(spec['width'])in {64,128}
    assert float(spec['residual_penalty'])==.1 and float(spec['learning_rate'])==.0003
    assert float(spec['ks_weight'])in {.1,.5}and float(spec['ks_temperature'])in {.25,1.}
    if mode=='glm':
        matrix=encoder.glm(frame,ctx[inner]);glm,glm_audits['inner']=_glm(matrix,y[inner],C,end)
        inner_offset=glm.decision_function(matrix);stop_offset=glm.decision_function(encoder.glm(frame,ctx[stop]));del matrix,glm
    else:
        intercept=float(logit(y[inner].mean()));inner_offset=np.full(len(inner),intercept);stop_offset=np.full(len(stop),intercept)
    x=dense(encoder,frame,ctx[inner]);xs=dense(encoder,frame,ctx[stop]);of=torch.tensor(inner_offset,dtype=torch.float32);ofs=torch.tensor(stop_offset,dtype=torch.float32)
    threshold=initial_threshold(y[inner],inner_offset);threshold_hash=_hash(np.asarray([threshold],dtype=np.float32))
    labels=torch.tensor(y[inner],dtype=torch.float32);net=model(x.shape[1],spec,seed,threshold)
    pilot=epochs(net,x,labels,of,spec,seed,end,120,stop=(xs,y[stop],ofs));selected=int(pilot['selected_epochs'])
    del x,xs,of,ofs,labels,net;gc.collect();_check(end)
    if mode=='glm':
        matrix=encoder.glm(frame,ctx);glm,glm_audits['context']=_glm(matrix,y,C,end)
        context_offset=glm.decision_function(matrix);query_offset=glm.decision_function(encoder.glm(frame,query));del matrix,glm
    else:
        intercept=float(logit(y.mean()));context_offset=np.full(50000,intercept);query_offset=np.full(20000,intercept)
    x=dense(encoder,frame,ctx);labels=torch.tensor(y,dtype=torch.float32);of=torch.tensor(context_offset,dtype=torch.float32)
    net=model(x.shape[1],spec,seed,threshold);full=epochs(net,x,labels,of,spec,seed,end,selected)
    assert full['epochs_completed']==selected
    del x,labels,of;gc.collect();xq=dense(encoder,frame,query);oq=torch.tensor(query_offset,dtype=torch.float32)
    probability=probabilities(net,xq,oq);replay=np.r_[probabilities(net,xq[:64],oq[:64],64),probabilities(net,xq[64:128],oq[64:128],64)]
    error=float(np.max(np.abs(probability[:128]-replay)));assert error<=5e-6
    assert probability.shape==(20000,)and np.isfinite(probability).all()and np.all((probability>=0)&(probability<=1));_check(end)
    audit=dict(model_family='Hybrid_smoothKS_Bernoulli_neural_CANN',ks_weight=float(spec['ks_weight']),ks_temperature=float(spec['ks_temperature']),ks_formula='mean(sigmoid((logit-threshold)/temperature)*(y/prevalence-(1-y)/(1-prevalence)))',loss='BCE+.1mean(residual^2)-lambda*signed_smoothKS',trainable_threshold=True,threshold_prediction_role=False,threshold_initialization='finite empirical signedKS argmax on42500innerGLM/intercept logits',threshold_initialization_sha256=threshold_hash,threshold_full_refit_reuses_inner_initialization=True,threshold_weight_decay=0.,threshold_weights_saved=False,exact_DMKS_or_KSNet_reproduction=False,offset_mode=mode,formula='sigmoid(frozenGLMlogit+zero_initialized_MLP_residual)',width=int(spec['width']),hidden_layers=2,activation='GELU',dropout=.1,learning_rate=float(spec['learning_rate']),weight_decay=.01,residual_energy_penalty=float(spec['residual_penalty']),batch_size=512,maximum_epochs=120,patience=12,gradient_clip_norm=5.,initial_network_output=0.,zero_epoch_GLM_reversion_allowed=True,pilot=pilot,full_refit=full,selected_epochs=selected,glm_parameters=glm_audits,inner_preprocessor=recipe,full_preprocessor=recipe,preprocessing_refit_on_context=False,context_rows=50000,inner_train_rows=42500,stopping_rows=7500,final_training_rows=50000,inner_train_rows_sha256=_ids(ctx[inner]),stopping_rows_sha256=_ids(ctx[stop]),inner_split_seed=int(spec['inner_split_seed']),seed=int(seed),batch_replay_max_abs_probability_error=error,query_offset_sha256=_hash(query_offset),context_offset_sha256=_hash(context_offset),query_labels_used=False,correction_used_for_early_stopping=False,execution_backend='CPU float32;2threads;CPU RNG',model_initialization_seed=int(seed),shuffle_seed=int(seed)+271828,full_refit_fresh_initialization=True,fitted_weights_saved=False,KV_caches_saved=False,wall_seconds=time.time()-started,sources=['https://doi.org/10.1016/j.csda.2018.10.004','https://doi.org/10.1080/10485252.2026.2638780','https://doi.org/10.1017/asb.2018.42','https://www.mdpi.com/1999-4893/16/2/99','https://docs.pytorch.org/docs/2.7/generated/torch.nn.BCEWithLogitsLoss.html'])
    del net,xq,oq,encoder;gc.collect()
    return probability,audit
