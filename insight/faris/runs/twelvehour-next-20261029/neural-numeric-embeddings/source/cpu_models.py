"""Bernoulli neural CANN; inner-only selection, fresh full-context refit."""
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

class NumericMap(torch.nn.Module):
    def __init__(self,features,metadata):
        super().__init__();self.kind=metadata['kind'];self.n=int(metadata['numeric_count']);self.features=features
        self.register_buffer('left',torch.tensor(metadata.get('left',[]),dtype=torch.float32))
        self.register_buffer('span',torch.tensor(metadata.get('span',[]),dtype=torch.float32))
        self.register_buffer('valid',torch.tensor(metadata.get('valid',[]),dtype=torch.float32))
        self.frequency=None
        if self.kind.startswith('periodic'):
            self.frequency=torch.nn.Parameter(torch.randn(self.n,4)*float(metadata['frequency_scale']))
        dimension=(int(metadata['bins'])if self.kind.startswith('ple')else 8)if self.kind!='raw'else 1
        self.output_features=features-self.n+self.n*dimension
    def forward(self,x):
        if self.kind=='raw':return x
        numeric=x[:,:self.n];tail=x[:,self.n:]
        if self.kind.startswith('ple'):
            encoded=torch.clamp((numeric[:,:,None]-self.left)/self.span,0.,1.)*self.valid
        else:
            phase=2*torch.pi*numeric[:,:,None]*self.frequency
            encoded=torch.cat([torch.cos(phase),torch.sin(phase)],dim=-1)
        return torch.cat([encoded.flatten(1),tail],dim=1)

def embedding_metadata(x,numeric_count,kind):
    assert kind in {'raw','ple8','ple16','periodic0p1','periodic1p0'}
    d=dict(kind=kind,numeric_count=int(numeric_count),fit_rows=len(x))
    if kind.startswith('ple'):
        bins=int(kind[3:]);q=np.quantile(x[:,:numeric_count].numpy(),np.linspace(0,1,bins+1),axis=0).T
        span=np.diff(q,axis=1);valid=span>1e-8
        d.update(bins=bins,left=q[:,:-1],span=np.where(valid,span,1.),valid=valid.astype(np.float32))
    elif kind.startswith('periodic'):d['frequency_scale']=.1 if kind=='periodic0p1'else 1.
    return d

class Network(torch.nn.Module):
    def __init__(self,features,width,metadata):
        super().__init__();self.embedding=NumericMap(features,metadata)
        self.body=torch.nn.Sequential(torch.nn.Linear(self.embedding.output_features,width),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(width,width),torch.nn.GELU(),torch.nn.Dropout(.1))
        self.head=torch.nn.Linear(width,1)
        torch.nn.init.zeros_(self.head.weight);torch.nn.init.zeros_(self.head.bias)
    def forward(self,x):return self.head(self.body(self.embedding(x))).squeeze(-1)

def model(features,spec,seed,metadata):
    torch.manual_seed(int(seed));return Network(features,int(spec['width']),metadata)

def probabilities(net,x,offset,batch=1024):
    net.eval();output=[]
    with torch.no_grad():
        for start in range(0,len(x),batch):output.append(torch.sigmoid(offset[start:start+batch]+net(x[start:start+batch])).numpy().astype(np.float64))
    return np.concatenate(output)

def epochs(net,x,y,offset,spec,seed,end,number,stop=None):
    optimizer=torch.optim.AdamW(net.parameters(),lr=float(spec['learning_rate']),weight_decay=.01)
    generator=torch.Generator(device='cpu').manual_seed(int(seed)+271828)
    penalty=float(spec['residual_penalty']);best_epoch=0;best_score=None;stale=0;done=0
    if stop is not None:
        xs,ys,ofs=stop;best_score=_ks(ys,probabilities(net,xs,ofs))
    for epoch in range(1,number+1):
        _check(end);net.train();permutation=torch.randperm(len(x),generator=generator)
        for index in permutation.split(512):
            _check(end);optimizer.zero_grad(set_to_none=True);residual=net(x[index])
            loss=torch.nn.functional.binary_cross_entropy_with_logits(offset[index]+residual,y[index])+penalty*torch.mean(residual.square())
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
    assert float(spec['residual_penalty'])in {0.,.1,.5}and float(spec['learning_rate'])in {.001,.0003}
    if mode=='glm':
        matrix=encoder.glm(frame,ctx[inner]);glm,glm_audits['inner']=_glm(matrix,y[inner],C,end)
        inner_offset=glm.decision_function(matrix);stop_offset=glm.decision_function(encoder.glm(frame,ctx[stop]));del matrix,glm
    else:
        intercept=float(logit(y[inner].mean()));inner_offset=np.full(len(inner),intercept);stop_offset=np.full(len(stop),intercept)
    x=dense(encoder,frame,ctx[inner]);xs=dense(encoder,frame,ctx[stop]);of=torch.tensor(inner_offset,dtype=torch.float32);ofs=torch.tensor(stop_offset,dtype=torch.float32)
    embedding=embedding_metadata(x,len(encoder.numeric),spec['encoding']);embedding_hashes={k:_hash(v)for k,v in embedding.items()if isinstance(v,np.ndarray)}
    labels=torch.tensor(y[inner],dtype=torch.float32);net=model(x.shape[1],spec,seed,embedding)
    pilot=epochs(net,x,labels,of,spec,seed,end,120,stop=(xs,y[stop],ofs));selected=int(pilot['selected_epochs'])
    del x,xs,of,ofs,labels,net;gc.collect();_check(end)
    if mode=='glm':
        matrix=encoder.glm(frame,ctx);glm,glm_audits['context']=_glm(matrix,y,C,end)
        context_offset=glm.decision_function(matrix);query_offset=glm.decision_function(encoder.glm(frame,query));del matrix,glm
    else:
        intercept=float(logit(y.mean()));context_offset=np.full(50000,intercept);query_offset=np.full(20000,intercept)
    x=dense(encoder,frame,ctx);labels=torch.tensor(y,dtype=torch.float32);of=torch.tensor(context_offset,dtype=torch.float32)
    net=model(x.shape[1],spec,seed,embedding);full=epochs(net,x,labels,of,spec,seed,end,selected)
    assert full['epochs_completed']==selected
    del x,labels,of;gc.collect();xq=dense(encoder,frame,query);oq=torch.tensor(query_offset,dtype=torch.float32)
    probability=probabilities(net,xq,oq);replay=np.r_[probabilities(net,xq[:64],oq[:64],64),probabilities(net,xq[64:128],oq[64:128],64)]
    error=float(np.max(np.abs(probability[:128]-replay)));assert error<=5e-6
    assert probability.shape==(20000,)and np.isfinite(probability).all()and np.all((probability>=0)&(probability<=1));_check(end)
    audit=dict(model_family='Numeric_embedding_Bernoulli_neural_CANN',numeric_encoding=spec['encoding'],embedding_knots_fit_rows=42500,embedding_fit_rows_sha256=_ids(ctx[inner]),embedding_array_hashes=embedding_hashes,embedding_recipe='PLE cumulative clamped inner-quantile segments, repeated edges masked; or trainable4frequencies numeric sin/cos; retain missing indicators and categorical tails',full_refit_embedding_knots_reused=True,periodic_frequency_refit_fresh=True,existing_feature_bank_mutated=False,offset_mode=mode,formula='sigmoid(frozenGLMlogit+zero_initialized_MLP_residual)',width=int(spec['width']),hidden_layers=2,activation='GELU',dropout=.1,learning_rate=float(spec['learning_rate']),weight_decay=.01,residual_energy_penalty=float(spec['residual_penalty']),batch_size=512,maximum_epochs=120,patience=12,gradient_clip_norm=5.,initial_network_output=0.,zero_epoch_GLM_reversion_allowed=True,pilot=pilot,full_refit=full,selected_epochs=selected,glm_parameters=glm_audits,inner_preprocessor=recipe,full_preprocessor=recipe,preprocessing_refit_on_context=False,context_rows=50000,inner_train_rows=42500,stopping_rows=7500,final_training_rows=50000,inner_train_rows_sha256=_ids(ctx[inner]),stopping_rows_sha256=_ids(ctx[stop]),inner_split_seed=int(spec['inner_split_seed']),seed=int(seed),batch_replay_max_abs_probability_error=error,query_offset_sha256=_hash(query_offset),context_offset_sha256=_hash(context_offset),query_labels_used=False,correction_used_for_early_stopping=False,execution_backend='CPU float32;2threads;CPU RNG',model_initialization_seed=int(seed),shuffle_seed=int(seed)+271828,full_refit_fresh_initialization=True,fitted_weights_saved=False,KV_caches_saved=False,wall_seconds=time.time()-started,sources=['https://arxiv.org/abs/2203.05556v4','https://github.com/yandex-research/rtdl-num-embeddings','https://doi.org/10.1017/asb.2018.42','https://www.mdpi.com/1999-4893/16/2/99','https://docs.pytorch.org/docs/2.7/generated/torch.nn.BCEWithLogitsLoss.html'])
    del net,xq,oq,encoder;gc.collect()
    return probability,audit
