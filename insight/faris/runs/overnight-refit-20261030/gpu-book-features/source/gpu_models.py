"""Full-size held-meta/outer predictions using inner-only book numerical maps."""
import time,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from common import R,load,sha,ids_sha
from book_features import BookInput
from gpu_kumo_engine import predict
from context_recipe import sample_context as draw_context

def sample_context(y,ratio,strategy,seed):
 # Compatibility for the inherited data-gate case-control checks, not our model sampler.
 y=np.asarray(y,int);rng=np.random.default_rng(int(seed)+49999);positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0);n=len(positive)*(1+int(ratio));assert n<=len(y)
 selected=np.r_[positive,rng.choice(negative,len(positive)*int(ratio),replace=False)]if strategy=='case_control'else rng.choice(len(y),n,replace=False)
 return np.sort(selected).astype(np.int64),int(seed)+49999

def train_predict(frame,ctx,y,query,spec,seed,device='cuda'):
 assert device=='cuda';started=time.time();ctx=np.asarray(ctx,np.int64);query=np.asarray(query,np.int64);y=np.asarray(y,int);assert len(ctx)==len(y)==50000 and len(query)==20000
 inner,stop=train_test_split(np.arange(50000),test_size=.15,stratify=y,random_state=int(spec['inner_split_seed']));encoder=BookInput(mode=spec['book_transform']).fit(frame,ctx[inner]);recipe=encoder.recipe();transformed=frame.copy();mapped=encoder._maps(encoder._numeric(frame,np.arange(len(frame))))
 for j,c in enumerate(encoder.numeric):transformed[c]=mapped[:,j].astype('float32')
 selected,draw=draw_context(y,int(spec['context_count']),seed,0);size=spec['size'];checkpoint=load('ENVIRONMENT.json')['checkpoint_'+size];ref=load('CHECKPOINTS.json')[size];expected=ref if isinstance(ref,str)else ref['sha256']
 raw,audit=predict(transformed.iloc[ctx[selected]],transformed.iloc[query],list(frame),y[selected],checkpoint,seed,engine='concat_context_vocab',estimators=int(spec['estimators']),pred_batch=256,mem_fraction=.7,size=size,expected_sha=expected)
 assert time.time()<float(spec['deadline_epoch']);audit.update(book_transform=spec['book_transform'],inner_preprocessor=recipe,inner_train_rows_sha256=ids_sha(ctx[inner]),stopping_rows_sha256=ids_sha(ctx[stop]),context_count=len(selected),context_draw_seed=draw,selected_rows_sha256=ids_sha(ctx[selected]),no_query_labels=True,execution_backend='CUDA AMP; existing sealed checkpoint',wall_seconds=time.time()-started)
 return raw,audit
