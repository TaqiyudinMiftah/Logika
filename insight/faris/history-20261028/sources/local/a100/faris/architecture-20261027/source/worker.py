"""Fit architecture on context only; choose epochs inside context; never load outer labels."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import argparse,copy,gc,hashlib,importlib.metadata,json,resource,socket,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from common import R,dump,sha,utc,ks
from architectures import Preprocessor,make_model,training_loss


def predict(model,numeric,cat,rows,batch_size=512):
    model.eval();pred=[]
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            logits=model(numeric[rows[start:start+batch_size]],cat[rows[start:start+batch_size]])
            if logits.ndim==1:logits=logits[:,None]
            pred.append(torch.sigmoid(logits).mean(1).float().cpu().numpy())
    return np.concatenate(pred)


def fit_epochs(model,numeric,cat,rows,y,spec,seed,stop_rows=None,stop_y=None,fixed_epochs=None):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    # Preserve paper practice of no weight decay on normalization, biases or tokenizer embeddings.
    decay=[];no_decay=[]
    for name,p in model.named_parameters():
        (no_decay if p.ndim<2 or any(s in name for s in ['embedding','numeric_weight','numeric_bias','cat_bias','cls'])else decay).append(p)
    opt=torch.optim.AdamW([{'params':decay,'weight_decay':spec['weight_decay']},{'params':no_decay,'weight_decay':0}],lr=spec['lr'])
    labels=torch.tensor(y,dtype=torch.float32,device=numeric.device);best=-float('inf');best_epoch=0;stale=0;best_state=None;trace=[]
    limit=fixed_epochs or spec['max_epochs']
    for epoch in range(1,limit+1):
        model.train();order=rng.permutation(len(rows));loss_sum=0.
        for start in range(0,len(order),spec['batch_size']):
            ids=order[start:start+spec['batch_size']];logits=model(numeric[rows[ids]],cat[rows[ids]])
            loss=training_loss(logits,labels[ids],spec['pair_weight']);assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();loss_sum+=float(loss.detach())*len(ids)
        item={'epoch':epoch,'training_loss':loss_sum/len(rows)}
        if stop_rows is not None:
            p=predict(model,numeric,cat,stop_rows);score=ks(stop_y,p);item['internal_stop_KS']=score
            if score>best+1e-6:
                best=score;best_epoch=epoch;stale=0
                # In-memory state only. User explicitly requests no serialized fitted weights.
                best_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
            else:stale+=1
        trace.append(item)
        if epoch==1 or epoch%5==0:print(json.dumps({'stage':'inner_epoch'if stop_rows is not None else 'refit_epoch',**item}),flush=True)
        if stop_rows is not None and epoch>=spec['min_epochs']and stale>=spec['patience']:break
    if stop_rows is not None:
        assert best_epoch>0 and best_state is not None
        del best_state
    return best_epoch if stop_rows is not None else limit,trace


def run(config,fold,pilot=False):
    started=time.time();spec=json.loads((R/'CONFIGS.json').read_text())[config].copy();dest=R/('pilots'if pilot else 'runs')/config/f'fold{fold}';dest.mkdir(parents=True,exist_ok=False)
    try:
        if pilot:spec.update(max_epochs=3,min_epochs=3,patience=3)
        paths=json.loads((R/'PATHS.json').read_text());root=Path(paths['frozen_root'])/f'fold{fold}';m=json.loads((root/'MANIFEST.json').read_text())
        assert sha(root/'features.parquet')==m['features_sha256'];assert sha(root/'roles.npz')==m['roles_sha256'];assert sha(R/'FEATURE_FREEZE.json')=='071c75e0f59cf92429bde9c76a64e27931e1eea03bbbbeac2e2589b5fee68dd6'
        roles=np.load(root/'roles.npz');context=roles['context'];query=np.r_[roles['correction'],roles['validation']]
        frame=pd.read_parquet(root/'features.parquet')[m['views'][spec['view']]];pre=Preprocessor().fit(frame,context);z,c=pre.transform(frame)
        assert not np.intersect1d(context,query).size and len(context)==50000 and len(query)==20000
        a,b=train_test_split(np.arange(len(context)),test_size=spec['early_stop_fraction'],stratify=roles['y_context'],random_state=spec['inner_split_seed']+fold)
        assert not np.intersect1d(a,b).size and len(a)+len(b)==50000
        seed=spec['seed']+fold;torch.set_num_threads(2);device='cuda'if torch.cuda.is_available()else 'cpu';torch.manual_seed(seed)
        if device=='cuda':torch.cuda.set_per_process_memory_fraction(.7);torch.cuda.manual_seed_all(seed)
        numeric=torch.tensor(z,device=device);cat=torch.tensor(c,device=device)
        record={'status':'running','config_id':config,'config':spec,'fold':fold,'seed':seed,'host':socket.gethostname(),'device':device,'pilot':pilot,
                'source_sha256':sha(__file__),'architecture_source_sha256':sha(R/'source/architectures.py'),'feature_freeze_sha256':sha(R/'FEATURE_FREEZE.json'),
                'features_sha256':m['features_sha256'],'roles_sha256':m['roles_sha256'],'columns':list(frame),'preprocessing':pre.recipe(),
                'context_rows_sha256':hashlib.sha256(context.tobytes()).hexdigest(),'inner_train_rows_sha256':hashlib.sha256(context[a].tobytes()).hexdigest(),
                'inner_stop_rows_sha256':hashlib.sha256(context[b].tobytes()).hexdigest(),'training_rows':len(a),'inner_stop_rows':len(b),'refit_rows':len(context),
                'early_stop_labels':'context only','correction_labels_used_for_training':False,'outer_labels_loaded':False,
                'environment':{k:importlib.metadata.version(k)for k in ['torch','numpy','pandas','scikit-learn','tabm']},'UTC':utc()}
        if (R/'CLOCK.json').exists():record['clock_sha256']=sha(R/'CLOCK.json')
        dump(record,dest/'START.json')
        model=make_model(z.shape[1],pre.cards,spec).to(device)
        chosen,trace=fit_epochs(model,numeric,cat,context[a],roles['y_context'][a],spec,seed,context[b],roles['y_context'][b])
        del model;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
        torch.manual_seed(seed);model=make_model(z.shape[1],pre.cards,spec).to(device)
        _,refit_trace=fit_epochs(model,numeric,cat,context,roles['y_context'],spec,seed,fixed_epochs=chosen)
        raw=predict(model,numeric,cat,query);again=predict(model,numeric,cat,query[:128]);assert np.max(abs(raw[:128]-again))<1e-6
        assert raw.shape==(20000,)and np.isfinite(raw).all()and 0<=raw.min()<=raw.max()<=1
        np.savez_compressed(dest/'prediction.npz',rows=query,raw=raw)
        dump({'internal_stop_trace':trace,'refit_trace':refit_trace,'selected_epochs':chosen,'weights_saved':False},dest/'TRAINING.json')
        dump({'status':'complete','seconds':time.time()-started,'selected_epochs':chosen,'prediction_sha256':sha(dest/'prediction.npz'),
              'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'peak_CUDA_bytes':torch.cuda.max_memory_allocated()if device=='cuda'else 0,
              'parameter_count':sum(p.numel()for p in model.parameters()),'device':device,'UTC':utc(),'fitted_weights_saved':False,
              'outer_labels_loaded':False,'prediction_repeat_max_abs':float(np.max(abs(raw[:128]-again)))},dest/'RESULT.json')
        print(json.dumps({'status':'complete','config':config,'fold':fold,'selected_epochs':chosen,'seconds':time.time()-started}),flush=True)
    except BaseException:dump({'status':'failed','seconds':time.time()-started,'error':traceback.format_exc(),'UTC':utc()},dest/'FAILURE.json');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--fold',type=int,required=True);parser.add_argument('--pilot',action='store_true');args=parser.parse_args();run(args.config,args.fold,args.pilot)
