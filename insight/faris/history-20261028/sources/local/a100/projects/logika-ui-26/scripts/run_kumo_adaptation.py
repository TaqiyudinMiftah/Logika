from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import resource
import shutil
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['HF_HOME']=os.environ.get('LOGIKA_HF_HOME',str(ROOT/'artifacts/hf_cache_cpu'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['HF_HUB_DISABLE_XET']='1'

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from huggingface_hub import hf_hub_download
from sdm import TableTensor
from sdm.evaluation import to_class_indices
from sdm.models import KumoTabular
import sdm.processing as sp

from logika_ui_26.experiment_io import cached_features,sha256_file,write_json,runtime_info
from logika_ui_26.experiments import build_folds,classification_metrics
from logika_ui_26.features import CATEGORICAL_FEATURES
from logika_ui_26.kumo import context_indices
from logika_ui_26.kumo_adapters import inject_adapters,frozen_digest,adapter_state,load_adapters


def seed_model(seed,threads):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def tensor(frame,features,device):
    values=frame[features].copy()
    stypes={}
    for name in features:
        if name in CATEGORICAL_FEATURES:
            values[name]=values[name].astype('string').fillna('__MISSING__')
            stypes[name]='categorical'
        else:
            values[name]=pd.to_numeric(values[name]).replace([np.inf,-np.inf],np.nan)
            stypes[name]='numerical'
    previous=torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return TableTensor.from_arrow(pa.Table.from_pandas(values,preserve_index=False),stypes=stypes,device=device)
    finally:
        torch.set_default_dtype(previous)


def targets(values,device):
    return TableTensor.from_arrow(pa.table({'target':pa.array(values,type=pa.int64())}),stypes={'target':'categorical'},device=device)


@torch.inference_mode()
def cached_predict(model,x,y,query,config,seed):
    model.clear()
    model.eval()
    with torch.amp.autocast(config['device'],dtype=torch.bfloat16,enabled=config['device']=='cuda'):
        model.fit(x,y,num_estimators=config['estimators'],estimator_batch_size=1,
                  generator=torch.Generator(device=config['device']).manual_seed(seed))
    result=[]
    for start in range(0,query.size(0),config['query_batch_size']):
        with torch.amp.autocast(config['device'],dtype=torch.bfloat16,enabled=config['device']=='cuda'):
            out=model.predict(query[start:start+config['query_batch_size']])
        result.append(out[['1']].numerical.squeeze(-1).float().cpu().numpy().copy())
    probability=np.concatenate(result).astype(float)
    model.clear()
    assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
    return probability


def step(model,optimizer,x,y,config,rng,seed):
    model.clear()
    model.train()
    count=config['training_context_rows']+config['training_query_rows']
    indices=rng.choice(x.size(0),size=count,replace=False)
    context=torch.from_numpy(indices[:config['training_context_rows']]).to(config['device'])
    query=torch.from_numpy(indices[config['training_context_rows']:]).to(config['device'])
    optimizer.zero_grad(set_to_none=True)
    recipe=model.default_recipe()
    recipe.output=sp.Identity()
    with torch.amp.autocast(config['device'],dtype=torch.bfloat16,enabled=config['device']=='cuda'):
        out=model(x_context=x[context],y_context=y[context],x_query=x[query],
                  num_estimators=1,recipe=recipe,generator=torch.Generator(device=config['device']).manual_seed(seed))[0]
    logits,labels=to_class_indices(out,y[query])
    loss=torch.nn.functional.cross_entropy(logits.float(),labels)
    if not logits.requires_grad or not torch.isfinite(loss):
        raise ValueError('Invalid gradient-bearing adaptation loss')
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum()==0
           for name,p in model.named_parameters() if '.lora_B.' in name):
        raise ValueError('An adapter output factor lacks finite nonzero gradients')
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise ValueError('A frozen parameter received a gradient')
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()),float(norm)


def optimizer_for(model,config):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=config['learning_rate'],weight_decay=config['weight_decay'])


def execute(config,output,pilot):
    seed_model(config['seed'],config['threads'])
    metadata=importlib.metadata.distribution('structured-data-models').read_text('direct_url.json')
    revision='329a707bbcf9ca147aa45a63fef8b2029f4ee37b'
    assert json.loads(metadata)['vcs_info']['commit_id']==revision
    requested_source=ROOT/config['selection_run']
    candidates=[requested_source,ROOT/'artifacts/experiments'/requested_source.name,
                ROOT/'artifacts/a100_runs'/requested_source.name]
    source=next((path for path in candidates if (path/'identity.json').is_file()),None)
    if source is None:
        raise FileNotFoundError('The pinned feature-selector run is unavailable')
    if config.get('selection_identity_sha256'):
        assert sha256_file(source/'identity.json')==config['selection_identity_sha256']
    source_identity=json.loads((source/'identity.json').read_text())
    for name,expected in source_identity['source_hashes'].items():
        assert sha256_file(source/'source'/name)==expected
    training=pd.read_csv(ROOT/'data/raw/train_applications.csv').sort_values('application_id').reset_index(drop=True)
    sample=pd.read_csv(ROOT/'data/raw/sample_submission.csv')
    folds=build_folds(training,config['seed'],5,.2)
    prior_plan=json.loads((source/'plan.json').read_text())
    selections=[]
    for fold,plan in zip(folds,prior_plan,strict=True):
        for key,prior_key in [('development','development_ids'),('inner_validation','inner_validation_ids'),('outer_training','outer_training_ids'),('outer_validation','outer_validation_ids')]:
            assert training.iloc[fold[key]].application_id.tolist()==plan[prior_key]
        folder=source/f"fold_{fold['fold']}"
        complete=json.loads((folder/'complete.json').read_text())
        assert sha256_file(folder/'selection.json')==complete['hashes']['selection.json']
        selection=json.loads((folder/'selection.json').read_text())
        assert selection['fitting_ids']==plan['development_ids']
        assert selection['inner_validation_ids']==plan['inner_validation_ids']
        selections.append(selection)
    columns=list(dict.fromkeys(name for selection in selections for name in selection['features']))
    frame,cache=cached_features(ROOT/'data/raw',ROOT/'data/processed/feature_cache','768MB','credit_v2',columns=['application_id',*columns])
    assert cache['parquet_sha256']==source_identity['feature_hash']
    frame=frame.set_index('application_id')
    assets={}
    for name in ['config.json','large/classifier.pt']:
        path=Path(hf_hub_download('nvidia/Kumo-Tabular',name,revision='v1.0.1',token=False))
        assets[name]=sha256_file(path)
    assert assets==json.loads((source/'fold_1/model.json').read_text())['asset_sha256']
    names=['scripts/run_kumo_adaptation.py','src/logika_ui_26/kumo_adapters.py',
           'src/logika_ui_26/kumo.py','src/logika_ui_26/experiments.py',
           'src/logika_ui_26/experiment_io.py','src/logika_ui_26/features.py',
           'src/logika_ui_26/credit_features.py','src/logika_ui_26/sdk_cache.py','pyproject.toml','uv.lock']
    if config['device']=='cpu':
        names.append('configs/runtime/kumo-cpu-requirements.txt')
    identity={'config':config,'features':columns,'source_hashes':{name:sha256_file(ROOT/name) for name in names},
              'feature_hash':cache['parquet_sha256'],'asset_sha256':assets,'sdk_commit':revision}
    write_json(output/'identity.json',identity)
    write_json(output/'runtime.json',runtime_info())
    write_json(output/'feature_cache.json',cache)
    for name in names:
        destination=output/'source'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
    plans=[{'fold':fold['fold'],**{key:training.iloc[fold[key]].application_id.tolist() for key in ['development','inner_validation','outer_training','outer_validation']}} for fold in folds]
    write_json(output/'folds.json',plans)
    oof=pd.Series(index=training.application_id,dtype=float)
    test_probabilities=[]
    metrics=[]
    for fold,plan,selection in zip(folds[:1] if pilot else folds,plans,selections):
        number=fold['fold']
        folder=output/f'fold_{number}'
        folder.mkdir()
        features=selection['features']
        write_json(folder/'selected_features.json',selection)
        seed_model(config['seed']+number,config['threads'])
        if config['device']=='cuda':
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError('CUDA BF16 is required')
            free,total=torch.cuda.mem_get_info()
            if free<(config['gpu_budget_gib']+1)*1024**3:
                raise RuntimeError('Insufficient free VRAM for the declared budget')
            torch.cuda.set_per_process_memory_fraction(config['gpu_budget_gib']*1024**3/total)
        model=KumoTabular(task='classification',size='large',pretrained=True,device=config['device'])
        if config['device']=='cuda':
            from logika_ui_26.sdk_cache import keep_fit_cache_on_cuda
            keep_fit_cache_on_cuda(model)
        report=inject_adapters(model,config['rank'],config['adapter_layers'])
        initial=adapter_state(model)
        x=tensor(frame.loc[plan['development']],features,config['device'])
        y=targets(training.iloc[fold['development']].target.to_numpy(dtype=np.int64),config['device'])
        context=context_indices(training,fold['development'],config['inner_context_rows'],config['seed']+number)
        context_ids=training.iloc[context].application_id.tolist()
        inner_ids=plan['inner_validation'][:512] if pilot else plan['inner_validation']
        xc=tensor(frame.loc[context_ids],features,config['device'])
        yc=targets(training.iloc[context].target.to_numpy(dtype=np.int64),config['device'])
        xi=tensor(frame.loc[inner_ids],features,config['device'])
        inner_target=training.set_index('application_id').loc[inner_ids,'target'].to_numpy(dtype=np.int64)
        from sklearn.metrics import roc_auc_score
        probability=cached_predict(model,xc,yc,xi,config,config['seed']+number)
        baseline=float(roc_auc_score(inner_target,probability))
        best_score,best_step,best_state=baseline,0,initial
        history=[{'step':0,'inner_auc':baseline}]
        optimizer=optimizer_for(model,config)
        rng=np.random.default_rng(config['seed']+number)
        stale=0
        started=time.monotonic()
        for count in range(1,(4 if pilot else config['max_steps'])+1):
            loss,norm=step(model,optimizer,x,y,config,rng,config['seed']+number+count)
            record={'step':count,'loss':loss,'gradient_norm':norm,'elapsed':time.monotonic()-started}
            if count%config['evaluation_every']==0 or pilot:
                p=cached_predict(model,xc,yc,xi,config,config['seed']+number)
                score=float(roc_auc_score(inner_target,p))
                record['inner_auc']=score
                if score>best_score:
                    best_score,best_step,best_state=score,count,adapter_state(model)
                    stale=0
                else:
                    stale+=1
            history.append(record)
            print(json.dumps({'fold':number,**record}),flush=True)
            if not pilot and stale>=config['patience_evaluations']:
                break
        assert frozen_digest(model)==report['frozen_signature']
        write_json(folder/'inner_history.json',history)
        write_json(folder/'inner_partitions.json',{'gradient_pool_ids':plan['development'],'context_ids':context_ids,'validation_ids':inner_ids})
        if pilot:
            final_state=adapter_state(model)
            torch.save(final_state,folder/'adapter.pt')
            load_adapters(model,initial)
            np.testing.assert_array_equal(probability,cached_predict(model,xc,yc,xi,config,config['seed']+number))
            load_adapters(model,torch.load(folder/'adapter.pt',weights_only=True,map_location='cpu'))
            np.testing.assert_array_equal(p,cached_predict(model,xc,yc,xi,config,config['seed']+number))
            write_json(folder/'model.json',{'adapter':report,'base_weights_unchanged':True,'asset_sha256':assets,'best_step':best_step,'pilot':True})
            write_json(output/'result.json',{'pilot':True,'qualified':False,'seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'steps':4,'adapter_checkpoint_exact':True,'base_weights_unchanged':True,'inner_query_rows':len(inner_ids),'training_context_rows':config['training_context_rows'],'training_query_rows':config['training_query_rows'],'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved() if config['device']=='cuda' else None})
            write_json(output/'status.json',{'state':'completed','pilot':True})
            return
        load_adapters(model,initial)
        seed_model(config['seed']+number,config['threads'])
        del optimizer,x,y,xc,yc,xi
        gc.collect()
        x=tensor(frame.loc[plan['outer_training']],features,config['device'])
        y=targets(training.iloc[fold['outer_training']].target.to_numpy(dtype=np.int64),config['device'])
        optimizer=optimizer_for(model,config)
        rng=np.random.default_rng(config['seed']+number+10000)
        outer_history=[]
        for count in range(1,best_step+1):
            loss,norm=step(model,optimizer,x,y,config,rng,config['seed']+number+count)
            outer_history.append({'step':count,'loss':loss,'gradient_norm':norm})
        assert frozen_digest(model)==report['frozen_signature']
        context=context_indices(training,fold['outer_training'],config['final_context_rows'],config['seed']+number)
        context_ids=training.iloc[context].application_id.tolist()
        assert not set(context_ids)&set(plan['outer_validation'])
        xc=tensor(frame.loc[context_ids],features,config['device'])
        yc=targets(training.iloc[context].target.to_numpy(dtype=np.int64),config['device'])
        query_ids=plan['outer_validation']+sample.application_id.tolist()
        xq=tensor(frame.loc[query_ids],features,config['device'])
        probabilities=cached_predict(model,xc,yc,xq,config,config['seed']+number)
        outer_probability=probabilities[:len(plan['outer_validation'])]
        test_probability=probabilities[len(plan['outer_validation']):]
        oof.loc[plan['outer_validation']]=outer_probability
        test_probabilities.append(test_probability)
        score=classification_metrics(training.iloc[fold['outer_validation']].target,outer_probability)
        metric={'fold':number,'selected_steps':best_step,'inner_auc':best_score,'inner_frozen_auc':baseline,'outer':score}
        metrics.append(metric)
        torch.save(adapter_state(model),folder/'adapter.pt')
        write_json(folder/'context_ids.json',{'ids':context_ids,'gradient_pool_ids':plan['outer_training']})
        write_json(folder/'outer_history.json',outer_history)
        write_json(folder/'metrics.json',metric)
        write_json(folder/'model.json',{'adapter':report,'base_weights_unchanged':True,'asset_sha256':assets,'selected_steps':best_step,'precision':'bfloat16' if config['device']=='cuda' else 'float32','estimators':config['estimators']})
        pd.DataFrame({'application_id':plan['outer_validation'],'default_probability':outer_probability}).to_csv(folder/'outer.csv',index=False)
        pd.DataFrame({'application_id':sample.application_id,'default_probability':test_probability}).to_csv(folder/'test.csv',index=False)
        files=['adapter.pt','context_ids.json','selected_features.json','inner_partitions.json','inner_history.json','outer_history.json','metrics.json','model.json','outer.csv','test.csv']
        write_json(folder/'manifest.json',{'identity':identity,'checksums':{name:sha256_file(folder/name) for name in files}})
        print(json.dumps(metric),flush=True)
        write_json(output/'status.json',{'state':'running','pilot':False,'folds':metrics})
        del model,optimizer,x,y,xc,yc,xq
        gc.collect()
    assert oof.notna().all()
    result=training.set_index('application_id')[['target','decision_datetime']].copy()
    result['default_probability']=oof
    result.to_csv(output/'oof.csv',index_label='application_id')
    pd.DataFrame({'application_id':sample.application_id,'default_probability':np.mean(test_probabilities,axis=0)}).to_csv(output/'submission.csv',index=False,float_format='%.10f')
    write_json(output/'result.json',{'run_id':output.name,'model':'kumo_lora','feature_count':120,'pooled_oof':classification_metrics(result.target,result.default_probability),'mean_fold_ks':float(np.mean([m['outer']['ks'] for m in metrics])),'fold_ks_std':float(np.std([m['outer']['ks'] for m in metrics],ddof=1)),'checksums':{name:sha256_file(output/name) for name in ['oof.csv','submission.csv']}})
    write_json(output/'status.json',{'state':'completed','pilot':False})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'configs/experiments/e060_kumo_lora_cpu.json')
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if config['device'] not in {'cpu','cuda'} or config['size']!='large':
        raise ValueError('This declared experiment requires CPU or CUDA Large')
    name=config['name']+('-pilot' if args.pilot else '')+'-'+datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    output=ROOT/'artifacts/experiments'/name
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',config)
    write_json(output/'status.json',{'state':'running','pilot':args.pilot})
    print('Kumo adaptation:',output,flush=True)
    try:
        execute(config,output,args.pilot)
    except BaseException as error:
        write_json(output/'status.json',{'state':'failed','pilot':args.pilot,'error':str(error)})
        raise


if __name__=='__main__':
    main()
