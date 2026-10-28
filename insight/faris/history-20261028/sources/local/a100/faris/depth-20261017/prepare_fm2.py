from pathlib import Path
import json,sys,time,subprocess,importlib.metadata
ROOT=Path(__file__).resolve().parent;source=ROOT/'fm2-source';e=ROOT/'fm2-evidence';e.mkdir(exist_ok=True)
for name in ['split.json','eda.json','research.md','rules.md','hypotheses.md','applicability.md']:
 (e/name).write_bytes((ROOT/'evidence'/name).read_bytes())
(e/'research.md').write_text((e/'research.md').read_text()+'''\nNVIDIA Kumo Tabular official model card https://huggingface.co/nvidia/Kumo-Tabular read 2026-10-17. Labeled examples are context; num_estimators controls ensemble. Installed SDK commit 329a707bbcf9ca147aa45a63fef8b2029f4ee37b fit/forward inspected in full. fit caches context without gradient updates; forward in training mode permits gradients. Bounded episodic AdamW updates are an engineering hypothesis, not a claimed NVIDIA fine-tuning recipe.\n''')
(e/'rules.md').write_text((e/'rules.md').read_text()+ '\nLatest user authorization explicitly includes foundation models and staged Kumo Large weights on node380. This supersedes the earlier FM exclusion for this new GPU campaign. No extra training data or paid provisioning.\n')
(e/'applicability.md').write_text('GPU-only Kumo Large using already staged official v1.0.1 weights. CPU feature pipeline is supporting work, not a CPU model search. No new model-weight downloads. TabPFN/other FM excluded because no staged pinned assets and short remaining window. All prior leakage/split boundaries retained. Full 3fold, no partial-fold pruning.\n')
(e/'hypotheses.md').write_text('HFM1: 8192/16384/32768/full outer-training context and 2/4/8 estimators may change Kumo quality. HFM2: promising historical feature views may transfer to Kumo. HFM3: 16/48/120 disjoint training-only episodic AdamW updates at LR 1e-6/5e-6/2e-5 may improve adaptation. Preserve matched zero-update controls. No untouched confirmation; development reuse disclosed.\n')
(source/'requirements.lock').write_text(''.join(f'{n}=={importlib.metadata.version(n)}\n' for n in ['numpy','pandas','scipy','scikit-learn','pyarrow','joblib','torch','structured-data-models','cloudpickle','optuna','psutil']))
# Pin and fingerprint the exact pretrained assets without authorizing any runtime download.
from huggingface_hub import hf_hub_download
assets={name:str(Path(hf_hub_download('nvidia/Kumo-Tabular',name,revision='v1.0.1',local_files_only=True))) for name in ['config.json','large/classifier.pt']}
views=json.loads((source/'views.json').read_text());viewnames=['control']+[v for v,r in views.items() if v.startswith('w2') and r['config']['reducer'] not in ['nmf','pls','ica','group_pca']][:5]
def cat(x):return {'type':'categorical','choices':x}
common={'view':cat(viewnames),'context':cat([8192,16384,32768,56000]),'estimators':cat([2,4,8]),'query_batch':cat([256,1024])}
resources={'cpus':4,'ram_mb':30000,'disk_mb':3000,'timeout_seconds':1800,'gpus':1,'vram_mb':62000}
families=[]
for kind in ['context','adapt']:
 p=dict(common,updates=cat([0] if kind=='context' else [16,48,120]))
 if kind=='adapt':p['lr']=cat([1e-6,5e-6,2e-5])
 anchor={k:v['choices'][0] for k,v in p.items()};anchor.update(view=('w2f_305f821e59f359ca' if kind=='context' else 'control'),context=16384,estimators=2,query_batch=256)
 families.append({'name':'kumo_'+kind,'hypothesis':'HFM1/HFM2/HFM3','parameters':p,'resources':resources,'anchors':[{'params':anchor}]})
c={'version':1,'id':'depth-20261017-node380-fm2','competition_root':str(ROOT),'adapter':'adapter.py','metric':'tie-aware KS','direction':'maximize','inputs':{'bank':str(ROOT/'bank'),**assets},'brief':{'description':'GPU Kumo Large context and gradient adaptation until 06:00 WIB','rules':str(e/'rules.md'),'compute_access':'authorized node380 A100 80GB,16CPU quota','external_data_policy':'none','pretrained_policy':'already cached official nvidia/Kumo-Tabular v1.0.1'},'evidence':{'research':str(e/'research.md'),'eda':str(e/'eda.json'),'hypotheses':str(e/'hypotheses.md'),'applicability':str(e/'applicability.md')},'protocols':{'screen':{'split':str(e/'split.json')}},'families':families,'budget_seconds':max(1,1792191600-time.time()-90),'storage_limit_mb':200000,'max_trials':300,'max_workers':1,'cpu_limit':16,'ram_limit_mb':160000,'seeds':[2026],'search_seed':3805,'reserve_fraction':.04,'breadth_fraction':.3,'coverage_pool_limit':4096,'disk_headroom_mb':10000,'max_retries':0,'family_failure_limit':4,'poll_seconds':1}
(ROOT/'fm-input.json').write_text(json.dumps(c,indent=2));campaign=ROOT/'campaign-fm2'
subprocess.run([sys.executable,str(ROOT/'toolkit/campaign.py'),'bundle','--config',str(ROOT/'fm-input.json'),'--source',str(source),'--output',str(campaign)],check=True)
subprocess.run([sys.executable,str(campaign/'runtime/campaign.py'),'validate',str(campaign)],check=True)
