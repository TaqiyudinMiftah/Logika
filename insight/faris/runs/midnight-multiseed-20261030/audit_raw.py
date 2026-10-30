"""Check actual prediction bytes and training roles before opening new outer scores."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,hashlib,time,collections
import numpy as np
from sklearn.model_selection import train_test_split
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ids=lambda x:hashlib.sha256(np.asarray(x,dtype='<i8').tobytes()).hexdigest()
C=json.loads((R/'CLOCK.json').read_text());config=json.loads((R/'CONFIG.json').read_text());seal=json.loads((R/'SOURCE_SEAL.json').read_text())
for n,h in seal.items():assert sha(R/n)==h,n
packets={};split_cache={};coverage=np.zeros(70000,int)
for f in range(5):
    p=R/f'packets/fold{f}';m=json.loads((p/'MANIFEST.json').read_text())
    for n,h in m['files'].items():assert sha(p/n)==h,(f,n)
    z=np.load(p/'roles.npz');ctx=z['context'];cor=z['correction'];val=z['validation']
    assert len(ctx)==50000 and len(cor)==6000 and len(val)==14000
    assert np.array_equal(np.sort(np.r_[ctx,cor,val]),np.arange(70000));coverage[val]+=1
    yprod=np.load(p/'y.npy');assert len(yprod)==70000
    assert np.array_equal(yprod[ctx],z['y_context']) and np.array_equal(yprod[cor],z['y_correction'])
    assert len(m['views']['actuarial99'])==len(m['views']['global99'])==99
    testids=np.load(p/'test_ids.npy').astype(str);assert len(testids)==len(set(testids))==30000
    for phase in ['validation','production']:
        context=ctx if phase=='validation' else np.arange(70000)
        y=z['y_context'] if phase=='validation' else yprod
        inner,stop=train_test_split(np.arange(len(context)),test_size=.15,stratify=y,random_state=20261029+f)
        split_cache[f,phase]=(ids(context[inner]),ids(context[stop]),len(inner),len(stop))
    packets[f]=(z,m,testids)
assert np.all(coverage==1)
summary=collections.Counter();records=[];failures=[]
for p in sorted((R/'runs').glob('*/*')):
    if not(p/'RESULT.json').exists():
        if(p/'FAILURE.json').exists():failures.append(str(p.relative_to(R)))
        continue
    j=json.loads((p/'RESULT.json').read_text());s=json.loads((p/'START.json').read_text());a=j['audit'];f=s['fold'];kind=s['kind'];phase=s['phase'];seed=s['base_seed'];z,m,_=packets[f]
    assert j['status']=='complete' and j['kind']==kind and j['base_seed']==seed and j['fold']==f and j['phase']==phase
    assert seed+f==s['seed'] and s['fresh_training'] and s['query_labels_loaded'] is False
    assert sha(p/'prediction.npz')==j['prediction_sha256'] and sha(p/'START.json')==j['start_sha256']
    assert s['config_sha256']==sha(R/'CONFIG.json') and s['clock_sha256']==sha(R/'CLOCK.json') and s['source_seal_sha256']==sha(R/'SOURCE_SEAL.json') and s['source_sha256']==sha(R/'worker.py')
    assert s['packet_sha256']==sha(R/f'packets/fold{f}/MANIFEST.json')
    context=z['context'] if phase=='validation' else np.arange(70000)
    query=np.r_[z['correction'],z['validation']] if phase=='validation' else np.arange(70000,100000)
    assert s['context_sha256']==ids(context) and s['query_sha256']==ids(query)
    assert s['context_rows']==len(context) and s['query_rows']==len(query)
    q=np.load(p/'prediction.npz');assert set(q.files)=={'rows','raw'} and np.array_equal(q['rows'],query)
    v=q['raw'];assert v.shape==query.shape and np.isfinite(v).all() and np.all((v>=0)&(v<=1))
    assert s['epoch']<C['admission_end_epoch'] and s['epoch']+j['seconds']<C['model_deadline_epoch']
    assert j['fitted_weights_saved'] is False and j['KV_cache_saved'] is False
    if phase=='validation':assert v.dtype==np.float64
    if kind=='kumo70':
        assert a['first_batch_replay_error']<1e-7 and a['num_estimators']==2 and a['seed']==seed+f
        assert a['pretrained_checkpoint_sha256']=='cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613'
        assert a['engine']=='concat_context_vocab'
        if phase=='production':assert v.dtype==np.float32
    else:
        assert a['batch_replay_max_abs_probability_error']<=5e-6 and a['query_labels_used'] is False and a['seed']==seed+f
        hinner,hstop,ninner,nstop=split_cache[f,phase]
        if kind.startswith('transformer_'):
            assert a['encoder_fit_rows_sha256']==hinner and a['encoder_fit_rows']==ninner and a['stop_rows']==nstop
            assert a['final_refit_rows']==len(context)
            assert a['model_family']==('FTinspiredLocalGLMcoeff' if kind=='transformer_coefficient' else 'FTinspiredCLScontrol')
        else:
            assert a['inner_train_rows_sha256']==hinner and a['stopping_rows_sha256']==hstop
            assert a['inner_train_rows']==ninner and a['stopping_rows']==nstop and a['final_training_rows']==len(context)
            assert a['correction_used_for_early_stopping'] is False and a['full_refit_fresh_initialization']
            assert a['offset_mode']=='intercept' and a['learning_rate']==.0003 and a['dropout']==.1 and a['weight_decay']==.01
            assert a['width']==(64 if kind=='localglm' else 128) and a['residual_energy_penalty']==(.1 if kind=='localglm' else 0.)
            assert a['full_refit']['epochs_completed']==a['selected_epochs'] and a['selected_epochs']==a['pilot']['selected_epochs']
    if s['reference'] and kind!='kumo70':assert j['reference_max_abs']<=5e-6
    summary[(kind,phase,bool(s['reference']))]+=1
    records.append(dict(path=str(p.relative_to(R)),prediction_sha256=j['prediction_sha256'],start_sha256=j['start_sha256'],seconds=j['seconds']))
for p in (R/'runs').rglob('*'):
    assert p.suffix not in {'.pt','.pth','.pkl','.pickle','.joblib','.bin'},p
eligible=[seed for seed in config['new_seeds'] if all((R/f'runs/{k}__s{seed}__f{f}/{phase}/RESULT.json').exists() for k in config['kinds'] for f in range(5) for phase in config['phases'])]
report=dict(status='PASS',epoch=time.time(),completed_fits=len(records),counts=[dict(kind=k,phase=p,reference=r,count=n) for (k,p,r),n in sorted(summary.items())],all_prediction_bytes_rows_seeds_and_roles_verified=True,all_packet_and_source_hashes_verified=True,inner_stopping_boundaries_reconstructed=True,no_new_outer_scores_computed=True,eligible_new_seeds=eligible,retained_failed_attempts=failures,records=records)
(R/'PRE_EVALUATION_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
