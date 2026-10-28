"""GPU-only Kumo Large context / fine-tuning search on frozen development folds."""
import json,os,time,gc,hashlib
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
import numpy as np
import joblib
from metaexp.features import Bank
from metaexp.pipeline import Preprocessor
from metaexp.common import metrics

DEADLINE=1792191600

def run_trial(ctx):
    import torch,pyarrow as pa
    from sdm import TableTensor
    from sdm.models import KumoTabular
    from kumo_cuda_cache import keep_fit_cache_on_cuda
    assert time.time()<DEADLINE and torch.cuda.is_available()
    torch.set_num_threads(ctx.spec['resources']['cpus']);torch.cuda.set_per_process_memory_fraction(.90)
    bank=Bank(Path(ctx.inputs['bank']).parent);split=json.loads(ctx.split.read_text())
    assert np.array_equal(bank.dev,split['development_rows']) and np.array_equal(bank.folds,split['folds'])
    rep=json.loads(Path(__file__).with_name('views.json').read_text())[ctx.params['view']]
    rep=json.loads(json.dumps(rep));rep['config']['weight']=1
    pred=np.full(len(bank.y),np.nan)
    for fold in range(3):
        ctx.check_stop();seed=ctx.seed+fold;rng=np.random.default_rng(seed)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        ti=np.flatnonzero(bank.folds!=fold);vi=np.flatnonzero(bank.folds==fold)
        pre=Preprocessor(bank,rep,'lgb',seed);x=pre.fit_transform(bank.dev[ti],bank.y[ti]);q=pre.transform(bank.dev[vi]);y=bank.y[ti]
        pre.bank=None;name=f'fold_{fold}/preprocessing.joblib';joblib.dump(pre,ctx.path(name),compress=1);ctx.model(name)
        def tensor(a):
            cols={f'f{i}':pa.array(a[:,i],type=pa.float64()) for i in range(a.shape[1])}
            previous=torch.get_default_dtype();torch.set_default_dtype(torch.float64)
            try:return TableTensor.from_arrow(pa.table(cols),stypes={k:'numerical' for k in cols},device='cuda')
            finally:torch.set_default_dtype(previous)
        def labels(a):return TableTensor.from_arrow(pa.table({'target':pa.array(a.astype(np.int64),type=pa.int64())}),stypes={'target':'categorical'},device='cuda')
        m=KumoTabular(task='classification',size='large',pretrained=True,device='cuda');cache=keep_fit_cache_on_cuda(m)
        steps=ctx.params['updates'];losses=[]
        if steps:
            m.train();opt=torch.optim.AdamW(m.parameters(),lr=ctx.params['lr'],weight_decay=.01)
            # Episodic adaptation uses disjoint context/query rows from outer TRAINING only.
            for step in range(steps):
                ctx.check_stop();ids=rng.choice(len(ti),min(160,len(ti)),replace=False);ci=ids[:-32];qi=ids[-32:]
                assert not np.intersect1d(ci,qi).size
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                    out=m(tensor(x[ci]),labels(y[ci]),tensor(x[qi]),num_estimators=1,estimator_batch_size=1,generator=torch.Generator(device='cuda').manual_seed(seed+step))
                    p=out[['1']].numerical.squeeze(-1).float().clamp(1e-6,1-1e-6)
                    assert p.requires_grad,'SDK output detached; gradient experiment invalid'
                loss=torch.nn.functional.binary_cross_entropy(p,torch.as_tensor(y[qi],device='cuda',dtype=torch.float32))
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();losses.append(float(loss.detach()))
                if step%16==0:
                    name=f'fold_{fold}/updates_{step+1}.pt';torch.save(m.state_dict(),ctx.path(name));ctx.model(name)
                    ctx.save_json(f'fold_{fold}/training.json',{'loss':losses,'updates':step+1,'train_rows':len(ti)})
            del opt
        m.eval()
        # Same frozen checkpoint is content-shared for zero-update trials; trained state retained separately.
        name=f'fold_{fold}/weights.pt';torch.save(m.state_dict(),ctx.path(name));ctx.model(name)
        from sklearn.model_selection import train_test_split
        count=min(ctx.params['context'],len(ti))
        if count==len(ti):ci=np.arange(len(ti))
        else:_,ci=train_test_split(np.arange(len(ti)),test_size=count,stratify=y,random_state=seed)
        np.save(ctx.path(f'fold_{fold}/context_rows.npy'),bank.dev[ti[ci]]);ctx.model(f'fold_{fold}/context_rows.npy')
        with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.bfloat16):
            m.fit(tensor(x[ci]),labels(y[ci]),num_estimators=ctx.params['estimators'],estimator_batch_size=1,generator=torch.Generator(device='cuda').manual_seed(seed))
        import cloudpickle
        name=f'fold_{fold}/sdk_fitted_recipe.pkl'
        with ctx.path(name).open('wb') as f:cloudpickle.dump(m._cache['recipe_execution'],f)
        ctx.model(name)
        for start in range(0,len(vi),ctx.params['query_batch']):
            ctx.check_stop()
            with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.bfloat16):
                out=m.predict(tensor(q[start:start+ctx.params['query_batch']]))
            pred[vi[start:start+ctx.params['query_batch']]]=out[['1']].numerical.squeeze(-1).float().cpu().numpy()
        np.savez_compressed(ctx.path(f'fold_{fold}/validation.npz'),rows=bank.dev[vi],prediction=pred[vi])
        ctx.save_json(f'fold_{fold}/provenance.json',{'sdk_revision':'329a707bbcf9ca147aa45a63fef8b2029f4ee37b','checkpoint':'nvidia/Kumo-Tabular@v1.0.1/large/classifier.pt','episode_context_rows':128,'episode_query_rows':32,'updates':steps,'losses':losses,'context_rows':count,'seed':seed,'cache':cache,'gpu':torch.cuda.get_device_name(),'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'torch_deterministic':torch.are_deterministic_algorithms_enabled(),'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,'matmul_precision':torch.get_float32_matmul_precision(),'preprocessing':pre.audit})
        del pre,m,x,q;gc.collect();torch.cuda.empty_cache()
    np.savez_compressed(ctx.path('validation.npz'),rows=bank.dev,fold=bank.folds,prediction=pred)
    return {'validation':['validation.npz'],'folds':3}

def evaluate(ctx,result):
    bank=Bank(Path(ctx.inputs['bank']).parent);z=np.load(ctx.path('validation.npz'));p=z['prediction']
    assert np.array_equal(z['rows'],bank.dev) and np.array_equal(z['fold'],bank.folds)
    assert p.shape==bank.y.shape and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    from scipy.stats import ks_2samp
    score=metrics(bank.y,p);ind=float(ks_2samp(p[bank.y==0],p[bank.y==1]).statistic);assert abs(score['ks']-ind)<1e-12
    return {'score':score['ks'],'per_fold':[metrics(bank.y[bank.folds==f],p[bank.folds==f])['ks'] for f in range(3)],'metrics':score,'independent_ks':ind,'caveat':'Reused adaptive development folds; no independent confirmation and no competition-test predictions'}
