"""One recipe-only model/config/fold experiment under the frozen Logika roles."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
os.environ.setdefault('SDM_CHUNK_MEMORY_FRACTION','0.02')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('DO_NOT_TRACK','1')
from pathlib import Path
import argparse,copy,datetime,hashlib,json,sys,time,traceback,shutil,tempfile
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'storage/extensions'))
sys.path.insert(0,str(R/'storage/compat'))
OLD=Path('/mnt/home/faris/logika-faris-exploration/transfer-20261025')
os.environ.setdefault('HF_HOME',str(R/'storage/huggingface'))
sys.path.insert(0,str(OLD/'source'))
import inputs as rb
import baseline as b
import feature_head

def dump(x,p):p.write_text(json.dumps(x,indent=2,default=str,allow_nan=False)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def run(args):
    from tabarena.models import get_model_registry
    registry=get_model_registry();info=registry[args.key]
    slug=info.method_metadata.display_name.replace('/','_').replace(' ','_')
    dest=R/'storage'/('smoke' if args.smoke else 'runs')/slug/f'config{args.config}'/f'fold{args.fold}'
    dest.mkdir(parents=True,exist_ok=False)
    start=time.time()
    assert R.is_relative_to(Path('/mnt/home/faris'))
    scratch_root=R/'runtime';scratch_root.mkdir(exist_ok=True)
    scratch=Path(tempfile.mkdtemp(prefix='fit-',dir=scratch_root))
    os.environ['TMPDIR']=str(scratch);tempfile.tempdir=str(scratch)
    try:
        assert not (R/'STOP').exists()
        assert shutil.disk_usage(R/'storage').free>5_000_000_000
        original,ix,q,context,yc,cols,query,head_cols,spline,oldraw,expected=rb.inputs(args.fold)
        X=context[cols].copy();Q=q[cols].copy();Y=context.target.astype(int).copy()
        cats=X.select_dtypes(include=['object','string','category']).columns
        for c in cols:
            if c in cats:
                # Vocabulary fitted on the 50k context only. Query-only values become missing.
                categories=sorted(X[c].dropna().astype(str).unique())
                X[c]=pd.Categorical(X[c].astype('string'),categories=categories)
                Q[c]=pd.Categorical(Q[c].astype('string'),categories=categories)
            else:
                X[c]=X[c].replace([np.inf,-np.inf],np.nan)
                Q[c]=Q[c].replace([np.inf,-np.inf],np.nan)
        if args.smoke:
            # Balanced smoke sample is only an execution check, never a model ranking.
            rng=np.random.default_rng(251025)
            rows=np.r_[rng.choice(np.flatnonzero(Y.to_numpy()==0),1024,replace=False),rng.choice(np.flatnonzero(Y.to_numpy()==1),1024,replace=False)]
            X=X.iloc[rows].reset_index(drop=True);Y=Y.iloc[rows].reset_index(drop=True);Q=Q.iloc[:128].copy();inner_y=yc[:128]
        else:inner_y=yc
        configs=json.loads((R/'CONFIGS.json').read_text())[args.key]
        params=copy.deepcopy(configs[args.config])
        for k in ['ag_args','ag_args_ensemble','ag.refit_hyperparameters']:params.pop(k,None)
        if args.key.startswith('Kumo-Tabular'):
            params.update(cache_context=True,estimator_batch_size=1)
            params['ag.max_batch_size']=512
        # Seed is passed through the model's documented seed parameter when defined.
        seed_name=getattr(info.model_cls,'seed_name',None)
        if seed_name:params[seed_name]=2026+args.fold
        problem='binary' if 'binary' in info.model_cls._supported_problem_types else 'regression'
        if problem=='regression':assert args.key in ['Nori','Nori-30M']
        model=info.model_cls(path=str(scratch/'model'),name='estimator',problem_type=problem,
            eval_metric='roc_auc' if problem=='binary' else 'root_mean_squared_error',hyperparameters=params)
        gpu=int(info.method_metadata.compute=='gpu')
        dump({'status':'running','started_UTC':now(),'key':args.key,'display':info.method_metadata.display_name,
            'config':args.config,'fold':args.fold,'smoke':args.smoke,'problem':problem,'params':params,
            'columns':cols,'context_rows':len(X),'query_rows':len(Q),'validation_labels_passed_to_fit':False,
            'source_sha256':b.sha(__file__),'configs_sha256':b.sha(R/'CONFIGS.json'),
            'packet_manifest_sha256':b.sha(rb.PACKETS/f'fold{args.fold}/manifest.json'),'scratch_path':str(scratch)},dest/'START.json')
        model.fit(X=X,y=Y,X_val=Q.iloc[:len(inner_y)],y_val=pd.Series(inner_y),num_cpus=2,num_gpus=gpu,time_limit=120 if args.smoke else 1800,verbosity=2)
        p=np.asarray(model.predict_proba(Q) if problem=='binary' else model.predict(Q)).reshape(-1)
        assert len(p)==len(Q) and np.isfinite(p).all()
        np.savez_compressed(dest/'raw_prediction_before_audit.npz',prediction=p)
        repeat=np.asarray(model.predict_proba(Q.iloc[:64]) if problem=='binary' else model.predict(Q.iloc[:64])).reshape(-1)
        repeated_same_shape=np.asarray(model.predict_proba(Q.iloc[:64]) if problem=='binary' else model.predict(Q.iloc[:64])).reshape(-1)
        replay=float(abs(repeated_same_shape-repeat).max());assert replay<1e-7,replay
        batch_shape_sensitivity=float(abs(repeat-p[:64]).max())
        # Half-precision kernels can differ when the query batch shape changes.
        # Require same-shape determinism separately; record the cross-shape change.
        assert batch_shape_sensitivity<.002,batch_shape_sensitivity
        retention='configuration_seed_source_hashes_and_predictions_only'
        metrics={}
        if not args.smoke:
            training=query.iloc[:6000].copy();training['target']=yc
            head=feature_head.fit(training,np.clip(p[:6000],1e-6,1-1e-6),head_cols,.003,.5)
            dump({'columns':head_cols,'penalty':.003,'strength':.5,'fit_rows':6000,'source_sha256':b.sha(feature_head.__file__),'fitted_weights_saved':False},dest/'correction_recipe.json')
            corrected=.75*b.corrected_probability(head,query.iloc[6000:],np.clip(p[6000:],1e-6,1-1e-6))[0]+.25*spline
            np.savez_compressed(dest/'predictions.npz',rows=np.r_[ix['correction'],ix['validation']],raw=p,corrected=corrected)
            metrics={'inner_KS':b.scores(yc,p[:6000])['KS'],'outer_raw':b.scores(original.target.iloc[ix['validation']].to_numpy(),p[6000:]),
                'outer_corrected':b.scores(original.target.iloc[ix['validation']].to_numpy(),corrected)}
        dump({'status':'smoke_passed' if args.smoke else 'fold_complete','key':args.key,'config':args.config,'fold':args.fold,
            'seconds':time.time()-start,'replay_max_abs':replay,'batch_shape_max_abs':batch_shape_sensitivity,'retention':retention,'fitted_weights_saved':False,
            'metrics':metrics,'completed_UTC':now(),'resolved_model_params':model.params,
            'shared_foundation_runtime_cache':os.environ['HF_HOME'],'default_execution_adjustment':'Kumo uses cache and estimator/query batching; same default ensemble count',
            'not_a_Tabarena_benchmark_reproduction':True},dest/'RESULT.json')
        print(json.dumps({'status':'complete','key':args.key,'fold':args.fold,'smoke':args.smoke,'seconds':time.time()-start,'metrics':metrics}),flush=True)
    except BaseException:
        dump({'status':'failed','error':traceback.format_exc(),'seconds':time.time()-start,'key':args.key,'fold':args.fold,'smoke':args.smoke},dest/'FAILURE.json');raise
    finally:
        shutil.rmtree(scratch,ignore_errors=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--fold',type=int,default=0);p.add_argument('--config',type=int,default=0);p.add_argument('--smoke',action='store_true');run(p.parse_args())
