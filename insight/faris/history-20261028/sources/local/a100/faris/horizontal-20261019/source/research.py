"""Finite mechanistic study: independent champions, specialists, and honest heads.

No hyperparameter search, competition-test prediction, or model promotion.
The legacy Kumo inputs retain explicitly unresolved snapshot provenance.
"""
import os
for _k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_k]='1'
import argparse,copy,hashlib,json,sys,time,traceback,warnings
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from scipy.stats import ks_2samp,spearmanr,rankdata
from sklearn.metrics import roc_auc_score,log_loss,average_precision_score
from features import ExploreBank
from metaexp.pipeline import fit_predict
from utils import read,write,sha,ident,ks

HERE=Path(__file__).resolve().parent
COVER=['account_month_count','repayment_installment_count','bureau_provider_count','ecom_completed_count']
APR_INPUT=['age','employment_tenure_months','declared_monthly_income','dependents','loan_amount',
           'term_months','bureau_score_mean','bureau_utilization_max','bureau_inquiries_max',
           'account_income_mean','account_outflow_mean','account_month_count']
HEAD_INPUT=['installment_income_ratio','account_income_cv','account_low_balance_days_mean',
            'account_net_inflow_mean','bureau_provider_count','account_month_count']
FUSIONS=['horizontal_probability','horizontal_logit','context_equal','convex_specialists','convex_enriched',
         'coverage_gate','offset_calibration','offset_pricing']

def setup(root):
    root=Path(root); c=read(root/'config.json'); b=ExploreBank(Path(c['inputs']['bank']).parent,c['inputs']['extras'])
    train=pd.read_csv(c['inputs']['train']);assert train.application_id.tolist()==b.metadata.application_id.iloc[:len(train)].tolist()
    return c,b,train

def frame_raw(b,rows,names):
    assert all(n in b.lookup for n in names)
    return b.raw(rows,names).astype(float)

def friend_root(c):return Path(c['inputs']['friend'])

def parts(c,f):
    with np.load(friend_root(c)/f'artifacts/fold{f}/indices.npz') as z:return {k:z[k].astype(int) for k in z.files}

def stored(c,f,kind,rows,train):
    name={'friend':'final_oof','kumo':'kumo_oof','holdout':'kumo_holdout'}[kind]
    p=pd.read_parquet(friend_root(c)/f'artifacts/fold{f}/{name}.parquet')
    assert p.application_id.tolist()==train.application_id.iloc[rows].tolist()
    assert np.array_equal(p.target.to_numpy(int),train.target.iloc[rows].to_numpy(int))
    q=p.prediction.to_numpy(float);assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
    return q

def metrics(y,p):
    return {'ks':ks(y,p),'auc':float(roc_auc_score(y,p)),
            'logloss':float(log_loss(y,np.clip(p,1e-8,1-1e-8))),
            'average_precision':float(average_precision_score(y,p))}

def independent_auc(y,p):
    y=np.asarray(y);positive=y==1;n1=int(positive.sum());n0=len(y)-n1
    return float((rankdata(p)[positive].sum()-n1*(n1+1)/2)/(n1*n0))

def supported(matches,ks_lower,auc_lower):
    return bool(np.mean([r['pooled_delta'] for r in matches])>0 and ks_lower>0 and
        np.mean([r['auc_delta'] for r in matches])>0 and auc_lower>-.0005 and
        all(r['fold_wins']>=4 and r['worst_fold_delta']>=-.003 and r['auc_delta']>=0 and
            r['auc_fold_wins']>=3 and r['logloss_delta']<=.0005 for r in matches))

def specialist(name):
    base=copy.deepcopy(read(HERE/'champion.json')[0]['recipe'])
    domain={'account_spline':'account','account_dynamic':'account','bureau_logit':'bureau','bureau_dynamic':'bureau','shopping_xgb':'shopping'}[name]
    c=base['representation']['config'];c.update(view='v3',sources=['application','profile',domain],
        select='true_gain',feature_count=64,encoding='onehot',transform='signed_log',
        reducer='none',spline=name in ['account_spline','account_dynamic'],interactions=False,append=True,
        representation='raw',indicator=True,window=180,clip=.005,scale='standard',missing='median')
    base.update(model='xgb' if name=='shopping_xgb' else 'linear',seed=2026)
    base['representation']['family']=domain
    if name.endswith('dynamic'):
        c['_extra_family']=domain
        suffixes=(['persistent_deficit','balance_recovery','surplus_last_change','income_like_credits_last_change'] if domain=='account'
                  else ['matched_provider_disagreement','score_change','score_staleness_product'])
        c['_extra_columns']=[f'new_{window}_{suffix}' for window in [180,'all'] for suffix in suffixes]
    return base

def verify_outputs(folder):
    out=read(folder/'complete.json')
    for name,h in out['files'].items():assert sha(folder/name)==h,('changed fit',folder,name)
    return out

def worker(root,specfile,folder):
    root=Path(root);folder=Path(folder);spec=read(specfile);c,b,t=setup(root)
    ix=parts(c,spec['fold']);tr=ix['train'] if spec['partition']=='full' else ix['context']
    qr=ix['validation'] if spec['partition']=='full' else np.r_[ix['correction'],ix['validation']]
    assert not np.intersect1d(tr,qr).size
    assert time.time()<read(root/'state.json')['deadline'] and not (root/'stop.json').exists()
    y=t.target.to_numpy(int);seed=spec['seed'];start=time.time()
    write(folder/'boundary.json',{'train_rows_sha':ident(tr.tolist()),'query_rows_sha':ident(qr.tolist()),
         'labels_sha':ident(y[tr].tolist()),'train_count':len(tr),'query_count':len(qr),
         'outer_validation_not_fitted':True,'contract_sha':c['contract_sha']})
    print(json.dumps({'started':spec,'train_rows':len(tr),'query_rows':len(qr)}),flush=True)
    try:
        if spec['model']=='pricing_aux':
            from sklearn.impute import SimpleImputer
            from sklearn.ensemble import HistGradientBoostingRegressor
            fill=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)
            x=frame_raw(b,tr,APR_INPUT);z=frame_raw(b,qr,APR_INPUT)
            x=np.sign(x)*np.log1p(np.abs(x));z=np.sign(z)*np.log1p(np.abs(z))
            x=fill.fit_transform(x);z=fill.transform(z)
            target=frame_raw(b,tr,['annual_interest_rate']).ravel();assert np.isfinite(target).all()
            model=HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=15,min_samples_leaf=100,
                learning_rate=.05,l2_regularization=20,early_stopping=False,random_state=seed)
            model.fit(x,target);pred=model.predict(z)
            joblib.dump({'model':model,'imputer':fill,'columns':APR_INPUT},folder/'model.joblib',compress=1)
            audit={'nuisance_target':'annual_interest_rate','default_labels_used':False,'columns':APR_INPUT}
        else:
            recipe=(copy.deepcopy(read(HERE/'champion.json')[int(spec['model'][-1])]['recipe'])
                    if spec['model'].startswith('champion') else specialist(spec['model']))
            if recipe['model']=='linear' and not spec['model'].startswith('champion'):
                from metaexp.pipeline import Preprocessor,predict
                from sklearn.linear_model import LogisticRegression
                from sklearn.exceptions import ConvergenceWarning
                pre=Preprocessor(b,recipe['representation'],'linear',seed);x=pre.fit_transform(tr,y[tr]);w=6
                model=LogisticRegression(C=.05,max_iter=1000,tol=.0001,class_weight={0:1,1:w},solver='lbfgs',random_state=seed)
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning);model.fit(x,y[tr])
                pred=predict(model,pre.transform(qr),w);ww=[]
            else:pred,pre,model,w,cdf,ww=fit_predict(b,recipe,tr,y[tr],qr,'cpu',seed)
            audit={'preprocessing':pre.audit,'recipe':recipe,'warnings':ww}
            pre.bank=None;joblib.dump({'model':model,'preprocessing':pre,'weight':w},folder/'model.joblib',compress=1)
        assert pred.shape==(len(qr),) and np.isfinite(pred).all()
        np.savez_compressed(folder/'prediction.npz',rows=qr,prediction=pred)
        write(folder/'audit.json',audit)
        write(folder/'complete.json',{'seconds':time.time()-start,'files':{p.name:sha(p) for p in folder.iterdir()
            if p.is_file() and p.name not in ['complete.json','worker.log','resources.json']}})
        print(json.dumps({'completed':spec['key'],'seconds':time.time()-start}),flush=True)
    except BaseException:
        write(folder/'failure.json',{'traceback':traceback.format_exc()});raise

def loadfit(root,key,rows):
    folder=Path(root)/'fits'/key;verify_outputs(folder);z=np.load(folder/'prediction.npz')
    assert np.array_equal(z['rows'],rows);return z['prediction']

def key(seed,f,part,model):return f's{seed}_f{f}_{part}_{model}'

def fit_schedule():
    out=[]
    # Seed repeat measures training sensitivity, not independent labels or new trials.
    for seed in [2026,2027]:
        for partition,models in [('full',[f'champion{i}' for i in range(5)]),
                    ('context',[f'champion{i}' for i in range(5)]+['account_spline','bureau_logit','shopping_xgb','pricing_aux','account_dynamic','bureau_dynamic'])]:
            for f in range(5):
                for model in models:
                    out.append(dict(key=key(seed,f,partition,model),seed=seed,fold=f,partition=partition,model=model))
    return out

def preflight(root):
    c,b,t=setup(root);n=len(t);checks=[]
    def check(name,ok,details):
        checks.append({'id':name,'status':'pass' if ok else 'fail','details':details});assert ok,(name,details)
    check('identity',n==70000 and t.application_id.is_unique and t.customer_id.is_unique,
          {'rows':n,'unique_applications':int(t.application_id.nunique()),'unique_customers':int(t.customer_id.nunique())})
    friend=pd.read_parquet(friend_root(c)/'data/train_features.parquet',columns=['application_id','customer_id','target'])
    imported=read(friend_root(c)/'reports/IMPORT_MANIFEST.json')['artifacts']
    verified=0
    for item in imported:
        path=friend_root(c)/item['destination']
        if path.is_file():assert sha(path)==item['sha256'],('import checksum',item['destination']);verified+=1
    check('friend_artifact_chain',verified>=26,{'verified_original_import_digests':verified})
    check('friend_same_labels',friend.application_id.tolist()==t.application_id.tolist() and
          friend.customer_id.tolist()==t.customer_id.tolist() and np.array_equal(friend.target.to_numpy(int),t.target.to_numpy(int)),sha(friend_root(c)/'data/train_features.parquet'))
    seen=np.zeros(n,int);pred=np.zeros(n);fold=np.zeros(n,int);support=[]
    from sklearn.model_selection import StratifiedKFold
    expected=list(StratifiedKFold(5,shuffle=True,random_state=31027).split(np.arange(n),t.target))
    for f in range(5):
        p=parts(c,f);tr,vi,ctx,cal=[p[k] for k in ['train','validation','context','correction']]
        assert np.array_equal(tr,expected[f][0]) and np.array_equal(vi,expected[f][1])
        assert len(ctx)==50000 and len(cal)==6000 and len(vi)==14000
        assert not np.intersect1d(ctx,cal).size and not np.intersect1d(tr,vi).size
        assert np.array_equal(np.sort(np.r_[ctx,cal]),np.sort(tr))
        m=read(friend_root(c)/f'artifacts/fold{f}/metadata.json')
        for label,rows in [('training',tr),('validation',vi),('context',ctx),('correction_training',cal)]:
            assert hashlib.sha256('\n'.join(t.application_id.iloc[rows]).encode()).hexdigest()==m[label+'_ids_sha256']
        pred[vi]=stored(c,f,'friend',vi,t);stored(c,f,'holdout',cal,t);stored(c,f,'kumo',vi,t)
        seen[vi]+=1;fold[vi]=f
        row={'fold':f,'context_positives':int(t.target.iloc[ctx].sum()),'correction_positives':int(t.target.iloc[cal].sum()),'validation_positives':int(t.target.iloc[vi].sum())}
        assert min(row[k] for k in row if k!='fold')>=500;support.append(row)
    check('fivefold_boundary_and_support',bool((seen==1).all()),support)
    for y,p in [(np.array([0,1,0,1]),np.ones(4)),(np.array([0,0,1,1]),np.array([0,0,1,1])),(np.array([0,1,0,1,1,0]),np.array([.1,.1,.2,.4,.4,.4]))]:
        assert abs(ks(y,p)-ks_2samp(p[y==1],p[y==0]).statistic)<1e-12
    score=ks(t.target,pred);assert abs(score-ks_2samp(pred[t.target==1],pred[t.target==0]).statistic)<1e-12
    check('known_ks_and_saved_baseline',abs(score-.5292994659005608)<1e-12,{'friend_pooled_ks':score,'constant_ks':0.})
    auc=float(roc_auc_score(t.target,pred))
    check('independent_auc_and_saved_baseline',abs(auc-independent_auc(t.target,pred))<1e-12 and abs(auc-.8472026883917805)<1e-12,{'friend_pooled_auc':auc})
    requested={name:specialist(name)['representation']['config']['_extra_columns'] for name in ['account_dynamic','bureau_dynamic']}
    check('named_feature_contrasts_present',all(set(cols).issubset(b.columns(specialist(name)['representation'])[0]) for name,cols in requested.items()),requested)
    check('no_identifier_inputs',all(not any(k in name for k in ['application_id','customer_id','target','datetime']) for name in APR_INPUT+COVER+HEAD_INPUT),'New head has an explicit feature allowlist')
    months=pd.to_datetime(t.decision_datetime).dt.strftime('%Y-%m')
    write(Path(root)/'eda.json',{'month_counts':months.value_counts().sort_index().to_dict(),
        'prevalence':float(t.target.mean()),'history_coverage_missing':{name:float(np.isnan(frame_raw(b,np.arange(n),[name])).mean()) for name in COVER},
        'test_labels_accessed':False,'entity_disjoint':True,'snapshot_point_in_time_certified':False})
    write(Path(root)/'preflight.json',{'checks':checks,'protocol_sha':c['contract_sha'],
        'pilot_admission':'pass','broad_search_admission':False,'reason':'Finite preregistered mechanistic pilots only; inherited snapshot provenance and adaptive label reuse prevent independent generalization claims.'})
    np.savez_compressed(Path(root)/'baseline.npz',target=t.target.to_numpy(int),friend=pred,fold=fold)
    return checks

def solve_weights(p,y):
    prior=np.array([.5,.5,0.,0.,0.]);penalty=.02
    def obj(w):
        q=np.clip(p@w,1e-7,1-1e-7);g=(q-y)/(q*(1-q))/len(y)
        return float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q))+.5*penalty*np.sum((w-prior)**2)),p.T@g+penalty*(w-prior)
    r=minimize(obj,prior,jac=True,method='SLSQP',bounds=[(.3,.7),(.3,.7)]+[(0,.1)]*3,
        constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(5)},options={'maxiter':300,'ftol':1e-12})
    assert r.success,r.message;assert abs(r.x.sum()-1)<1e-7
    return {'weights':r.x,'objective':float(r.fun),'penalty':penalty}

def gate_fit(z,p,y):
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler();z=scaler.fit_transform(z);x=np.column_stack([np.ones(len(z)),np.clip(z,-4,4)])
    def obj(v):
        s=expit(x@v);w=.2+.6*s;q=np.clip(w*p[:,0]+(1-w)*p[:,1],1e-7,1-1e-7)
        g=(q-y)/(q*(1-q))*(p[:,0]-p[:,1])*.6*s*(1-s)/len(y)
        return float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q))+.015*(v@v)),x.T@g+.03*v
    r=minimize(obj,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-12,'gtol':1e-8})
    assert r.success,r.message
    return {'scaler':scaler,'beta':r.x,'objective':float(r.fun),'penalty':.03}

def gate_predict(m,z,p):
    x=np.column_stack([np.ones(len(z)),np.clip(m['scaler'].transform(z),-4,4)])
    w=.2+.6*expit(x@m['beta']);return w*p[:,0]+(1-w)*p[:,1]

def offset_fit(raw,p,y):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler,SplineTransformer
    imp=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)
    # Only correction-training rows fit all learned head transformations.
    z=np.sign(raw)*np.log1p(abs(raw));z=imp.fit_transform(z)
    spline=SplineTransformer(n_knots=3,degree=2,knots='quantile',extrapolation='constant');z=spline.fit_transform(z)
    scale=StandardScaler();z=scale.fit_transform(z);x=np.column_stack([np.ones(len(z)),np.clip(z,-5,5)])
    offset=logit(np.clip(p,1e-6,1-1e-6))
    def obj(v):
        eta=offset+x@v;g=x.T@(expit(eta)-y)/len(y);g[1:]+=.03*v[1:]
        return float(np.mean(np.logaddexp(0,eta)-y*eta)+.015*(v[1:]@v[1:])),g
    r=minimize(obj,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':400,'ftol':1e-12,'gtol':1e-7})
    assert r.success and np.max(abs(obj(r.x)[1]))<1e-4,r.message
    return {'imputer':imp,'spline':spline,'scale':scale,'beta':r.x,'strength':.25,'max_logit_change':.5,'penalty':.03}

def offset_predict(m,raw,p):
    z=np.sign(raw)*np.log1p(abs(raw));z=m['scale'].transform(m['spline'].transform(m['imputer'].transform(z)))
    x=np.column_stack([np.ones(len(z)),np.clip(z,-5,5)]);change=np.clip(.25*(x@m['beta']),-.5,.5)
    return expit(logit(np.clip(p,1e-6,1-1e-6))+change)

def fuse_fold(root,seed,f):
    root=Path(root);c,b,t=setup(root);ix=parts(c,f);hi,vi=ix['correction'],ix['validation'];qr=np.r_[hi,vi];nh=len(hi)
    folder=root/'heads'/f's{seed}_f{f}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():verify_outputs(folder);return
    cp=np.column_stack([loadfit(root,key(seed,f,'context',f'champion{i}'),qr) for i in range(5)])
    fp=np.column_stack([loadfit(root,key(seed,f,'full',f'champion{i}'),vi) for i in range(5)])
    champion=np.sort(cp,axis=1)[:,1:4].mean(1);full=np.sort(fp,axis=1)[:,1:4].mean(1)
    kumo=np.r_[stored(c,f,'holdout',hi,t),stored(c,f,'kumo',vi,t)]
    friend=stored(c,f,'friend',vi,t)
    columns=[kumo,champion]+[loadfit(root,key(seed,f,'context',name),qr) for name in ['account_spline','bureau_logit','shopping_xgb']]
    p=np.column_stack(columns);y=t.target.to_numpy(int)[hi];equal=.5*(kumo+champion)
    weights=solve_weights(p[:nh],y)
    enriched=p.copy()
    for j,name in [(2,'account_dynamic'),(3,'bureau_dynamic')]:enriched[:,j]=loadfit(root,key(seed,f,'context',name),qr)
    enriched_weights=solve_weights(enriched[:nh],y)
    z=np.log1p(np.nan_to_num(frame_raw(b,qr,COVER),nan=0).clip(0));z=np.column_stack([z,abs(logit(np.clip(kumo,1e-6,1-1e-6))-logit(np.clip(champion,1e-6,1-1e-6)))])
    gate=gate_fit(z[:nh],p[:nh,:2],y)
    apr_expected=loadfit(root,key(seed,f,'context','pricing_aux'),qr)
    apr=frame_raw(b,qr,['annual_interest_rate']).ravel();aprres=apr-apr_expected
    raw=np.column_stack([apr,apr_expected,aprres,frame_raw(b,qr,HEAD_INPUT)])
    calibration=offset_fit(logit(np.clip(equal[:nh],1e-6,1-1e-6))[:,None],equal[:nh],y)
    pricing=offset_fit(np.column_stack([logit(np.clip(equal[:nh],1e-6,1-1e-6)),raw[:nh]]),equal[:nh],y)
    outputs={'friend':friend,'champion':full,'context_champion':champion[nh:],
        'horizontal_probability':.5*(friend+full),
        'horizontal_logit':expit(.5*(logit(np.clip(friend,1e-6,1-1e-6))+logit(np.clip(full,1e-6,1-1e-6)))),
        'context_equal':equal[nh:],'convex_specialists':p[nh:]@weights['weights'],
        'convex_enriched':enriched[nh:]@enriched_weights['weights'],
        'account_spline':p[nh:,2],'account_dynamic':enriched[nh:,2],
        'bureau_logit':p[nh:,3],'bureau_dynamic':enriched[nh:,3],
        'coverage_gate':gate_predict(gate,z[nh:],p[nh:,:2]),
        'offset_calibration':offset_predict(calibration,logit(np.clip(equal[nh:],1e-6,1-1e-6))[:,None],equal[nh:]),
        'offset_pricing':offset_predict(pricing,np.column_stack([logit(np.clip(equal[nh:],1e-6,1-1e-6)),raw[nh:]]),equal[nh:])}
    for name,value in outputs.items():assert len(value)==len(vi) and np.isfinite(value).all()
    for name,m in [('convex',weights),('convex_enriched',enriched_weights),('gate',gate),('calibration',calibration),('pricing',pricing)]:joblib.dump(m,folder/(name+'.joblib'),compress=1)
    np.savez_compressed(folder/'prediction.npz',rows=vi,**outputs)
    # No outer scores are consulted until all folds and both seeds finish.
    write(folder/'audit.json',{'head_fit_rows':hi.tolist(),'query_rows':vi.tolist(),'upstream_fit_rows_sha':ident(ix['context'].tolist()),
        'same_outer_fold':True,'head_never_sees_in_sample_base_predictions':True,
        'friend_corrected_predictions_not_used_to_fit_heads':True,'feature_columns':HEAD_INPUT,'pricing_aux_inputs':APR_INPUT,
        'fusion_order':['raw_kumo','context_champion','account_spline','bureau_logit','shopping_xgb'],
        'convex_weights':weights['weights'].tolist(),'enriched_weights':enriched_weights['weights'].tolist(),'training_seed':seed,'historical_selection_reuse':True})
    write(folder/'complete.json',{'files':{p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='complete.json'}})

def finish(root):
    root=Path(root);c,b,t=setup(root);y=t.target.to_numpy(int);fold=np.load(root/'baseline.npz')['fold'];preds={}
    for seed in [2026,2027]:
        current={}
        for f in range(5):
            folder=root/'heads'/f's{seed}_f{f}';verify_outputs(folder);z=np.load(folder/'prediction.npz');rows=z['rows']
            for name in z.files:
                if name=='rows':continue
                if name not in current:current[name]=np.full(len(y),np.nan)
                current[name][rows]=z[name]
        assert all(np.isfinite(p).all() for p in current.values());preds[seed]=current
        np.savez_compressed(root/f'oof_s{seed}.npz',target=y,fold=fold,**current)
    # Fitted ensemble scores, not the all-data refit. No prediction of competition rows.
    rows=[]
    for seed,ds in preds.items():
        for name,p in ds.items():
            row={'seed':seed,'recipe':name,**metrics(y,p),'comparisons':{},
                 'fold_metrics':[metrics(y[fold==f],p[fold==f]) for f in range(5)]}
            for anchor in ['friend','champion']:
                q=ds[anchor];fd=[ks(y[fold==f],p[fold==f])-ks(y[fold==f],q[fold==f]) for f in range(5)]
                row['comparisons'][anchor]={'pooled_delta':ks(y,p)-ks(y,q),'fold_deltas':fd,'mean_fold_delta':float(np.mean(fd)),
                    'fold_wins':int(sum(d>0 for d in fd)),'worst_fold_delta':min(fd),'spearman':float(spearmanr(p,q).statistic),
                    'auc_delta':float(roc_auc_score(y,p)-roc_auc_score(y,q)),
                    'auc_fold_deltas':[float(roc_auc_score(y[fold==f],p[fold==f])-roc_auc_score(y[fold==f],q[fold==f])) for f in range(5)],
                    'logloss_delta':float(log_loss(y,p)-log_loss(y,q))}
            rows.append(row)
    write(root/'scoreboard.json',rows)
    # All fixed comparisons are reported. No winning weight search or repeated tuning.
    rng=np.random.default_rng(20261019);groups=[np.flatnonzero((fold==f)&(y==cl)) for f in range(5) for cl in [0,1]]
    pairs=[(name,anchor) for name in FUSIONS for anchor in ['friend','champion']]
    pairs += [('convex_specialists','context_equal'),('coverage_gate','context_equal'),
              ('offset_pricing','offset_calibration'),('horizontal_logit','horizontal_probability'),
              ('convex_enriched','convex_specialists'),('account_dynamic','account_spline'),('bureau_dynamic','bureau_logit')]
    boot={str(pair):[] for pair in pairs};boot_auc={str(pair):[] for pair in pairs};reps=2000
    for repeat in range(reps):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);ys=y[ix]
        cache={(seed,name):ks(ys,p[ix]) for seed,ds in preds.items() for name,p in ds.items()}
        auc_cache={(seed,name):roc_auc_score(ys,p[ix]) for seed,ds in preds.items() for name,p in ds.items()}
        for name,anchor in pairs:
            boot[str((name,anchor))].append(np.mean([cache[(seed,name)]-cache[(seed,anchor)] for seed in preds]))
            boot_auc[str((name,anchor))].append(np.mean([auc_cache[(seed,name)]-auc_cache[(seed,anchor)] for seed in preds]))
        if (repeat+1)%100==0:print(json.dumps({'bootstrap_completed':repeat+1,'bootstrap_total':reps}),flush=True)
    uncertainty=[];tail=.025/(2*len(pairs))
    for name,anchor in pairs:
        vals=np.asarray(boot[str((name,anchor))]);matches=[]
        for seed,ds in preds.items():
            fd=[ks(y[fold==f],ds[name][fold==f])-ks(y[fold==f],ds[anchor][fold==f]) for f in range(5)]
            matches.append({'pooled_delta':ks(y,ds[name])-ks(y,ds[anchor]),'fold_wins':sum(d>0 for d in fd),'worst_fold_delta':min(fd),
                'auc_delta':float(roc_auc_score(y,ds[name])-roc_auc_score(y,ds[anchor])),
                'auc_fold_wins':sum(roc_auc_score(y[fold==f],ds[name][fold==f])>roc_auc_score(y[fold==f],ds[anchor][fold==f]) for f in range(5)),
                'logloss_delta':float(log_loss(y,ds[name])-log_loss(y,ds[anchor]))})
        lo,hi=np.quantile(vals,[tail,1-tail]);point=float(np.mean([r['pooled_delta'] for r in matches]))
        alo,ahi=np.quantile(boot_auc[str((name,anchor))],[tail,1-tail]);apoint=float(np.mean([r['auc_delta'] for r in matches]))
        uncertainty.append({'recipe':name,'reference':anchor,'mean_seed_pooled_delta':point,'familywise_95_CI':[float(lo),float(hi)],
             'mean_seed_auc_delta':apoint,'auc_familywise_95_CI':[float(alo),float(ahi)],
             'ks_only_gain':bool(point>0 and apoint<0),
             'stable':supported(matches,lo,alo),
             'historical_search_not_corrected':True})
    write(root/'uncertainty.json',{'comparisons':uncertainty,'method':'paired fold/class bootstrap, 2000 resamples, Bonferroni 23 comparisons times two metrics; conditional on fitted predictions; paired seeds averaged, not treated as independent observations',
        'auc_noninferiority_margin':.0005,'requires_positive_point_auc_each_seed':True,'logloss_max_increase':.0005})
    diversity=[]
    p=preds[2026]
    for a in p:
        for d in p:
            if a>=d:continue
            diversity.append({'a':a,'b':d,'spearman':float(spearmanr(p[a],p[d]).statistic),
                 'error_correlation':float(np.corrcoef(y-p[a],y-p[d])[0,1]),'mean_abs_probability_difference':float(np.mean(abs(p[a]-p[d])))})
    write(root/'diversity.json',diversity)
    cohort=[];metadata=b.metadata.iloc[:len(y)];month=pd.to_datetime(metadata.decision_datetime).dt.strftime('%Y-%m').to_numpy()
    conditions={str(m):month==m for m in np.unique(month)}
    for name in COVER:
        v=frame_raw(b,np.arange(len(y)),[name]).ravel();conditions[name+'_absent']=(~np.isfinite(v))|(v==0)
        conditions[name+'_present']=np.isfinite(v)&(v>0)
    for label,ix in conditions.items():
        if ix.sum()<300 or min(y[ix].sum(),(1-y[ix]).sum())<30:continue
        for name in FUSIONS:cohort.append({'cohort':label,'recipe':name,'rows':int(ix.sum()),
            'delta_friend':ks(y[ix],p[name][ix])-ks(y[ix],p['friend'][ix]),
            'auc_delta_friend':float(roc_auc_score(y[ix],p[name][ix])-roc_auc_score(y[ix],p['friend'][ix])),
            'logloss_delta_friend':float(log_loss(y[ix],p[name][ix])-log_loss(y[ix],p['friend'][ix])),
            'diagnostic_only':True})
    write(root/'cohort_diagnostics.json',cohort)
    useful=[name for name in FUSIONS if all(r['stable'] for r in uncertainty if r['recipe']==name and r['reference'] in ['friend','champion'])]
    lines=['# Horizontal research results','','Completed two training seeds on the same five outer folds. Historical labels and feature choices were already reused; no untouched validation remains.',
      '', 'Inherited Kumo and v3 DPD/paid snapshot availability remains uncertified. This is a conditional benchmark, not independent validation or automatic promotion.',
      '', 'Primary comparison: predeclared equal-probability combination of the two independently successful pipelines. Learned heads use only the disjoint 6,000-row correction set, upstream models only the 50,000-row context.',
      '', '| Recipe | Seed | KS | AUC | KS delta vs friend | AUC delta vs friend | KS delta vs champion | AUC delta vs champion |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['recipe']}|{r['seed']}|{r['ks']:.6f}|{r['auc']:.6f}|{r['comparisons']['friend']['pooled_delta']:+.6f}|{r['comparisons']['friend']['auc_delta']:+.6f}|{r['comparisons']['champion']['pooled_delta']:+.6f}|{r['comparisons']['champion']['auc_delta']:+.6f}|")
    lines+=['',f'Both-reference conditional stability criterion passed: {useful or "none"}.',
       'No positive conclusion follows from a higher point estimate alone. Confidence intervals do not undo historical adaptive feature selection.',
       '', 'Mechanism contrasts: horizontal_probability vs both anchors; context_equal vs the full champions measures fitting-partition cost; convex_specialists and coverage_gate vs context_equal test fusion; offset_pricing vs offset_calibration separates pricing features from calibration.',
       '', 'No competition test inference, submission refit, upload, or changes to the existing submission. Every fitted model and transformer remains in fits/ and heads/.',
       'A KS-only gain with declining AUC is rejected. Conditional support requires nonnegative pooled AUC improvement in both seeds and positive mean improvement, at least 3/5 AUC fold wins per seed, adjusted AUC lower bound >-0.0005, and log-loss increase <=0.0005, in addition to the strict KS criterion.',
       'The 2,000-resample intervals at the adjusted extreme quantiles have limited Monte Carlo resolution. They describe these fitted predictions only; they do not establish leaderboard generalization.',
       'See scoreboard.json, uncertainty.json, diversity.json, cohort_diagnostics.json and source manifests for details.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(root/'result.json',{'status':'complete','base_fits':len(fit_schedule()),'head_fits':50,'training_seeds':2,'outer_folds':5,
        'fusion_recipes':len(FUSIONS),'conditional_stable_candidates':useful,'independent_confirmation':False,'point_in_time_certified':False,'submission_created':False})

def tests():
    rng=np.random.default_rng(37);n=800;y=rng.integers(0,2,n);p=expit(rng.normal(size=(n,5)))
    w=solve_weights(p,y);assert abs(w['weights'].sum()-1)<1e-8 and (w['weights'][:2]>=.3-1e-8).all()
    z=rng.normal(size=(n,5));g=gate_fit(z,p[:,:2],y);out=gate_predict(g,z,p[:,:2]);assert (out>=p[:,:2].min(1)-1e-8).all() and (out<=p[:,:2].max(1)+1e-8).all()
    xx=rng.normal(size=(n,4));m=offset_fit(xx,p[:,0],y);a=offset_predict(m,xx,p[:,0]);assert np.max(abs(logit(a)-logit(p[:,0])))<=.5+1e-10
    for yy,pp in [(y,np.round(p[:,0],1)),(np.array([0,1]),np.ones(2))]:assert abs(ks(yy,pp)-ks_2samp(pp[yy==1],pp[yy==0]).statistic)<1e-12
    for pp in [np.round(p[:,0],1),p[:,0],np.ones(n)]:assert abs(independent_auc(y,pp)-roc_auc_score(y,pp))<1e-12
    good={'pooled_delta':.002,'auc_delta':.001,'fold_wins':5,'worst_fold_delta':.0001,'auc_fold_wins':4,'logloss_delta':-.0002}
    assert supported([good,good],.0001,-.0001)
    assert not supported([good,{**good,'auc_delta':-.0001}],.0001,-.0001)
    assert not supported([good,good],.0001,-.0006)
    assert not supported([good,good],-.0001,-.0001)
    assert len(fit_schedule())==160 and len({s['key'] for s in fit_schedule()})==160
    for base,new in [('account_spline','account_dynamic'),('bureau_logit','bureau_dynamic')]:
        a=specialist(base);b=specialist(new);b['representation']['config'].pop('_extra_family');b['representation']['config'].pop('_extra_columns');assert a==b
    print('Synthetic optimizer, independent KS/AUC, AUC rejection, schedule, convex-bound and offset-limit tests passed')

def run_trial(ctx,params):raise RuntimeError('No generic search allowed: use the sealed finite-research controller')
def evaluate(ctx,artifact):return read(Path(artifact)/'result.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['worker','preflight','fuse','finish','test']);p.add_argument('root',nargs='?');p.add_argument('--spec');p.add_argument('--folder');p.add_argument('--seed',type=int);p.add_argument('--fold',type=int);a=p.parse_args()
    if a.action=='worker':worker(a.root,a.spec,a.folder)
    elif a.action=='preflight':preflight(a.root)
    elif a.action=='fuse':fuse_fold(a.root,a.seed,a.fold)
    elif a.action=='finish':finish(a.root)
    else:tests()
