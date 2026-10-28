"""Frozen train/test separability diagnostic; never reads default targets."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='2'
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from catboost import CatBoostClassifier, Pool
import baseline as b

R=Path(__file__).resolve().parents[1]
A=R.parent
D=R/'population_check'
FULL=A/'submission-full-periodicity-20261021'
INPUT=A/'ranking-pipeline-20261020/input'
PARAMS=dict(iterations=300,depth=4,learning_rate=.05,l2_leaf_reg=10,
            loss_function='Logloss',bootstrap_type='No',random_strength=0,
            thread_count=2,verbose=False,allow_writing_files=False)


def main():
    plan=json.loads((R/'POPULATION_CHECK_PLAN.json').read_text())
    assert b.sha(__file__)==plan['source_sha256']
    assert time.time()<json.loads((R/'CONTINUATION_20261025.json').read_text())['training_deadline_epoch']
    D.mkdir(exist_ok=False)
    started=time.time()
    spec=json.loads((A/'kumo-coverage-placement-20261021/conditions/fold0/control90/boundary.json').read_text())
    columns=spec['feature_columns']
    path=FULL/'feature_build/fold0/features.parquet'
    complete=json.loads((path.parent/'complete.json').read_text())
    assert b.sha(path)==complete['feature_table_sha256']
    frame=pd.read_parquet(path,columns=['application_id']+columns)
    apps=pd.read_parquet(FULL/'applications.parquet',columns=['application_id','decision_datetime'])
    ids=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id'])
    assert len(frame)==100000 and len(ids)==70000
    assert frame.application_id.tolist()==apps.application_id.tolist()
    assert frame.application_id.iloc[:70000].tolist()==ids.application_id.tolist()
    assert frame.application_id.is_unique
    coverage=pd.read_parquet(FULL/'coverage.parquet').set_index('application_id').loc[frame.application_id]
    x=frame[columns].copy()
    for c in coverage.columns: x[c]=coverage[c].to_numpy()
    dates=pd.to_datetime(apps.decision_datetime)
    x['calendar_year']=dates.dt.year.to_numpy()
    x['calendar_month']=dates.dt.month.to_numpy()
    x['calendar_dayofyear']=dates.dt.dayofyear.to_numpy()
    assert not set(x.columns)&{'target','application_id','customer_id','row','split'}
    cats=x.select_dtypes(include=['object','string','category']).columns.tolist()
    for c in x:
        if c in cats: x[c]=x[c].fillna('MISSING').astype(str)
        else: x[c]=pd.to_numeric(x[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    label=np.r_[np.zeros(70000,dtype=int),np.ones(30000,dtype=int)]
    # These are domain-classification folds, not the frozen default-prediction folds.
    splitter=StratifiedKFold(5,shuffle=True,random_state=2026102514)
    pred=np.full(len(x),np.nan);fold_ids=np.full(len(x),-1);importance=[];fold_scores=[]
    for fold,(tr,va) in enumerate(splitter.split(x,label)):
        model=CatBoostClassifier(**PARAMS,random_seed=2026102514+fold)
        model.fit(Pool(x.iloc[tr],label=label[tr],cat_features=cats))
        pool=Pool(x.iloc[va],cat_features=cats)
        pred[va]=model.predict_proba(pool)[:,1];fold_ids[va]=fold
        path=D/f'domain_fold{fold}.cbm';model.save_model(path)
        reload=CatBoostClassifier();reload.load_model(path)
        assert np.array_equal(pred[va],reload.predict_proba(pool)[:,1])
        importance.append(model.feature_importances_)
        fold_scores.append(float(roc_auc_score(label[va],pred[va])))
        np.savez_compressed(D/f'roles{fold}.npz',training=tr,validation=va)
        print(json.dumps(dict(domain_fold_complete=fold)),flush=True)
    assert np.isfinite(pred).all() and (fold_ids>=0).all()
    auc=float(roc_auc_score(label,pred))
    # Fixed-prediction stratified paired bootstrap. Each observation appears OOF once.
    order=np.argsort(pred,kind='stable');yl=label[order];ps=pred[order]
    starts=np.r_[0,np.flatnonzero(ps[1:]!=ps[:-1])+1]
    rng=np.random.default_rng(20261025140);draws=[]
    groups=[np.flatnonzero(label==cl) for cl in (0,1)]
    for _ in range(1000):
        counts=np.zeros(len(label),dtype=int)
        for g in groups: counts[g]=rng.multinomial(len(g),np.full(len(g),1/len(g)))
        pcount=np.add.reduceat(counts[order]*yl,starts)
        ncount=np.add.reduceat(counts[order]*(1-yl),starts)
        nprior=np.cumsum(ncount)-ncount
        draws.append(float(np.sum(pcount*(nprior+.5*ncount))/(pcount.sum()*ncount.sum())))
    numeric=[];categorical=[]
    for c in x:
        left=x[c].iloc[:70000];right=x[c].iloc[70000:]
        if c in cats:
            lev=sorted(set(left)|set(right))
            l=left.value_counts(normalize=True).reindex(lev,fill_value=0)
            rr=right.value_counts(normalize=True).reindex(lev,fill_value=0)
            categorical.append(dict(column=c,total_variation=float(.5*np.abs(l-rr).sum())))
        else:
            ln=left.dropna().to_numpy();rn=right.dropna().to_numpy()
            numeric.append(dict(column=c,nonmissing_KS=float(ks_2samp(ln,rn).statistic) if len(ln)and len(rn)else None,
                                missing_train=float(left.isna().mean()),missing_test=float(right.isna().mean())))
    mean_importance=np.mean(importance,axis=0)
    ci=np.quantile(draws,[.025,.975]).tolist()
    materially_separable=auc>=.60 and ci[0]>.55
    np.savez_compressed(D/'domain_OOF.npz',application_id=frame.application_id.to_numpy(str),domain=label,domain_fold=fold_ids,prediction=pred)
    np.savez_compressed(D/'bootstrap.npz',AUC=draws)
    out=dict(status='complete',default_target_values_read=False,domain_rows=100000,domain_train_rows=70000,domain_test_rows=30000,
             AUC=auc,conditional_95pct_AUC_interval=ci,fold_AUC=fold_scores,
             logloss=float(log_loss(label,pred)),constant_logloss=float(log_loss(label,np.full(len(label),.3))),
             material_shift_rule_passed=bool(materially_separable),rule='OOF domain AUC>=.60 and conditional95pct lowerbound>.55',
             interpretation='Evidence for detectable shift under this fixed classifier only; low AUC does not prove identical distributions or label relationships.',
             columns=x.columns.tolist(),categorical_columns=cats,params=PARAMS,
             top_model_importance=[dict(column=x.columns[i],importance=float(mean_importance[i])) for i in np.argsort(-mean_importance)[:20]],
             numeric_shift=sorted(numeric,key=lambda z:z['nonmissing_KS']or 0,reverse=True),categorical_shift=categorical,
             source_feature_path=str(FULL/'feature_build/fold0/features.parquet'),source_feature_sha256=complete['feature_table_sha256'],
             prior_policy='The same existing64kcontext-fitted label-free feature transformations are applied to train and test. Domain learner sees no default labels.',
             source_sha256=b.sha(__file__),plan_sha256=b.sha(R/'POPULATION_CHECK_PLAN.json'),seconds=time.time()-started,
             files={p.name:b.sha(p)for p in D.iterdir()if p.is_file()})
    b.dump(out,D/'RESULT.json');print(json.dumps({k:out[k]for k in ['status','AUC','conditional_95pct_AUC_interval','fold_AUC','material_shift_rule_passed','seconds']}),flush=True)


if __name__=='__main__':main()
