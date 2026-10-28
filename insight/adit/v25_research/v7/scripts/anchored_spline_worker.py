"""Small fixed contribution from a full-outer-training latent spline expert."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='2'
from pathlib import Path
import argparse,io,json,sys,traceback,warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer,MissingIndicator
from sklearn.preprocessing import SplineTransformer,StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'
sys.path[:0]=[str(ROOT/'scripts'),str(V/'scripts')]
from baseline import prepare,corrected_probability,dump,sha,scores,ids_hash

FRAME=None
INTERACTIONS=['re_price_log_posterior','re_bureau_score_posterior','re_late_30_posterior','installment_observed_income_ratio']


def expanded(frame,spec):
    d=frame[spec['input_features']].copy()
    if spec['population_interactions']:
        group=(frame.aux_account_months.gt(0).astype(int)*2+frame.aux_allocation_count.gt(0).astype(int)).to_numpy()
        for g in [1,2,3]:
            d['population_'+str(g)]=(group==g).astype(float)
            for c in INTERACTIONS:
                value=pd.to_numeric(frame[c],errors='coerce').to_numpy(float)
                # Unavailable values remain NaN; only observed group-members
                # receive the corresponding within-group linear product.
                d[f'population_{g}_x_{c}']=np.where(group==g,value,0.)
    return d


def fit_spline(training,spec):
    d=expanded(training,spec);columns=list(d);cats=d.select_dtypes(include=['object','string','category']).columns.tolist()
    numeric=[c for c in columns if c not in cats]
    continuous=[c for c in numeric if c in spec['input_features'] and d[c].nunique(dropna=True)>10]
    linear=[c for c in numeric if c not in continuous]
    encoder=ColumnTransformer([
        ('spline',Pipeline([('fill',SimpleImputer(strategy='median',keep_empty_features=True)),('basis',SplineTransformer(n_knots=3,degree=2,knots='quantile',extrapolation='constant',include_bias=False)),('scale',StandardScaler())]),continuous),
        ('linear',Pipeline([('fill',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),linear),
        ('missing',MissingIndicator(features='all',sparse=False,error_on_new=False),numeric),
        ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])
    x=encoder.fit_transform(prepare(d,columns,cats)).astype(np.float64)
    model=LogisticRegression(C=1/(len(training)*spec['penalty']),solver='lbfgs',max_iter=400,tol=1e-7)
    with threadpool_limits(limits=2),warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always');model.fit(x,training.target.to_numpy(int))
    assert not any(issubclass(w.category,ConvergenceWarning) for w in messages),'Spline did not converge'
    training_score=scores(training.target,model.predict_proba(x)[:,1])
    bundle=dict(encoder=encoder,model=model,columns=columns,categorical=cats,spec=spec)
    audit=dict(training_rows=len(training),standalone_training_metrics=training_score,
        coefficients=model.coef_.size+1,iterations=model.n_iter_.tolist(),
        new_model_sum_loss_C=model.C,mean_loss_L2=spec['penalty'],outer_labels_used=False,
        warnings=[str(w.message) for w in messages])
    return bundle,audit


def predict(bundle,frame):
    d=expanded(frame,bundle['spec']);x=bundle['encoder'].transform(prepare(d,bundle['columns'],bundle['categorical']))
    return bundle['model'].predict_proba(x)[:,1]


def fit_adapter(f,spec,columns):
    indices=np.load(ROOT/'artifacts'/f"fold{f['fold']}"/'indices.npz');training_ids=np.r_[indices['context'],indices['correction']]
    assert len(np.unique(training_ids))==56000 and not np.intersect1d(training_ids,indices['validation']).size
    original=pd.read_parquet(ROOT/'data/train_features.parquet',columns=['application_id','target'])
    training=FRAME.loc[original.application_id.iloc[training_ids]].copy();training['target']=original.target.to_numpy()[training_ids]
    bundle,audit=fit_spline(training,spec)
    audit['full_outer_training_ids_sha256']=ids_hash(original,training_ids)
    val=FRAME.loc[f['val'].application_id]
    audit['standalone_outer_validation_metrics']=scores(f['val'].target,predict(bundle,val))
    return dict(spline=bundle,head=f['head']),audit,audit['coefficients']


def probability(model,frame,kumo,head):
    return corrected_probability(model['head'],frame,kumo)


def final_prediction(model,f,p,tp):
    new_val=predict(model['spline'],FRAME.loc[f['val'].application_id]);new_test=predict(model['spline'],FRAME.loc[f['test'].application_id])
    return .95*p+.05*new_val,.95*tp+.05*new_test


def postprocess(out,result,tr,y,folds,reference,prediction,frames):
    from stability_worker import postprocess as common
    result['overfitting_assessment']['diagnostic']=result['spec']['rationale']+'; the new standalone spline and preprocessing use all56k outer-training rows. The original Kumo correction remains fixed after its original6k fitting. Fixed95%V25 plus5%new spline; no outer weight or route selection. Baseline provenance is inherited.'
    for row in result['folds']:
        row['new_spline_training_rows']=56000
        row['new_spline_training_ids_sha256']=row['optimization']['full_outer_training_ids_sha256']
    common(out,result,tr,y,folds,reference,prediction,frames)
    result.update(original_Kumo_correction_unchanged=True,original_spline_cache_unchanged=True,
        final_weights=dict(original_V25=.95,new_spline=.05),new_spline_training_rows=56000,
        source_cache_sha256=sha(V/'anchored_spline/data/inputs.parquet'))


def preflight():
    rng=np.random.default_rng(282026);specs=json.loads((V/'anchored_spline_plan.json').read_text());n=1600
    cols=specs[1]['input_features'];d=pd.DataFrame({c:rng.lognormal(size=n) for c in cols if c!='employment_type'})
    d['employment_type']=np.resize(['salaried','self_employed','variable_income'],n);d['aux_account_months']=rng.integers(0,7,n);d['aux_allocation_count']=rng.integers(0,8,n)
    d['target']=rng.binomial(1,.14,n);d.loc[d.index[::7],'re_price_log_posterior']=np.nan;checks=[]
    for spec in specs:
        model,audit=fit_spline(d.iloc[:1300],spec);query=d.iloc[1300:].copy();p=predict(model,query)
        query.target=1-query.target;assert np.array_equal(p,predict(model,query))
        stream=io.BytesIO();joblib.dump(model,stream);stream.seek(0);assert np.array_equal(p,predict(joblib.load(stream),query))
        assert np.isfinite(p).all() and np.all((p>0)&(p<1));checks.append(dict(experiment=spec['id'],parameters=audit['coefficients'],iterations=audit['iterations']))
    dump(dict(status='PASS',synthetic_only=True,label_isolation_checked=True,checkpoint_replay_exact=True,branches=checks,source_sha256=sha(__file__)),V/'anchored_spline/reports/PREFLIGHT.json')
    print('ANCHORED_SPLINE_PREFLIGHT_PASS',flush=True)


def main(ident):
    global FRAME
    from conservative_adapters import main as evaluate
    spec=next(s for s in json.loads((V/'anchored_spline_plan.json').read_text()) if s['id']==ident)
    receipt=json.loads((V/'anchored_spline/reports/PREFLIGHT.json').read_text());build=json.loads((V/'anchored_spline/reports/FEATURE_BUILD.json').read_text())
    assert receipt['status']=='PASS' and receipt['source_sha256']==sha(__file__)
    assert build['sha256']==sha(V/'anchored_spline/data/inputs.parquet') and build['cross_fold_count_invariance']
    FRAME=pd.read_parquet(V/'anchored_spline/data/inputs.parquet').set_index('application_id');assert FRAME.index.is_unique and 'target' not in FRAME
    return evaluate(ident,plan_file='anchored_spline_plan.json',adapter_fit=fit_adapter,adapter_probability=probability,
        method_formulas={spec['method']:spec['formula']},spectral_methods=[],worker_source='anchored_spline_worker.py',
        final_prediction=final_prediction,postprocess=postprocess)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run');parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    try:
        if args.preflight:preflight()
        else:assert args.run;main(args.run)
    except Exception:
        if args.run:dump(dict(status='FAILED',traceback=traceback.format_exc()),V/'runs'/args.run/'STATUS.json')
        raise
