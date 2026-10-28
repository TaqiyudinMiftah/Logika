"""Original fixed spline77 learner, fitted on a supplied training-only table."""
import warnings
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer,MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer,StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import baseline as b

def fit(training,columns):
    assert len(training)==56000
    assert not set(columns)&{'target','customer_id','application_id','split'}
    cats=training[columns].select_dtypes(include=['object','string','category']).columns.tolist()
    numeric=[c for c in columns if c not in cats]
    continuous=[c for c in numeric if training[c].nunique(dropna=True)>10]
    linear=[c for c in numeric if c not in continuous]
    encoder=ColumnTransformer([
        ('spline',Pipeline([('fill',SimpleImputer(strategy='median')),('basis',SplineTransformer(n_knots=5,degree=3,knots='quantile',extrapolation='constant')),('scale',StandardScaler())]),continuous),
        ('linear',Pipeline([('fill',SimpleImputer(strategy='median')),('scale',StandardScaler())]),linear),
        ('missing',MissingIndicator(features='all',sparse=False,error_on_new=False),numeric),
        ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])
    with threadpool_limits(limits=2),warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        x=encoder.fit_transform(b.prepare(training,columns,cats))
        model=LogisticRegression(C=.05,solver='newton-cholesky',max_iter=80,tol=1e-5)
        model.fit(x,training.target.astype(int),sample_weight=np.ones(len(training)))
    messages=[str(w.message)for w in captured]
    assert not any('converge' in m.lower()for m in messages),messages
    return dict(encoder=encoder,model=model,columns=columns,categorical=cats,
        fit_rows=56000,fit_ids_sha=b.ids_hash(training,np.arange(len(training))),
        resolved_parameters=model.get_params(),warnings=messages)
