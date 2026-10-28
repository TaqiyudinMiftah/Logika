"""Reproducible, label-free repayment feature factory for the frozen V25 study.

fit() receives explicit context application IDs. transform() uses only each
applicant's own pre-decision repayment history plus the fitted context priors.
Identifiers align records and never become model inputs.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln,digamma

ALIASES=[('v3_repay_all_late_'+str(t)+'_posterior_24',t,None)for t in [1,7,30,90]]+[('v3_repay_180d_late_1_posterior_24',1,180)]
MEANS=['re_late_1_posterior','re_late_30_posterior']
FORBIDDEN={'target','default_probability','default_label'}

def _beta_fit(k,n):
    valid=n>0;k=k[valid];n=n[valid]
    if not len(n):raise ValueError('Context contains no mature repayment opportunities')
    mean=float(k.sum()/n.sum())
    if not 0<mean<1:raise ValueError('Both historical event outcomes are required by the frozen prior estimator')
    def objective(theta):
        a,b=np.exp(theta);value=betaln(k+a,n-k+b)-betaln(a,b)
        ga=digamma(k+a)-digamma(n+a+b)-digamma(a)+digamma(a+b)
        gb=digamma(n-k+b)-digamma(n+a+b)-digamma(b)+digamma(a+b)
        return -float(value.mean()),-np.array([ga.mean()*a,gb.mean()*b])
    result=minimize(objective,np.log([mean*12,(1-mean)*12]),jac=True,method='L-BFGS-B',bounds=[(-7,10)]*2,options=dict(maxiter=200,ftol=1e-11,gtol=1e-7))
    if not result.success:raise RuntimeError(result.message)
    return dict(alpha=float(np.exp(result.x[0])),beta=float(np.exp(result.x[1])),objective=float(result.fun),gradient_max_abs=float(abs(result.jac).max()),iterations=int(result.nit))

def _prepare(applications,repayments):
    if FORBIDDEN&set(applications)or FORBIDDEN&set(repayments):
        raise ValueError('Provide label-free source tables to the feature factory')
    apps=applications[['application_id','customer_id','decision_datetime']].copy()
    if not apps.application_id.is_unique or not apps.customer_id.is_unique:
        raise ValueError('This dataset has one application per customer; duplicate identifiers are unsupported')
    apps.decision_datetime=pd.to_datetime(apps.decision_datetime)
    events=repayments[['customer_id','installment_id','due_date','max_days_past_due']].copy()
    events=events.merge(apps[['customer_id','decision_datetime']],on='customer_id',how='inner',validate='many_to_one')
    events.due_date=pd.to_datetime(events.due_date)
    events['age']=(events.decision_datetime-events.due_date).dt.total_seconds()/86400
    events=events[events.age>0].copy()
    if events.duplicated(['customer_id','installment_id']).any():raise ValueError('Installments must be deduplicated')
    if not np.isfinite(events.max_days_past_due).all()or not events.max_days_past_due.ge(0).all():
        raise ValueError('Eligible max DPD observations must be finite and nonnegative')
    return apps,events

def _counts(apps,events,threshold,minimum_age,window=None):
    part=events[events.age.ge(minimum_age)&(True if window is None else events.age.le(window))].copy()
    part['late']=part.max_days_past_due.ge(threshold).astype(np.int64)
    counts=part.groupby('customer_id').late.agg(['sum','count']).reindex(apps.customer_id).fillna(0)
    return counts['sum'].to_numpy(float),counts['count'].to_numpy(float)

class MaturityFeatures:
    """Two maturity30 posterior means, optionally the five eligible v3 aliases.

    Modes are frozen research recipes, not an automated feature selector.
    Existing posterior-standard-deviation columns are intentionally unchanged.
    """
    def __init__(self,mode='posterior_means_only'):
        if mode not in ['posterior_means_only','eligible_denominators']:raise ValueError(mode)
        self.mode=mode;self.parameters=None

    def fit(self,applications,repayments,fit_application_ids):
        apps,events=_prepare(applications,repayments)
        ids=pd.Index(list(fit_application_ids),name='application_id')
        if not ids.is_unique or len(ids)==0 or not ids.isin(apps.application_id).all():raise ValueError('Explicit, unique, present context IDs are required')
        positions=pd.Series(np.arange(len(apps)),index=apps.application_id).loc[ids].to_numpy()
        beta={}
        for t in [1,30]:
            k,n=_counts(apps,events,t,30);beta[str(t)]=_beta_fit(k[positions],n[positions])
        prevalence={}
        if self.mode=='eligible_denominators':
            for name,t,window in ALIASES:
                k,n=_counts(apps,events,t,t,window)
                if not n[positions].sum()>0:raise ValueError('No eligible context opportunities for '+name)
                prevalence[name]=float(k[positions].sum()/n[positions].sum())
        self.parameters=dict(schema='maturity_features_v1',mode=self.mode,mature_days=30,shrinkage=24,beta=beta,prevalence=prevalence,fit_rows=len(ids),fit_application_ids_sha256=hashlib.sha256('\n'.join(ids.astype(str)).encode()).hexdigest(),default_labels_used=False)
        return self

    def transform(self,applications,repayments):
        if self.parameters is None:raise RuntimeError('Fit or load the transformer first')
        apps,events=_prepare(applications,repayments);out=apps[['application_id']].copy()
        for t in [1,30]:
            k,n=_counts(apps,events,t,30);prior=self.parameters['beta'][str(t)]
            out[f're_late_{t}_posterior']=(k+prior['alpha'])/(n+prior['alpha']+prior['beta'])
        if self.mode=='eligible_denominators':
            for name,t,window in ALIASES:
                k,n=_counts(apps,events,t,t,window)
                out[name]=(k+24*self.parameters['prevalence'][name])/(n+24)
        return out

    def save(self,path):
        if self.parameters is None:raise RuntimeError('No fitted parameters')
        Path(path).write_text(json.dumps(self.parameters,indent=2,allow_nan=False)+'\n')

    @classmethod
    def load(cls,path):
        parameters=json.loads(Path(path).read_text())
        if parameters['schema']!='maturity_features_v1':raise ValueError('Unsupported feature schema')
        result=cls(parameters['mode']);result.parameters=parameters;return result
