"""Small interaction representations fitted only on V25 correction-training rows."""
import numpy as np
from scipy.optimize import minimize, check_grad
from scipy.special import expit
from threadpoolctl import threadpool_limits
from baseline import prepare, design

PAIRS = [
 ('risk_conditional_negative_months','installment_observed_income_ratio'),
 ('risk_conditional_nonnegative_months','installment_observed_income_ratio'),
 ('risk_conditional_negative_months','re_late_30_posterior'),
 ('risk_conditional_nonnegative_months','re_late_30_posterior'),
 ('aux_allocation_count','re_late_30_posterior'),
 ('aux_allocation_count','re_late_1_posterior'),
 ('re_bureau_posterior_sd','re_bureau_score_posterior'),
 ('re_price_posterior_sd','re_price_log_posterior'),
 ('re_income_geometric_mean','re_declared_income_log_gap'),
 ('re_outflow_income_geometric_ratio','v2_existing_total_burden_observed'),
 ('re_late_30_posterior','re_bureau_score_posterior'),
 ('risk_conditional_negative_months','re_declared_income_log_gap'),
]


def gradient_check():
    rng=np.random.default_rng(123);u=rng.normal(size=(23,7));x=rng.normal(size=(23,5));z=rng.normal(size=23);y=rng.integers(0,2,23);rank=3
    n=x.shape[1];theta=rng.normal(0,.1,n+u.shape[1]*rank)
    def objective(t):
        b=t[:n];v=t[n:].reshape(u.shape[1],rank);uv=u@v
        eta=z+x@b+.5*np.sum(uv**2-(u*u)@(v*v),axis=1);res=(expit(eta)-y)/len(y)
        loss=np.mean(np.logaddexp(0,eta)-y*eta)+.005*np.sum(t*t)
        gb=x.T@res+.01*b;gv=u.T@(res[:,None]*uv)-((u*u).T@res)[:,None]*v+.01*v
        return loss,np.r_[gb,gv.ravel()]
    error=float(check_grad(lambda t:objective(t)[0],lambda t:objective(t)[1],theta,epsilon=1e-6))
    assert error<2e-6,error
    return dict(status='PASS',finite_difference_gradient_error=error)


def inputs(frame,base,ref=None):
    cat=base['categorical'];nums=[c for c in base['columns'] if c not in cat]
    d=prepare(frame,base['columns'],cat);raw=d[nums].to_numpy(float)
    missing=~np.isfinite(raw)
    if ref is None:
        med=np.nanmedian(raw,axis=0);med=np.where(np.isfinite(med),med,0.)
        fill=np.where(missing,med,raw);mean=fill.mean(0);sd=fill.std(0);sd=np.where(sd>1e-6,sd,1.)
        ref=dict(median=med,mean=mean,sd=sd,numeric=nums,categories={c:sorted(d[c].unique()) for c in cat})
    fill=np.where(missing,ref['median'],raw)
    u=np.clip((fill-ref['mean'])/ref['sd'],-5,5)
    parts=[u,missing.astype(float)]
    for c in cat:parts.append(np.column_stack([(d[c]==level).to_numpy(float) for level in ref['categories'][c]]))
    return np.column_stack(parts),ref


def fit(f,spec):
    base=f['head'];x,z=f['basis']['correction'];y=f['h'].target.to_numpy(float)
    u,ref=inputs(f['h'],base);kind=spec['representation'];l2=spec['penalty'];reg=spec['interaction_penalty']
    b0=base['coefficients'].copy();nbase=len(b0)
    metadata=dict(kind=kind,base_head=base,input_reference=ref,spec=dict(correction_strength=spec['strength'],penalty=l2),columns=base['columns'],categorical=base['categorical'])
    if kind=='pairs':
        indices=[(ref['numeric'].index(a),ref['numeric'].index(b)) for a,b in PAIRS]
        raw=np.column_stack([u[:,a]*u[:,b] for a,b in indices]);mean=raw.mean(0);sd=raw.std(0);sd=np.where(sd>1e-6,sd,1.)
        q=(raw-mean)/sd;w=np.column_stack([x,q]);penalty=np.r_[0.,np.full(nbase-1,l2),np.full(len(indices),reg)]
        def objective(theta):
            eta=z+w@theta;res=(expit(eta)-y)/len(y)
            loss=np.mean(np.logaddexp(0,eta)-y*eta)+.5*np.sum(penalty*theta**2)
            return loss,w.T@res+penalty*theta
        initial=np.r_[b0,np.zeros(len(indices))]
        metadata.update(pair_indices=indices,pair_names=PAIRS,pair_mean=mean,pair_sd=sd)
    else:
        assert kind=='fm'
        rank=spec['rank'];rng=np.random.default_rng(432026+f['fold'])
        initial=np.r_[b0,rng.normal(0,.03,size=(u.shape[1],rank)).ravel()]
        penalty=np.r_[0.,np.full(nbase-1,l2)]
        def objective(theta):
            b=theta[:nbase];v=theta[nbase:].reshape(u.shape[1],rank);uv=u@v
            interaction=.5*np.sum(uv**2-(u**2)@(v**2),axis=1)
            eta=z+x@b+interaction;res=(expit(eta)-y)/len(y)
            loss=np.mean(np.logaddexp(0,eta)-y*eta)+.5*np.sum(penalty*b**2)+.5*reg*np.sum(v*v)
            gb=x.T@res+penalty*b
            gv=u.T@(res[:,None]*uv)-((u*u).T@res)[:,None]*v+reg*v
            return loss,np.r_[gb,gv.ravel()]
        metadata.update(rank=rank,input_width=u.shape[1])
    with threadpool_limits(limits=2):
        optimized=minimize(objective,initial,method='L-BFGS-B',jac=True,options=dict(maxiter=600,ftol=1e-10,gtol=2e-6))
        loss,grad=objective(optimized.x)
    assert np.isfinite(optimized.x).all() and np.max(abs(grad))<2e-4,(optimized.message,np.max(abs(grad)))
    metadata.update(coefficients=optimized.x,base_coefficients=nbase,interaction_penalty=reg,objective=float(loss),gradient_max_abs=float(np.max(abs(grad))),iterations=int(optimized.nit),optimizer_success=bool(optimized.success),input_features_unchanged=True)
    return metadata


def predict(bundle,frame,p):
    x,z=design(bundle['base_head'],frame,p);u,_=inputs(frame,bundle['base_head'],bundle['input_reference'])
    theta=bundle['coefficients'];n=bundle['base_coefficients'];delta=x@theta[:n]
    if bundle['kind']=='pairs':
        raw=np.column_stack([u[:,a]*u[:,b] for a,b in bundle['pair_indices']]);q=(raw-bundle['pair_mean'])/bundle['pair_sd']
        delta+=q@theta[n:]
    else:
        v=theta[n:].reshape(u.shape[1],bundle['rank']);uv=u@v
        delta+=.5*np.sum(uv**2-(u*u)@(v*v),axis=1)
    return expit(z+bundle['spec']['correction_strength']*delta),delta
