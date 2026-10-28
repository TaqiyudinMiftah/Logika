"""Fold-local measurement model and an uncertainty-integrated actuarial GLM."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,betaln,digamma,polygamma
from scipy.special import roots_hermitenorm

def aggregate(index,value,n):return np.bincount(index,weights=value,minlength=n)
def divide(a,b,default=0.):
    a,b=np.broadcast_arrays(a,b)
    return np.divide(a,b,out=np.full(a.shape,default,dtype=float),where=b>0)
def normal_posterior(raw,precision,train,minimum=1e-8):
    valid=train[np.isfinite(raw[train])&(precision[train]>0)]
    assert len(valid)>20
    mean=float(raw[valid].mean())
    variance=max(float(raw[valid].var()-np.mean(1/precision[valid])),minimum)
    vv=1/(precision+1/variance)
    mm=vv*(np.nan_to_num(raw)*precision+mean/variance)
    return mm,vv,dict(prior_mean=mean,prior_variance=variance,training_measurements=len(valid))

def measurement(data,train,n):
    """No selected-application default labels are accepted by this function."""
    fit=np.zeros(n,bool);fit[train]=True;metadata={}
    p=data['prices'];order=p['order'];oc=p['order_customer'];cat=p['category'];v=p['log_price']
    nc=int(cat.max())+1;no=len(oc);row=oc[order];fm=fit[row];counts=np.bincount(cat[fm],minlength=nc)
    total=aggregate(oc,np.ones(no),n);items=np.bincount(order,minlength=no)
    alpha=divide(aggregate(cat[fm],v[fm],nc),counts,float(v[fm].mean()))
    alpha-=np.average(alpha,weights=counts)
    for _ in range(8):
        effect=divide(aggregate(row,v-alpha[cat],n),np.bincount(row,minlength=n))
        alpha=divide(aggregate(cat[fm],v[fm]-effect[row[fm]],nc),counts)
        alpha-=np.average(alpha,weights=counts)
    adjusted=v-alpha[cat];om=divide(aggregate(order,adjusted,no),items)
    df=int(np.sum((items-1)[fit[oc]]));assert df>0
    item_var=max(float(np.sum((adjusted[fm]-om[order[fm]])**2)/df),1e-8)
    cm=divide(aggregate(oc,om,n),total);residual=(om-cm[oc])**2
    df2=int(np.sum((total[train]-1).clip(0)));assert df2>0
    correction=item_var*np.sum(((1-1/total[oc])/items)[fit[oc]])
    order_var=max(float((residual[fit[oc]].sum()-correction)/df2),1e-8)
    weights=1/(order_var+item_var/items);precision=aggregate(oc,weights,n)
    raw=divide(aggregate(oc,weights*om,n),precision,np.nan)
    price,pv,prior=normal_posterior(raw,precision,train)
    metadata['price']=dict(**prior,category_centers=alpha.tolist(),item_variance=item_var,
                          order_variance=order_var,unseen_categories=int((counts==0).sum()))
    b=data['bureau'];bureau_count=np.bincount(b['customer'],minlength=n)
    pairs=b['pairs'][fit[b['pair_customer']]]
    offsets=np.array([0.,np.nanmean(pairs[:,1]-pairs[:,0]),np.nanmean(pairs[:,2]-pairs[:,0])]);offsets-=offsets.mean()
    diffs=[np.nanvar(pairs[:,a]-pairs[:,c]) for a,c in [(0,1),(0,2),(1,2)]]
    v01,v02,v12=diffs
    noise=np.maximum([(v01+v02-v12)/2,(v01+v12-v02)/2,(v02+v12-v01)/2],1.)
    weights=1/noise[b['provider']];precision=aggregate(b['customer'],weights,n)
    raw=divide(aggregate(b['customer'],weights*(b['score']-offsets[b['provider']]),n),precision,np.nan)
    bureau,bv,prior=normal_posterior(raw,precision,train,1.)
    metadata['bureau']=dict(**prior,provider_offsets=offsets.tolist(),provider_noise_variances=noise.tolist())
    r=data['repayment'];k,nn=r['k'],r['n'];valid=train[nn[train]>0];kk=k[valid];nt=nn[valid]
    population=float(kk.sum()/nt.sum());initial=np.log([max(population*12,.01),max((1-population)*12,.01)])
    def beta_loss(theta):
        a,b=np.exp(theta);return -float(np.mean(betaln(kk+a,nt-kk+b)-betaln(a,b)))
    opt=minimize(beta_loss,initial,method='L-BFGS-B',bounds=[(-7,10)]*2,
                 options=dict(maxiter=1000,ftol=1e-12))
    assert opt.success,opt.message
    a,b=np.exp(opt.x);aa=k+a;bb=nn-k+b
    late=digamma(aa)-digamma(bb);lv=polygamma(1,aa)+polygamma(1,bb)
    metadata['repayment']=dict(alpha=float(a),beta=float(b),negative_loglikelihood=float(opt.fun),
        approximation='Exact beta logit mean/variance; normal approximation when integrating default risk.')
    inc=data['income'];employment=data['employment'];income=np.empty(n);iv=np.empty(n);details={}
    groups=sorted(set(employment[train]))
    for label in [None]+groups:
        tr=train if label is None else train[employment[train]==label]
        if len(tr)<30:continue
        count=inc['diff_count'][tr].sum()
        if count<30:continue
        noise=max(float((inc['diff_square'][tr].sum()/count-(inc['diff_sum'][tr].sum()/count)**2)/2),1e-5)
        precision=inc['count']/noise
        mean,var,prior=normal_posterior(inc['mean'],precision,tr,1e-4)
        mask=np.ones(n,bool) if label is None else employment==label
        income[mask]=mean[mask];iv[mask]=var[mask]
        details['GLOBAL' if label is None else str(label)]=dict(**prior,noise_variance=noise)
    metadata['income']=details
    means=np.column_stack([price,bureau,late,income]);variance=np.column_stack([pv,bv,lv,iv])
    assert np.isfinite(means).all() and np.isfinite(variance).all() and (variance>=0).all()
    missing=np.column_stack([total==0,bureau_count==0,nn==0,inc['count']==0]).astype(float)
    return means,variance,missing,metadata

def design(means,variance,missing,rows,train,cov_names):
    cov=rows['covariates'].copy();names=list(cov_names)
    for j,name in enumerate(names):
        if name not in ['age','annual_interest_rate','has_account','has_repayments']:
            cov[:,j]=np.sign(cov[:,j])*np.log1p(np.abs(cov[:,j]))
    declared=rows['declared_income'];declared=np.where(declared>0,np.log(declared),np.nan)
    cov=np.column_stack([cov,declared]);names+=['log_declared_income']
    cov_missing=~np.isfinite(cov)
    raw=np.column_stack([means,cov]);center=np.nanmedian(raw[train],axis=0)
    center=np.nan_to_num(center);q=np.nanquantile(raw[train],[.25,.75],axis=0)
    scale=q[1]-q[0];scale=np.where(np.isfinite(scale)&(scale>1e-6),scale,1.)
    standardized=(raw-center)/scale
    # Latent means are left unclipped so their variance remains consistent.
    standardized[:,4:]=np.clip(standardized[:,4:],-8,8)
    standardized=np.nan_to_num(standardized,nan=0.,posinf=8.,neginf=-8.)
    groups=sorted(set(rows['employment'][train]))
    cat=np.column_stack([rows['employment']==label for label in groups[1:]])
    x=np.column_stack([np.ones(len(raw)),standardized,missing,cov_missing,cat]).astype(float)
    v=variance/(scale[:4]**2)
    metadata=dict(center=center.tolist(),scale=scale.tolist(),employment_levels=list(map(str,groups)),
        columns=['intercept','latent_price','latent_bureau','latent_late_logit','latent_log_income']+names+
        ['missing_'+s for s in ['price','bureau','late','income']]+['missing_'+s for s in names]+['employment_'+str(s) for s in groups[1:]])
    return x,v,metadata

def integrated(beta,x,v,points=128,gradient=False):
    eta=x@beta;vv=v@(beta[1:5]**2);sd=np.sqrt(np.maximum(vv,1e-20))
    nodes,weights=roots_hermitenorm(points);weights=weights/np.sqrt(2*np.pi)
    pp=np.empty(len(x));aa=np.empty(len(x));bb=np.empty(len(x))
    for first in range(0,len(x),2048):
        sl=slice(first,first+2048);q=expit(eta[sl,None]+sd[sl,None]*nodes)
        pp[sl]=q@weights
        if gradient:
            qq=q*(1-q);aa[sl]=qq@weights;bb[sl]=qq@(weights*nodes)
    return (pp,aa,bb,sd) if gradient else pp

def objective(beta,x,v,y,ridge,integrate):
    if integrate:
        p,a,b,sd=integrated(beta,x,v,gradient=True)
    else:p=expit(x@beta)
    p=np.clip(p,1e-12,1-1e-12)
    penalty=beta.copy();penalty[0]=0
    loss=-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))+.5*ridge*np.dot(penalty,penalty)
    if integrate:
        dp=(p-y)/(p*(1-p)*len(y));g=x.T@(dp*a)
        g[1:5]+=beta[1:5]*(v.T@(dp*b/sd))
    else:g=x.T@(p-y)/len(y)
    return float(loss),g+ridge*penalty

def fit(x,v,y,integrate,initial=None):
    bounds=[(None,None)]*x.shape[1]
    for j,sign in enumerate([1,-1,1,-1],1):bounds[j]=(0,None) if sign>0 else (None,0)
    if initial is None:
        initial=np.zeros(x.shape[1]);initial[0]=np.log(y.mean()/(1-y.mean()))
    result=minimize(objective,initial,args=(x,v,y,.001,integrate),jac=True,method='L-BFGS-B',
                    bounds=bounds,options=dict(maxiter=1000,ftol=1e-12,gtol=1e-7,maxls=40))
    projected=result.jac.copy()
    for j,(lo,hi) in enumerate(bounds):
        if lo is not None and result.x[j]<=lo+1e-8 and projected[j]>0:projected[j]=0
        if hi is not None and result.x[j]>=hi-1e-8 and projected[j]<0:projected[j]=0
    norm=float(np.max(abs(projected)))
    assert result.success and norm<1e-4,(result.message,norm)
    return result.x,dict(success=True,iterations=int(result.nit),objective=float(result.fun),projected_gradient_max=norm)

def preflight():
    from scipy.optimize import check_grad
    rng=np.random.default_rng(20261028);x=rng.normal(size=(75,9));x[:,0]=1
    v=rng.uniform(.01,.3,(75,4));beta=rng.normal(0,.2,9);y=rng.integers(0,2,75)
    f=lambda z:objective(z,x,v,y,.001,True)[0]
    g=lambda z:objective(z,x,v,y,.001,True)[1]
    error=float(check_grad(f,g,beta));assert error<1e-6,error
    zero=float(np.max(abs(integrated(beta,x,np.zeros_like(v))-expit(x@beta))))
    assert zero<1e-12
    _,meta=fit(x,v,y,False);return dict(status='passed',analytic_gradient_error=error,zero_variance_equivalence_error=zero,optimizer=meta)
