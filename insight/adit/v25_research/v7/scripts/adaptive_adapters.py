"""Small statistical follow-ups; Kumo/context and outer folds stay frozen."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='2'
from pathlib import Path
import argparse,hashlib,json,sys,traceback,io
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize,check_grad
from scipy.special import expit,logit,logsumexp
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer,MissingIndicator
from sklearn.preprocessing import SplineTransformer,StandardScaler
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'
sys.path[:0]=[str(ROOT/'scripts'),str(V/'scripts')]
from baseline import design,corrected_probability,dump,sha
from experiments import fit_head
from conservative_adapters import SPECTRAL,solve,signed,residual_objective,main as evaluate


def inner_predictions(f,columns):
    h=f['h'];p=np.full(len(h),np.nan);assign=np.full(len(h),-1);heads=[];checks=[]
    splitter=StratifiedKFold(3,shuffle=True,random_state=52027+f['fold'])
    for block,(ti,qi) in enumerate(splitter.split(h,h.target)):
        assert len(ti)==4000 and len(qi)==2000
        assert not np.intersect1d(ti,qi).size
        head=fit_head(h.iloc[ti],f['kh'][ti],columns,.003,.5)
        p[qi],_=corrected_probability(head,h.iloc[qi],f['kh'][qi]);assign[qi]=block
        heads.append(head)
        checks.append(dict(block=block,fit_rows=len(ti),query_rows=len(qi),disjoint=True,
            fit_ids_sha256=hashlib.sha256('\n'.join(h.application_id.iloc[ti].astype(str)).encode()).hexdigest(),
            query_ids_sha256=hashlib.sha256('\n'.join(h.application_id.iloc[qi].astype(str)).encode()).hexdigest()))
    assert np.isfinite(p).all() and np.all(assign>=0)
    return p,heads,assign,checks


def affine_objective(t,z,y,penalty):
    eta=t[0]+t[1]*z;res=expit(eta)-y;v=t-np.array([0.,1.])
    return float(np.mean(np.logaddexp(0,eta)-y*eta)+.5*penalty*np.dot(v,v)),np.array([res.mean(),np.mean(res*z)])+penalty*v


def group_labels(frame):
    n=frame.aux_allocation_count.to_numpy(float)
    history=np.select([~np.isfinite(n),n<=0,n<=3],['missing','zero','1_to3'],default='over3')
    employment=frame.employment_type.fillna('missing').astype(str).to_numpy()
    return employment,history


def group_matrix(frame,levels):
    employment,history=group_labels(frame)
    return np.column_stack([np.ones(len(frame))]+[(values==level).astype(float) for values,labels in [(employment,levels['employment']),(history,levels['history'])] for level in labels])


def calibration_group_objective(beta,x,z,y,penalty):
    eta=z+x@beta
    return float(np.mean(np.logaddexp(0,eta)-y*eta)+.5*np.dot(penalty,beta*beta)),x.T@(expit(eta)-y)/len(y)+penalty*beta


def regret_objective(beta,x,z,y,groups,penalty,tau):
    eta=z+x@beta;loss=np.logaddexp(0,eta)-y*eta
    reference=np.logaddexp(0,z)-y*z
    unique=np.unique(groups);regret=np.array([np.mean((loss-reference)[groups==g]) for g in unique])
    weights=np.exp(regret/tau-logsumexp(regret/tau))
    grad=np.zeros(x.shape[1]);res=expit(eta)-y
    for g,w in zip(unique,weights):
        mask=groups==g;grad+=w*(x[mask].T@res[mask])/mask.sum()
    p=beta.copy();p[0]=0.
    return float(tau*(logsumexp(regret/tau)-np.log(len(unique)))+.5*penalty*np.dot(p,p)),grad+penalty*p


def spectral_basis(frame,encoder,missing,fit=False):
    a=signed(frame,SPECTRAL)
    if fit:
        basis=encoder.fit_transform(a);missing.fit(a)
    else:basis=encoder.transform(a)
    return np.column_stack([basis,missing.transform(a)])


def fit_adapter(f,spec,columns):
    method=spec['method'];h=f['h'];y=h.target.to_numpy(int)
    teacher,_=corrected_probability(f['head'],h,f['kh']);z=logit(np.clip(teacher,1e-6,1-1e-6))
    model=dict(kind=method,spec=spec);audit={}
    if method in ['crossfit_calibration','delete_block_bagging','partial_pool_group_calibration']:
        inner,heads,assign,checks=inner_predictions(f,columns)
        zi=logit(np.clip(inner,1e-6,1-1e-6))
        model['inner_checks']=checks;model['inner_fold_assignment']=assign
        audit.update(inner_checks=checks,inner_training_targets_never_in_own_prediction=True,no_spline_used_for_inner_calibration=True)
    if method=='delete_block_bagging':
        model['heads']=heads;parameters=sum(len(p['coefficients']) for p in heads)
        audit.update(gradient_max_abs=max(p['gradient_max_abs'] for p in heads),iterations=sum(p['iterations'] for p in heads))
    elif method=='crossfit_calibration':
        with threadpool_limits(limits=2):
            fit=minimize(affine_objective,np.array([0.,1.]),args=(zi,y,spec['calibration_penalty']),jac=True,method='L-BFGS-B',bounds=[(-.5,.5),(.5,1.5)],options=dict(maxiter=300,ftol=1e-12,gtol=1e-8))
        assert fit.success and .5<=fit.x[1]<=1.5
        model['affine']=fit.x;parameters=2
        # L-BFGS-B can validly stop at a constrained boundary with nonzero raw gradient.
        gradient=affine_objective(fit.x,zi,y,spec['calibration_penalty'])[1]
        projected=gradient.copy()
        for j,(lo,hi) in enumerate([(-.5,.5),(.5,1.5)]):
            if (fit.x[j]<=lo+1e-8 and gradient[j]>0) or (fit.x[j]>=hi-1e-8 and gradient[j]<0):projected[j]=0.
        assert max(abs(projected))<1e-4
        audit.update(affine_intercept=float(fit.x[0]),affine_slope=float(fit.x[1]),gradient_max_abs=float(max(abs(projected))),iterations=int(fit.nit),objective=float(fit.fun))
    elif method=='partial_pool_group_calibration':
        emp,hist=group_labels(h);levels=dict(employment=sorted(set(emp)),history=sorted(set(hist)))
        x=group_matrix(h,levels);pen=np.full(x.shape[1],spec['group_penalty']);pen[0]=spec['global_penalty']
        beta,optimization=solve(calibration_group_objective,np.zeros(x.shape[1]),(x,zi,y,pen))
        probabilities=expit(zi+x@beta);curvature=(x.T*(probabilities*(1-probabilities)))@x/len(y)+np.diag(pen)
        # Penalized local curvature is only a conditional diagnostic, not a full Bayesian posterior.
        model.update(levels=levels,coefficients=beta,conditional_curvature_sd=np.sqrt(np.diag(np.linalg.inv(curvature))/len(y)))
        audit.update(optimization,group_counts={name:{level:int(np.sum(values==level)) for level in labels} for name,values,labels in [('employment',emp,levels['employment']),('history',hist,levels['history'])]})
        parameters=len(beta)
    elif method=='orthogonal_spectral_residual':
        encoder=Pipeline([('fill',SimpleImputer(strategy='median',keep_empty_features=True)),('basis',SplineTransformer(n_knots=3,degree=2,knots='quantile',extrapolation='constant')),('scale',StandardScaler())])
        missing=MissingIndicator(features='all',sparse=False,error_on_new=False)
        spectral=spectral_basis(h,encoder,missing,True);base,_=design(f['head'],h,f['kh'])
        gram=base.T@base/len(h)+spec['projection_ridge']*np.eye(base.shape[1]);projection=np.linalg.solve(gram,base.T@spectral/len(h));residual=spectral-base@projection
        beta,optimization=solve(residual_objective,np.zeros(residual.shape[1]),(residual,z,y,spec['new_penalty'],spec['teacher_tether']))
        model.update(encoder=encoder,missing=missing,projection=projection,coefficients=beta)
        audit.update(optimization,projection_rows=len(h),projection_condition_number=float(np.linalg.cond(gram)),projection_uses_target=False,residual_basis_columns=residual.shape[1])
        parameters=len(beta)
    else:
        assert method=='group_regret_robust_head'
        head=fit_head(h,f['kh'],columns,spec['penalty'],.5);x,offset=design(head,h,f['kh'])
        emp,hist=group_labels(h);keys=np.array([a+'|'+('observed' if b not in ['zero','missing'] else b) for a,b in zip(emp,hist)])
        counts={s:int(np.sum(keys==s)) for s in set(keys)};groups=np.array([s if counts[s]>=spec['min_group_rows'] else 'OTHER' for s in keys])
        head['coefficients'],optimization=solve(regret_objective,head['coefficients'],(x,offset,y,groups,spec['penalty'],spec['robust_temperature']))
        model['head']=head;audit.update(optimization,group_counts={s:int(np.sum(groups==s)) for s in set(groups)},rare_groups_merged=True,regret_reference='Frozen Kumo margin without correction')
        parameters=len(head['coefficients'])
    return model,audit,parameters


def probability(model,frame,kumo,base_head):
    method=model['kind'];spec=model['spec']
    p,d=corrected_probability(base_head,frame,kumo);z=logit(np.clip(p,1e-6,1-1e-6))
    if method=='delete_block_bagging':
        p=np.mean([corrected_probability(head,frame,kumo)[0] for head in model['heads']],axis=0)
        return p,logit(np.clip(p,1e-6,1-1e-6))-logit(np.clip(kumo,1e-6,1-1e-6))
    if method=='group_regret_robust_head':return corrected_probability(model['head'],frame,kumo)
    if method=='crossfit_calibration':
        a,b=model['affine'];delta=spec['calibration_strength']*(a+(b-1)*z)
    elif method=='partial_pool_group_calibration':
        delta=spec['calibration_strength']*np.clip(group_matrix(frame,model['levels'])@model['coefficients'],-spec['residual_cap'],spec['residual_cap'])
    else:
        assert method=='orthogonal_spectral_residual'
        spectral=spectral_basis(frame,model['encoder'],model['missing']);base,_=design(base_head,frame,kumo)
        delta=spec['residual_strength']*((spectral-base@model['projection'])@model['coefficients'])
    return expit(z+delta),d+delta


def preflight():
    rng=np.random.default_rng(20261027);n=510;x=rng.normal(size=(n,9));x[:,0]=1.;z=rng.normal(-1.,1.,n);y=rng.binomial(1,expit(z));beta=rng.normal(0,.05,x.shape[1]);groups=np.arange(n)%6
    jobs=[('affine',affine_objective,np.array([.1,1.1]),(z,y,.02)),('partial_pool',calibration_group_objective,beta,(x,z,y,np.full(x.shape[1],.04))),('group_regret',regret_objective,beta,(x,z,y,groups,.003,.05)),('spectral_residual',residual_objective,beta,(x,z,y,.03,.2))]
    checks=[]
    for name,fun,b,args in jobs:
        error=check_grad(lambda v:fun(v,*args)[0],lambda v:fun(v,*args)[1],b)
        assert error<1e-5,(name,error);checks.append(dict(objective=name,finite_difference_gradient_error=float(error)))
    synthetic=pd.DataFrame({f'x{j}':rng.lognormal(size=6000) for j in range(3)})
    synthetic['employment_type']=np.resize(['salaried','self_employed','variable_income'],6000)
    synthetic['application_id']=[f'synthetic_{i}' for i in range(6000)]
    k=expit(rng.normal(-1.,1.,6000));synthetic['target']=rng.binomial(1,k)
    f=dict(h=synthetic,kh=k,fold=0);inner,heads,assignment,isolation=inner_predictions(f,['x0','x1','x2','employment_type'])
    assert len(heads)==3 and len(np.unique(assignment))==3 and np.isfinite(inner).all()
    before=np.argsort(z);after=np.argsort(.1+1.1*z);assert np.array_equal(before,after),'Positive affine calibration must preserve within-score ranking'
    synthetic['aux_allocation_count']=rng.choice([0.,1.,3.,10.,np.nan],6000)
    for name in SPECTRAL:synthetic[name]=rng.random(6000)
    basecols=['x0','x1','x2','employment_type']
    f['head']=fit_head(synthetic,k,basecols,.003,.5)
    query=synthetic.iloc[:100].copy();query['employment_type']='UNSEEN'
    branch_checks=[]
    for spec in json.loads((V/'adaptive_plan.json').read_text()):
        model,audit,nparams=fit_adapter(f,spec,basecols)
        prediction,delta=probability(model,query,k[:100],f['head'])
        assert np.isfinite(prediction).all() and np.all((prediction>0)&(prediction<1)) and np.isfinite(delta).all()
        buffer=io.BytesIO();joblib.dump(model,buffer);buffer.seek(0)
        replay,_=probability(joblib.load(buffer),query,k[:100],f['head'])
        assert np.array_equal(prediction,replay),'Serialized checkpoint must replay exactly'
        branch_checks.append(dict(method=spec['method'],parameters=nparams,unseen_employment_predicts=True,checkpoint_replay_exact=True))
    passed=dict(status='PASS',synthetic_only=True,real_competition_scores_not_evaluated=True,gradient_checks=checks,branch_checks=branch_checks,inner_isolation=isolation,positive_affine_rank_preserved=True,source_sha256=sha(__file__))
    dump(passed,V/'adaptive/reports/PREFLIGHT.json');print(json.dumps(passed),flush=True)


def main(ident):
    spec=next(s for s in json.loads((V/'adaptive_plan.json').read_text()) if s['id']==ident)
    preflight=json.loads((V/'adaptive/reports/PREFLIGHT.json').read_text());assert preflight['status']=='PASS' and preflight['source_sha256']==sha(__file__)
    return evaluate(ident,plan_file='adaptive_plan.json',adapter_fit=fit_adapter,adapter_probability=probability,method_formulas={spec['method']:spec['formula']},spectral_methods=['orthogonal_spectral_residual'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--preflight',action='store_true');args=p.parse_args()
    try:
        if args.preflight:preflight()
        else:assert args.run;main(args.run)
    except Exception:
        if args.run:dump(dict(status='FAILED',traceback=traceback.format_exc()),V/'runs'/args.run/'STATUS.json')
        raise
