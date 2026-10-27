"""Compact, prespecified stability experiments for the frozen Kumo pipeline."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,io,json,shutil,sys,traceback
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit,logit
from scipy.optimize import check_grad
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer,MissingIndicator
from sklearn.preprocessing import SplineTransformer,StandardScaler,OneHotEncoder,QuantileTransformer
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'
sys.path[:0]=[str(ROOT/'scripts'),str(V/'scripts')]
from baseline import prepare,design,corrected_probability,dump,sha,scores
from experiments import fit_head,bootstrap
from conservative_adapters import main as evaluate,solve,group_objective,SPECTRAL
from joint_followups import local_cache,generalization_review
from modeling_features import error_analysis


def quadratic_objective(beta,x,z,y,penalty):
    eta=z+x@beta
    return float(np.mean(np.logaddexp(0,eta)-y*eta)+.5*beta@penalty@beta),x.T@(expit(eta)-y)/len(y)+penalty@beta


def solve_quadratic_newton(initial,x,z,y,penalty):
    beta=initial.copy();backtracks=0
    with threadpool_limits(limits=2):
        for iteration in range(60):
            value,gradient=quadratic_objective(beta,x,z,y,penalty)
            if np.max(abs(gradient))<1e-7:break
            p=expit(z+x@beta);hessian=(x.T*(p*(1-p)))@x/len(y)+penalty
            step=np.linalg.solve(hessian,gradient);descent=float(gradient@step);assert descent>0
            length=1.
            for _ in range(40):
                candidate=beta-length*step;new_value,_=quadratic_objective(candidate,x,z,y,penalty)
                if np.isfinite(new_value) and new_value<=value-1e-4*length*descent:break
                length*=.5;backtracks+=1
            else:raise RuntimeError('Quadratic-head Newton line search failed')
            beta=candidate
        value,gradient=quadratic_objective(beta,x,z,y,penalty)
    assert np.max(abs(gradient))<1e-6,('Newton not converged',np.max(abs(gradient)))
    return beta,dict(objective=value,gradient_max_abs=float(np.max(abs(gradient))),iterations=iteration+1,solver='Analytic Hessian Newton with Armijo backtracking',backtracks=backtracks)


def jeffreys_objective(beta,x,z,y,penalty,information_ridge):
    eta=z+x@beta;p=expit(eta);w=p*(1-p);n=len(y)
    info=(x.T*w)@x/n+information_ridge*np.eye(x.shape[1])
    sign,logdet=np.linalg.slogdet(info);assert sign>0
    inverse=np.linalg.inv(info)
    leverage=w*np.sum((x@inverse)*x,axis=1)/n
    dev=beta.copy();dev[0]=0.
    value=np.mean(np.logaddexp(0,eta)-y*eta)+.5*penalty*np.dot(dev,dev)-.5*logdet/n
    grad=x.T@(p-y-.5*leverage*(1-2*p))/n+penalty*dev
    return float(value),grad


def alternate_head(h,k,columns,spec):
    cats=h[columns].select_dtypes(include=['object','string','category']).columns.tolist();nums=[c for c in columns if c not in cats]
    steps=[('fill',SimpleImputer(strategy='median',keep_empty_features=True))]
    if spec['method']=='gaussian_rank':steps.append(('rank',QuantileTransformer(n_quantiles=min(100,len(h)),output_distribution='normal',subsample=None,random_state=54027)))
    steps.extend([('basis',SplineTransformer(n_knots=spec.get('n_knots',3),degree=2,knots='quantile',extrapolation='constant')),('scale',StandardScaler())])
    encoder=ColumnTransformer([('smooth',Pipeline(steps),nums),('missing',MissingIndicator(features='all',sparse=False,error_on_new=False),nums),('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])
    basis=encoder.fit_transform(prepare(h,columns,cats));z=logit(np.clip(k,1e-6,1-1e-6));mean=float(z.mean());sd=max(float(z.std()),1e-8)
    x=np.column_stack([np.ones(len(h)),(z-mean)/sd,basis]);pen=np.full(x.shape[1],.003);pen[0]=0.
    beta,audit=solve(group_objective,np.zeros(x.shape[1]),(x,z,h.target.to_numpy(int),pen))
    return dict(encoder=encoder,columns=columns,categorical=cats,offset_mean=mean,offset_sd=sd,coefficients=beta,spec=dict(correction_strength=.5,penalty=.003),positive_weight=1.,**audit)


def fit_adapter(f,spec,columns):
    h=f['h'];y=h.target.to_numpy(int);k=f['kh'];model=dict(kind=spec['method'],spec=spec)
    cache_audit=[]
    if spec['input_source']=='V178':
        if f.get('synthetic_preflight'):
            assert h.application_id.str.startswith('SYNTHETIC_').all();local=f['synthetic_local']
        else:local,cache_audit=local_cache(f)
        model['local']=local;k=local.reindex(h.application_id).to_numpy(float)
    method=spec['method']
    if method in ['reduced_knots','gaussian_rank']:head=alternate_head(h,k,columns,spec)
    else:head=fit_head(h,k,columns,.003,.5)
    x,z=design(head,h,k);audit=dict(gradient_max_abs=head['gradient_max_abs'],iterations=head['iterations'],raw_source=spec['input_source'],source_cache=cache_audit)
    if method=='coefficient_smoothing':
        penalty=.003*np.eye(x.shape[1]);penalty[0,0]=0.;smooth=head['encoder'].named_transformers_['smooth'];n=len(head['encoder'].transformers_[0][2]);width=smooth.named_steps['basis'].n_features_out_//n;scale=smooth.named_steps['scale'].scale_;rows=[]
        for j in range(n):
            for q in range(width-2):
                row=np.zeros(x.shape[1]);ix=j*width+q;row[2+ix:2+ix+3]=np.array([1.,-2.,1.])/scale[ix:ix+3];rows.append(row)
        difference=np.array(rows);penalty+=spec['roughness_penalty']*(difference.T@difference)
        head['coefficients'],optimization=solve_quadratic_newton(head['coefficients'],x,z,y,penalty);audit.update(optimization,roughness_rows=len(rows));model['quadratic_penalty']=penalty
    elif method=='jeffreys_stabilized':
        head['coefficients'],optimization=solve(jeffreys_objective,head['coefficients'],(x,z,y,.003,spec['information_ridge']));audit.update(optimization,exact_classical_Firth=False,information_ridge=spec['information_ridge'])
    elif method=='laplace_predictive':
        p=expit(z+x@head['coefficients']);pen=np.full(x.shape[1],len(h)*.003);pen[0]=0.
        precision=(x.T*(p*(1-p)))@x+np.diag(pen)+1e-6*np.eye(x.shape[1]);covariance=np.linalg.inv(precision)
        assert np.linalg.eigvalsh(covariance).min()>0;model['coefficient_covariance']=covariance;audit.update(conditional_posterior_only=True,precision_condition_number=float(np.linalg.cond(precision)),approximate_logistic_normal_integral=True)
    model['head']=head
    return model,audit,len(head['coefficients'])


def probability(model,frame,k,base_head):
    if 'local' in model:k=model['local'].reindex(frame.application_id).to_numpy(float)
    assert np.isfinite(k).all()
    head=model['head'];p,d=corrected_probability(head,frame,k)
    if model['kind']=='laplace_predictive':
        x,z=design(head,frame,k);eta=z+.5*x@head['coefficients'];variance=.25*np.maximum(np.sum((x@model['coefficient_covariance'])*x,axis=1),0.)
        eta=eta/np.sqrt(1+np.pi*variance/8);return expit(eta),eta-z
    return p,d


def postprocess(out,result,tr,y,folds,reference,pred,frames):
    generalization_review(out,result,tr,y,folds,reference,pred,frames)
    baseline=pd.read_csv(V/'runs/V193/OOF.csv',float_precision='round_trip');assert baseline.application_id.tolist()==tr.application_id.tolist();p=baseline.prediction.to_numpy();b=scores(y,p);ci=bootstrap(y,p,pred,folds)
    wins=sum(scores(y[folds==i],pred[folds==i])['KS']>scores(y[folds==i],p[folds==i])['KS'] for i in range(5))
    comp=dict(control='V193',control_metrics=b,candidate_metrics=result['OOF'],delta_KS=result['OOF']['KS']-b['KS'],delta_AUC=result['OOF']['AUC']-b['AUC'],fold_wins=int(wins),confidence_interval=ci);result['comparison_to_V193']=comp;dump(comp,out/'MATCHED_V193.json')
    details=out/'vs_V193';details.mkdir(exist_ok=True);errors=error_analysis(details,tr,y,folds,p,pred,frames)
    for suffix in ['json','md']:shutil.copyfile(details/('ERROR_ANALYSIS.'+suffix),out/('ERROR_ANALYSIS_V193.'+suffix))
    gap=float(np.mean([f['train_validation_head_KS_gap'] for f in result['folds']]));findings=[]
    if comp['delta_KS']>0 and comp['delta_AUC']>=0:findings.append('POOLED_KS_AND_AUC_IMPROVE_ON_REUSED_DEVELOPMENT_FOLDS')
    elif comp['delta_KS']>0:findings.append('KS_GAIN_WITH_AUC_TRADEOFF')
    elif comp['delta_AUC']>0:findings.append('BROAD_RANKING_GAIN_WITHOUT_KS_GAIN')
    else:findings.append('NO_GAIN_AGAINST_DEVELOPMENT_CONTROL')
    if gap>.016:findings.append('CORRECTION_HEAD_GAP_REMAINS_HIGH')
    if ci['delta_KS'][0]<=0:findings.append('CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT')
    insight=dict(experiment=result['experiment'],method=result['spec']['method'],hypothesis=result['spec']['rationale'],formula=result['spec']['formula'],findings=findings,comparison_to_V193=comp,head_gap=gap,top_risk_capture=errors['risk_cutoffs'],next_action='Inspect source-specific errors and change one mechanism; do not promote by pooledKS alone or tune on outer labels.',public_incumbent='V25',independent_generalization_established=False)
    result['continuous_insight']=insight;dump(insight,out/'CONTINUOUS_INSIGHT.json')
    with (out/'REPORT.md').open('a') as f:f.write(f"\n\nV193: deltaKS={comp['delta_KS']:+.10f}, deltaAUC={comp['delta_AUC']:+.10f}, wins={wins}/5, paired deltaKS CI={ci['delta_KS']}.\n\nFindings: {', '.join(findings)}.\n")


def preflight():
    rng=np.random.default_rng(54027);x=rng.normal(size=(150,7));x[:,0]=1.;z=rng.normal(-1,1,len(x));y=rng.binomial(1,expit(z));beta=rng.normal(0,.1,x.shape[1]);penalty=.01*np.eye(x.shape[1])
    checks=[]
    for name,fun,args in [('quadratic',quadratic_objective,(x,z,y,penalty)),('stabilized_Jeffreys',jeffreys_objective,(x,z,y,.003,.0001))]:
        err=check_grad(lambda b:fun(b,*args)[0],lambda b:fun(b,*args)[1],beta);assert err<1e-5,(name,err);checks.append(dict(method=name,gradient_error=float(err)))
    n=1200;h=pd.DataFrame(dict(application_id=[f'SYNTHETIC_{i}' for i in range(n)],a=rng.lognormal(size=n),b=rng.lognormal(size=n),employment_type=np.resize(['salaried','self_employed','variable_income'],n)))
    h['aux_allocation_count']=rng.integers(0,20,n);k=expit(rng.normal(-1,1,n));h['target']=rng.binomial(1,k)
    for c in SPECTRAL:h[c]=rng.random(n)
    basecols=['a','b','employment_type'];head=fit_head(h,k,basecols,.003,.5);f=dict(h=h,kh=k,head=head,fold=0,synthetic_preflight=True,synthetic_local=pd.Series(expit(logit(k)+.05),index=h.application_id));branches=[]
    for spec in json.loads((V/'stability_plan.json').read_text()):
        cols=basecols+(SPECTRAL if spec['spectral_inputs']=='stable9' else []);model,audit,count=fit_adapter(f,spec,cols);q=h.iloc[:100].copy();q.employment_type='UNSEEN';p,d=probability(model,q,k[:100],head);assert np.isfinite(p).all() and np.all((p>0)&(p<1))
        stream=io.BytesIO();joblib.dump(model,stream);stream.seek(0);again,_=probability(joblib.load(stream),q,k[:100],head);assert np.array_equal(p,again);branches.append(dict(method=spec['method'],parameters=count,unseen_employment_safe=True,checkpoint_replay_exact=True))
    dump(dict(status='PASS',synthetic_only=True,gradient_checks=checks,branches=branches,source_sha256=sha(__file__)),V/'stability/reports/PREFLIGHT.json');print('STABILITY_PREFLIGHT_PASS',flush=True)


def main(ident):
    spec=next(s for s in json.loads((V/'stability_plan.json').read_text()) if s['id']==ident);pre=json.loads((V/'stability/reports/PREFLIGHT.json').read_text());assert pre['status']=='PASS' and pre['source_sha256']==sha(__file__)
    methods=[spec['method']] if spec['spectral_inputs']=='stable9' else []
    return evaluate(ident,plan_file='stability_plan.json',adapter_fit=fit_adapter,adapter_probability=probability,method_formulas={spec['method']:spec['formula']},spectral_methods=methods,spectral_columns=SPECTRAL,worker_source='stability_worker.py',postprocess=postprocess)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run');parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    try:
        if args.preflight:preflight()
        else:assert args.run;main(args.run)
    except Exception:
        if args.run:dump(dict(status='FAILED',traceback=traceback.format_exc()),V/'runs'/args.run/'STATUS.json')
        raise
