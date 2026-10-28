"""Two new prediction contrasts fitted ONLY on the reserved6k correction rows."""
import hashlib
import numpy as np
from scipy.special import logit,expit
from scipy.optimize import minimize
import baseline as b

def objective(beta,x,offset,y,penalties):
 eta=offset+x@beta;n=len(y)
 value=np.mean(np.logaddexp(0,eta)-y*eta)+.5*np.sum(penalties*beta**2)
 grad=x.T@(expit(eta)-y)/n+penalties*beta
 return float(value),grad

def contrasts(kumo,classification,regression):
 log=lambda p:logit(np.clip(np.asarray(p,float),1e-6,1-1e-6))
 k=log(kumo);return np.column_stack([log(classification)-k,log(regression)-k])

def fit(base,frame,kumo,classification,regression,y,active=(0,1)):
 assert len(frame)==len(y)==6000 and frame.application_id.is_unique
 assert base['spec']['penalty']==.003 and base['spec']['correction_strength']==.5
 x,offset=b.design(base,frame,kumo);z=contrasts(kumo,classification,regression)[:,active]
 mean=z.mean(axis=0);sd=np.maximum(z.std(axis=0),.1);xx=np.column_stack([x,(z-mean)/sd])
 penalty=np.r_[np.array([0.]+[.003]*(x.shape[1]-1)),np.full(len(active),.03)]
 initial=np.r_[base['coefficients'],np.zeros(len(active))]
 result=minimize(objective,initial,args=(xx,offset,np.asarray(y,int),penalty),method='L-BFGS-B',jac=True,
  options=dict(maxiter=500,ftol=1e-11,gtol=1e-7))
 val,grad=objective(result.x,xx,offset,np.asarray(y,int),penalty)
 assert result.success and np.max(abs(grad))<1e-4,result.message
 return dict(base=base,coefficients=result.x,contrast_mean=mean,contrast_sd=sd,penalties=penalty,
  training_ids_sha256=hashlib.sha256('\n'.join(frame.application_id.astype(str)).encode()).hexdigest(),training_rows=6000,
  active_contrasts=list(active),objective=val,gradient_max_abs=float(np.max(abs(grad))),contrast_names=[['classification_minus_Kumo_log_odds','regression_minus_Kumo_log_odds'][i]for i in active],
  spec=dict(main_effect_penalty=.003,new_contrast_penalty=.03,correction_strength=.5,minimum_contrast_sd=.1,
    base_encoder_fit_on_same_reserved_correction_rows=True,meta_predictors_exclude_correction_labels=True))

def predict(bundle,frame,kumo,classification,regression):
 assert 'target'not in frame.columns
 x,offset=b.design(bundle['base'],frame,kumo);z=contrasts(kumo,classification,regression)[:,bundle['active_contrasts']]
 xx=np.column_stack([x,(z-bundle['contrast_mean'])/bundle['contrast_sd']])
 return expit(offset+.5*(xx@bundle['coefficients']))
