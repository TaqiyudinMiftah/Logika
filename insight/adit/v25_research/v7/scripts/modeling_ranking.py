"""Training-only ranking corrections tethered to the frozen V25 correction."""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from baseline import design


def fit(f,spec):
    base=f['head'];x,z=f['basis']['correction'];y=f['h'].target.to_numpy(float)
    teacher=z+.5*(x@base['coefficients']);n=len(y);n1=y.sum();n0=n-n1
    lam=spec['loss_weight'];gamma=spec['teacher_tether'];ridge=.03
    kind=spec['rankloss']
    if kind=='pairwise':
        rng=np.random.default_rng(77331+f['fold']);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
        ip=np.repeat(pos,32);im=rng.choice(neg,size=len(ip),replace=True)
        xp=x[ip]-x[im];dp=teacher[ip]-teacher[im]
    else:
        thresholds=np.quantile(teacher,[.5,.65,.8]);tau=.75
        sign=np.where(y==1,1/n1,-1/n0)
    def objective(theta):
        delta=x@theta;eta=teacher+delta
        loss=np.mean(np.logaddexp(0,eta)-y*eta)+.5*gamma*np.mean(delta*delta)+.5*ridge*np.sum(theta[1:]**2)
        grad=x.T@((expit(eta)-y+gamma*delta)/n);grad[1:]+=ridge*theta[1:]
        if kind=='pairwise':
            margins=dp+xp@theta
            loss+=lam*np.mean(np.logaddexp(0,-margins))
            grad+=lam*xp.T@(-expit(-margins)/len(margins))
        else:
            probability=expit((eta[:,None]-thresholds)/tau)
            separation=np.sum(sign[:,None]*probability,axis=0).mean()
            loss-=lam*separation
            grad-=lam*x.T@(sign*np.mean(probability*(1-probability),axis=1)/tau)
        return float(loss),grad
    check_error=None
    if f['fold']==0:
        rng=np.random.default_rng(223);initial=np.zeros(x.shape[1]);loss0,g0=objective(initial);errors=[]
        for _ in range(4):
            direction=rng.normal(size=len(initial));direction/=np.linalg.norm(direction);eps=1e-5
            numerical=(objective(initial+eps*direction)[0]-objective(initial-eps*direction)[0])/(2*eps)
            errors.append(abs(numerical-g0@direction))
        check_error=float(max(errors));assert check_error<1e-6,check_error
    with threadpool_limits(limits=2):
        optimized=minimize(objective,np.zeros(x.shape[1]),method='L-BFGS-B',jac=True,options=dict(maxiter=600,ftol=1e-11,gtol=1e-7))
        loss,grad=objective(optimized.x)
    assert optimized.success and np.max(abs(grad))<1e-4
    model=dict(base)
    model['coefficients']=base['coefficients']+optimized.x/spec['strength']
    model['spec']=dict(base['spec'],correction_strength=spec['strength'])
    model.update(gradient_max_abs=float(np.max(abs(grad))),objective=loss,iterations=int(optimized.nit),ranking_student_coefficients=optimized.x,ranking_teacher_coefficients=base['coefficients'],loss_type=kind,loss_weight=lam,teacher_tether=gamma,ridge=ridge,training_rows=n,outer_labels_used=False,directional_gradient_check_error=check_error)
    if kind=='soft_cdf':model.update(training_thresholds=thresholds,softness=tau)
    else:model['training_pairs']=len(ip)
    return model
