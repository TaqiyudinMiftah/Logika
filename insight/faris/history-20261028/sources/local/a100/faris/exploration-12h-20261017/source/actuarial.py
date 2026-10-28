"""12-month default-frequency models; no invented severity or survival labels.

Binomial links and beta-binomial credibility are modelling hypotheses, not
regulatory credit scores. Supervised scorecard maps are inner-cross-fitted.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class BinomialGLM(ClassifierMixin,BaseEstimator):
 def __init__(self,link='logit',strength=.05,weight=1,max_features=128,seed=2026):
  self.link=link;self.strength=strength;self.weight=weight;self.max_features=max_features;self.seed=seed
 def fit(self,x,y):
  from sklearn.feature_selection import f_classif
  import warnings
  x=np.nan_to_num(np.asarray(x,dtype=float));y=np.asarray(y)
  with warnings.catch_warnings():
   warnings.simplefilter('ignore');scores=f_classif(x,y)[0]
  self.columns_=np.argsort(np.nan_to_num(scores,nan=-1,posinf=1e30),kind='stable')[-min(self.max_features,x.shape[1]):]
  self.scale_=StandardScaler();z=np.clip(self.scale_.fit_transform(x[:,self.columns_]),-8,8);z=np.column_stack([np.ones(len(z)),z]);w=np.where(y==1,self.weight,1.);den=w.sum();penalty=1/max(self.strength*len(y),1.)
  def objective(beta):
   eta=z@beta
   if self.link=='logit':loss=np.logaddexp(0,eta)-y*eta;gradient=expit(eta)-y
   elif self.link=='cloglog':
    e=np.clip(eta,-25,7);t=np.exp(e);loss=(1-y)*t-y*np.log(-np.expm1(-t));g=np.zeros(len(t));small=t<50;g[small]=t[small]/np.expm1(t[small]);gradient=((1-y)*t-y*g)*((eta>=-25)&(eta<=7))
   else:raise ValueError(self.link)
   reg=beta.copy();reg[0]=0
   return float(w@loss/den+.5*penalty*(reg@reg)),z.T@(w*gradient)/den+penalty*reg
  beta=np.zeros(z.shape[1]);rate=np.clip(np.average(y,weights=w),1e-6,1-1e-6);beta[0]=np.log(rate/(1-rate)) if self.link=='logit' else np.log(-np.log1p(-rate))
  self.optimization_=minimize(objective,beta,jac=True,method='L-BFGS-B',options={'maxiter':400,'ftol':1e-9});self.coef_=self.optimization_.x;self.classes_=np.array([0,1]);return self
 def predict_proba(self,x):
  x=np.nan_to_num(np.asarray(x,dtype=float));z=np.clip(self.scale_.transform(x[:,self.columns_]),-8,8);eta=self.coef_[0]+z@self.coef_[1:];p=expit(eta) if self.link=='logit' else -np.expm1(-np.exp(np.clip(eta,-25,7)));p=np.clip(p,1e-8,1-1e-8);return np.column_stack([1-p,p])


class CredibilityScorecard(ClassifierMixin,BaseEstimator):
 def __init__(self,bins=5,strength=24,weight=1,max_features=64,credibility=False,C=.05,seed=2026):
  self.bins=bins;self.strength=strength;self.weight=weight;self.max_features=max_features;self.credibility=credibility;self.C=C;self.seed=seed
 def _maps(self,x,y):
  prior=float((y.sum()+.5)/(len(y)+1));maps=[]
  for j in range(x.shape[1]):
   a=x[:,j];finite=np.isfinite(a);edges=np.unique(np.quantile(a[finite],np.linspace(0,1,self.bins+1)[1:-1])) if finite.any() else np.array([]);b=np.where(finite,np.searchsorted(edges,a,side='right')+1,0);n=np.bincount(b,minlength=len(edges)+2);k=np.bincount(b,weights=y,minlength=len(n));s=max(self.strength,1e-3);post=(k+s*prior)/(n+s);woe=np.log(np.clip(post,1e-6,1-1e-6)/np.clip(1-post,1e-6,1)) - np.log(prior/(1-prior));unc=np.sqrt(post*(1-post)/(n+s+1));maps.append((edges,woe,unc,n/(n+s)))
  return maps
 def _apply(self,x,maps):
  values=[]
  for j,(edges,woe,unc,z) in enumerate(maps):
   a=x[:,j];b=np.where(np.isfinite(a),np.searchsorted(edges,a,side='right')+1,0);values.append(woe[b])
   if self.credibility:values.extend([unc[b],z[b]])
  return np.column_stack(values)
 def fit(self,x,y):
  x=np.asarray(x,dtype=float);y=np.asarray(y)
  # The supervised univariate screen sees outer-training labels only.
  # Bin/target maps for a training row exclude that row's inner fold.
  from sklearn.feature_selection import f_classif
  import warnings
  with warnings.catch_warnings():
   warnings.simplefilter('ignore');scores=f_classif(np.nan_to_num(x),y)[0]
  self.columns_=np.argsort(np.nan_to_num(scores,nan=-1,posinf=1e30),kind='stable')[-min(self.max_features,x.shape[1]):];x=x[:,self.columns_]
  dimensions=x.shape[1]*(3 if self.credibility else 1);oof=np.empty((len(x),dimensions));self.crossfit_=np.empty(len(x),np.int8)
  for f,(ti,vi) in enumerate(StratifiedKFold(3,shuffle=True,random_state=self.seed).split(x,y)):
   oof[vi]=self._apply(x[vi],self._maps(x[ti],y[ti]));self.crossfit_[vi]=f
  self.maps_=self._maps(x,y);self.scale_=StandardScaler();z=self.scale_.fit_transform(oof);self.model_=LogisticRegression(C=self.C,max_iter=800,class_weight={0:1,1:self.weight},random_state=self.seed).fit(z,y);self.classes_=np.array([0,1]);return self
 def predict_proba(self,x):return self.model_.predict_proba(self.scale_.transform(self._apply(np.asarray(x)[:,self.columns_],self.maps_)))


def reliability(y,p):
 """Descriptive out-of-fold probability diagnostics, never tuning on confirmation."""
 from sklearn.metrics import brier_score_loss
 y=np.asarray(y);p=np.asarray(p);edges=np.linspace(0,1,11);bins=np.minimum(np.searchsorted(edges,p,side='right')-1,9);table=[]
 for b in range(10):
  ix=bins==b
  if ix.any():table.append({'bin':b,'n':int(ix.sum()),'predicted':float(p[ix].mean()),'observed':float(y[ix].mean())})
 return {'brier':float(brier_score_loss(y,p)),'observed_default_rate':float(y.mean()),'predicted_default_rate':float(p.mean()),'observed_to_predicted':float(y.sum()/max(p.sum(),1e-12)),'reliability':table,'outcome':'12-month 90-DPD frequency; no severity target or realized-loss claim'}
