"""Fold-local Gaussian score measurements with a shared application component.

No default outcomes are accepted. Application IDs group reports, never predict.
"""
import numpy as np,pandas as pd,itertools
def occasion_variance(d):
 s=d.pivot(index=['customer_id','application_id'],columns='bureau_provider',values='score');t=d.pivot(index=['customer_id','application_id'],columns='bureau_provider',values='day');parts=[]
 for p,q in itertools.combinations(sorted(s.columns),2):
  e=s[[p,q]].dropna().copy();e['day']=t.loc[e.index,[p,q]].mean(axis=1);e=e.reset_index().sort_values(['customer_id','day','application_id'],kind='stable');g=e.groupby('customer_id');first=g.head(1).set_index('customer_id');last=g.tail(1).set_index('customer_id').reindex(first.index);ok=(last.day-first.day)>=90;x=(last.loc[ok,p]-first.loc[ok,p]).to_numpy(float);y=(last.loc[ok,q]-first.loc[ok,q]).to_numpy(float);assert len(x)>1000;parts.append((len(x),float(np.mean((x-x.mean())*(y-y.mean())))/2))
 return max(0.,sum(n*v for n,v in parts)/sum(n for n,v in parts)),parts
def summaries(d,par):
 z=d[['customer_id','application_id']].copy();w=1/d.bureau_provider.map(par['noise']).to_numpy(float);z['precision']=w;z['weighted']=(d.score-d.bureau_provider.map(par['offset'])).to_numpy(float)*w
 g=z.groupby(['customer_id','application_id'])[['precision','weighted']].sum();o=par['occasion'];g['effective_precision']=g.precision/(1+o*g.precision);g['effective_weighted']=g.weighted/(1+o*g.precision)
 c=g.groupby(level=0)[['effective_precision','effective_weighted']].sum();return g,c
def fit(d,shared_occasion):
 assert 'target'not in d and not d.duplicated(['customer_id','application_id','bureau_provider']).any();s=d.pivot(index=['customer_id','application_id'],columns='bureau_provider',values='score');names=sorted(s.columns);assert len(names)==3;var={};mean={}
 for a,b in itertools.combinations(names,2):
  diff=(s[a]-s[b]).dropna();var[a,b]=float(diff.var(ddof=0));mean[a,b]=float(diff.mean())
 a,b,c=names;offset=np.array([0.,-mean[a,b],-mean[a,c]]);offset-=offset.mean();noise=np.maximum([(var[a,b]+var[a,c]-var[b,c])/2,(var[a,b]+var[b,c]-var[a,c])/2,(var[a,c]+var[b,c]-var[a,b])/2],1e-8);o,parts=occasion_variance(d)if shared_occasion else(0.,[])
 par=dict(offset=dict(zip(names,offset.tolist())),noise=dict(zip(names,noise.tolist())),occasion=o,occasion_moments=parts);_,customers=summaries(d,par);latent=customers.effective_weighted/customers.effective_precision;par.update(prior_mean=float(latent.mean()),prior_variance=max(float(latent.var(ddof=0)-(1/customers.effective_precision).mean()),1e-8),customer_count=len(customers),default_labels_used=False)
 return par
def transform(d,apps,par):
 assert'target'not in d and'target'not in apps and apps.customer_id.is_unique;g,c=summaries(d,par);c=c.reindex(apps.customer_id).fillna(0);var_mu=1/(1/par['prior_variance']+c.effective_precision.to_numpy());mean_mu=var_mu*(par['prior_mean']/par['prior_variance']+c.effective_weighted.to_numpy());index=pd.MultiIndex.from_frame(apps[['customer_id','application_id']]);cur=g.reindex(index).fillna(0);lam=cur.precision.to_numpy();weighted=cur.weighted.to_numpy();o=par['occasion'];den=1+o*lam
 mean=mean_mu/den+o*weighted/den;var=o/den+var_mu/(den*den);assert np.isfinite(mean).all()and(var>0).all()
 return mean,np.sqrt(var)
def verify_identity(par):
 # Exact joint Gaussian conditioning with two occasions, three measurements.
 x=np.array([610.,650.,625.]);noise=np.array([200.,400.,300.]);occasion=np.array([0,0,1]);mu=630.;tau=900.;omega=17.;v=tau+omega*(occasion[:,None]==occasion[None,:])+np.diag(noise);cross=tau+omega*(occasion==1);direct=mu+cross@np.linalg.solve(v,x-mu);variance=tau+omega-cross@np.linalg.solve(v,cross)
 labels=['A','B','C'];frame=pd.DataFrame(dict(customer_id=['c']*3,application_id=['old','old','current'],bureau_provider=labels,score=x));p=dict(offset=dict.fromkeys(labels,0.),noise=dict(zip(labels,noise)),occasion=omega,prior_mean=mu,prior_variance=tau);m,s=transform(frame,pd.DataFrame(dict(customer_id=['c'],application_id=['current'])),p);assert abs(m[0]-direct)<1e-10 and abs(s[0]**2-variance)<1e-10
 p['occasion']=0.;m,s=transform(frame,pd.DataFrame(dict(customer_id=['c'],application_id=['current'])),p);precision=1/tau+sum(1/noise);assert abs(m[0]-(mu/tau+sum(x/noise))/precision)<1e-10 and abs(s[0]**2-1/precision)<1e-10
