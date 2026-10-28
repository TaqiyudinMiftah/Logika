from pathlib import Path
import json,time,hashlib
import numpy as np,pandas as pd
from scipy.stats import norm
import measurement as m
R=Path(__file__).resolve().parent;A=R.parent;RAW=A/'exploration-12h-20261017/raw';INPUT=A/'ranking-pipeline-20261020/input'
def main():
 start=time.time();assert not(R/'SCREEN_RESULT.json').exists();assert json.loads((R/'OCCASION_RESULT.json').read_text())['all_three_shared_change_supported'];m.verify_identity(None)
 ids=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id']);ix=np.load(INPUT/'artifacts/fold0/indices.npz');a=pd.read_csv(RAW/'train_applications.csv',usecols=['application_id','customer_id','decision_datetime'],parse_dates=['decision_datetime']).set_index('application_id').loc[ids.application_id.iloc[ix['context']]].reset_index();assert a.customer_id.is_unique
 d=pd.read_csv(RAW/'bureau_observations.csv',parse_dates=['report_datetime','available_datetime']);d=d[d.customer_id.isin(a.customer_id)&d.score.notna()].copy();cut=a.set_index('customer_id').decision_datetime;assert(d.report_datetime<d.customer_id.map(cut)).all()and(d.available_datetime<d.customer_id.map(cut)).all();d['day']=d.report_datetime.astype('int64')/86400e9
 perm=np.random.default_rng(20261202).permutation(len(a));fit_ids=set(a.customer_id.iloc[perm[:25000]]);test=a.iloc[perm[25000:]].copy();train=d[d.customer_id.isin(fit_ids)].copy();validation=d[d.customer_id.isin(test.customer_id)].copy();assert set(train.customer_id).isdisjoint(validation.customer_id);par={name:m.fit(train,flag)for name,flag in [('static',False),('occasion',True)]};selected=test.set_index('customer_id').application_id;rows=[];effects=[]
 for provider in sorted(d.bureau_provider.unique()):
  hidden=validation[(validation.bureau_provider==provider)&(validation.application_id==validation.customer_id.map(selected))].copy();assert len(hidden)>1000 and hidden.customer_id.is_unique;features=validation.drop(index=hidden.index);assert not set(features.index)&set(hidden.index);apps=test.set_index('customer_id').loc[hidden.customer_id].reset_index();out={}
  for name,p in par.items():
   mean,sd=m.transform(features,apps,p);prediction=mean+p['offset'][provider];out[name]=(hidden.score.to_numpy(float)-prediction)**2
  diff=out['occasion']-out['static'];se=float(diff.std(ddof=1)/np.sqrt(len(diff)));critical=float(norm.ppf(1-.05/6));rows.append(dict(provider=str(provider),responses=len(diff),MSE_static=float(out['static'].mean()),MSE_occasion=float(out['occasion'].mean()),delta=float(diff.mean()),secondary_CI=[float(diff.mean()-critical*se),float(diff.mean()+critical*se)]));effects.append(pd.DataFrame(dict(customer_id=hidden.customer_id.to_numpy(),delta=diff)))
 effects=pd.concat(effects).groupby('customer_id').delta.mean();mean=float(effects.mean());se=float(effects.std(ddof=1)/np.sqrt(len(effects)));ci=[mean-1.96*se,mean+1.96*se];supported=ci[1]<0 and all(r['delta']<0 for r in rows)
 result=dict(status='complete',primary_customers=len(effects),primary_delta_MSE=mean,primary_CI=ci,provider_results=rows,parameters=par,measurement_screen_passed=supported,classification_fit_admitted=False,default_labels_opened=False,fit_customers=25000,measurement_validation_customers=25000,heldout_score_absent_from_own_inputs=True,exact_Gaussian_identity_and_zero_limit_passed=True,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in[R/'measurement.py',Path(__file__),R/'MEASUREMENT_SCREEN.md']},seconds=time.time()-start,qualification='Holdoutscorepredictiononly; notdefaultAUC/KS. Conditionalnormalapproximation; no guaranteeofdownstreamgain.');(R/'SCREEN_RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
