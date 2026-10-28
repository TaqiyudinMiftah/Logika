"""Label-free observed-history support audit; no prediction or feature selection."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time
import numpy as np,pandas as pd
import baseline as b
R=Path(__file__).resolve().parents[1];A=R.parent;RAW=A/'exploration-12h-20261017/raw';DATA=A/'kumo-temperature-adaptation-20261021/full_data';INPUT=A/'ranking-pipeline-20261020/input'
def main():
 assert not(R/'HISTORY_SUPPORT.json').exists();start=time.time()
 apps=pd.concat([pd.read_csv(RAW/(s+'_applications.csv'),usecols=['application_id','customer_id','decision_datetime'])for s in ['train','test']],ignore_index=True);apps.decision_datetime=pd.to_datetime(apps.decision_datetime);assert apps.customer_id.is_unique
 ac=pd.read_csv(RAW/'account_behavior.csv',usecols=['customer_id','month'],parse_dates=['month']).merge(apps,on='customer_id',validate='many_to_one');assert (ac.month+pd.offsets.MonthBegin(1)<=ac.decision_datetime).all()and not ac.duplicated(['customer_id','month']).any();ac['serial']=ac.month.dt.year*12+ac.month.dt.month;ag=ac.groupby('customer_id').serial.agg(['count','min','max']);ag['span']=ag['max']-ag['min']+1;ag['gaps']=ag['span']-ag['count']
 result=dict(account_observed_customers=len(ag),account_complete_month_count_quantiles=ag['count'].quantile([0,.1,.5,.9,1]).tolist(),account_calendar_gap_fraction=float(ag.gaps.gt(0).mean()),account_gap_count_distribution={str(k):int(v)for k,v in ag.gaps.value_counts().items()})
 bu=pd.read_csv(RAW/'bureau_observations.csv',usecols=['customer_id','bureau_provider','report_datetime','available_datetime','score'],parse_dates=['report_datetime','available_datetime']).merge(apps,on='customer_id',validate='many_to_one');assert (bu.available_datetime<bu.decision_datetime).all()and(bu.report_datetime<bu.decision_datetime).all();providers=bu.groupby('customer_id').bureau_provider.nunique();reports=bu.groupby('customer_id').size();scored=bu[bu.score.notna()].groupby('customer_id').size();result.update(bureau_provider_count_distribution={str(k):int(v)for k,v in providers.value_counts().items()},bureau_report_count_quantiles=reports.quantile([0,.1,.5,.9,1]).tolist(),bureau_scored_report_count_quantiles=scored.quantile([0,.1,.5,.9,1]).tolist())
 original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','customer_id']);checks=[]
 for f in range(5):
  ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz');ctx=original.iloc[ix['context']];x=pd.read_parquet(DATA/f'fold{f}/context.parquet',columns=['application_id','has_account','aux_account_months']);assert x.application_id.tolist()==ctx.application_id.tolist();full=ag['count'].reindex(ctx.customer_id,fill_value=0).to_numpy();recent=x.aux_account_months.to_numpy();flag=x.has_account.to_numpy()
  checks.append(dict(fold=f,context_only=True,positive_history_length_unique=int(np.unique(full[full>0]).size),recent_count_exactly_source_flag_times_three=bool(np.array_equal(recent,3*flag)),full_count_differs_from_recent=bool(np.any(full!=recent)),full_account_count_quantiles=np.quantile(full,[0,.1,.5,.9,1]).tolist(),context_ids_sha256=b.ids_hash(original,ix['context'])))
 b.dump(dict(status='complete',rows=100000,default_labels_opened=False,shared_prior_fitted=False,classification_training_admitted=False,observed_support=result,context_checks=checks,source_hashes={name:b.sha(RAW/name)for name in ['account_behavior.csv','bureau_observations.csv']},source_sha256=b.sha(Path(__file__)),seconds=time.time()-start,qualification='First/last observed history is not certified source-enrollment duration. Fullcounts are not assumed missing-at-random or causal risk. Existing nonselected feature-bank columns and teammate lifecycle catalog already contain related support measures. This is not proof of predictive novelty or a new experiment admission.'),R/'HISTORY_SUPPORT.json')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
