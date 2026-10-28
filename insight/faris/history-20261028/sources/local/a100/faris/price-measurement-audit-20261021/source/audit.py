import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time
import numpy as np,pandas as pd
import baseline as b
R=Path(__file__).resolve().parents[1];A=R.parent;RAW=A/'exploration-12h-20261017/raw';INPUT=A/'ranking-pipeline-20261020/input'
def main():
 assert not(R/'RESULT.json').exists()and time.time()<1792585871.823748-10800;start=time.time()
 original=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id','customer_id']);apps=pd.read_csv(RAW/'train_applications.csv',usecols=['application_id','customer_id','decision_datetime']);apps=apps.set_index('application_id').loc[original.application_id].reset_index();apps.decision_datetime=pd.to_datetime(apps.decision_datetime);ix=np.load(INPUT/'artifacts/fold0/indices.npz');context=apps.iloc[ix['context']];assert len(context)==50000
 raw=RAW/'ecommerce_transactions.csv';ec=pd.read_csv(raw,usecols=['customer_id','order_id','order_item_id','transaction_datetime','product_category','quantity','item_total','order_status'],parse_dates=['transaction_datetime']).merge(context,on='customer_id',validate='many_to_one');assert ec.transaction_datetime.lt(ec.decision_datetime).all()and ec.quantity.gt(0).all()and ec.item_total.gt(0).all()and ec.order_item_id.is_unique
 ec['unit']=ec.item_total/ec.quantity;ec['at5000']=ec.unit.eq(5000);ec['at6000']=ec.unit.eq(6000);ec['at7000']=ec.unit.eq(7000);ec['at8000']=ec.unit.eq(8000);ec['grid1000']=ec.item_total.mod(ec.quantity*1000).eq(0);assert ec.groupby('order_id').customer_id.nunique().max()==1
 groups=[]
 for(category,status),part in ec.groupby(['product_category','order_status']):
  groups.append(dict(category=category,status=status,item_rows=len(part),orders=int(part.order_id.nunique()),customers=int(part.customer_id.nunique()),minimum_unit=float(part.unit.min()),at5000_count=int(part.at5000.sum()),at6000_count=int(part.at6000.sum()),at7000_count=int(part.at7000.sum()),at8000_count=int(part.at8000.sum()),at5000_fraction=float(part.at5000.mean()),grid1000_fraction=float(part.grid1000.mean()),unit_quantiles=part.unit.quantile([0,.01,.1,.5,.9,1]).tolist()))
 customer=ec.groupby('customer_id').agg(items=('unit','size'),floor_items=('at5000','sum'),orders=('order_id','nunique'));customer['floor_share']=customer.floor_items/customer['items'];customer['affected']=customer.floor_items.gt(0)
 affected=ec[ec.at5000];byorder=ec.groupby('order_id').at5000.agg(['sum','count']);allfloor=byorder['sum'].eq(byorder['count'])
 summary=dict(item_rows=len(ec),orders=int(ec.order_id.nunique()),customers=int(ec.customer_id.nunique()),minimum_unit=float(ec.unit.min()),global_at5000_count=int(ec.at5000.sum()),global_at5000_fraction=float(ec.at5000.mean()),customers_with5000=int(customer.affected.sum()),fraction_customers_with5000=float(customer.affected.mean()),floor_share_affected_quantiles=customer.loc[customer.affected,'floor_share'].quantile([0,.1,.5,.9,1]).tolist(),orders_all_items5000=int(allfloor.sum()),customers_with_multiple_floor_orders=int(affected.groupby('customer_id').order_id.nunique().ge(2).sum()),grid1000_fraction=float(ec.grid1000.mean()))
 # Independent raw mask/group count consistency; no default labels are available.
 assert int(sum(g['at5000_count']for g in groups))==summary['global_at5000_count']==int(customer.floor_items.sum())
 customer.reset_index().to_parquet(R/'customer_floor_counts.parquet',index=False);pd.DataFrame(groups).to_csv(R/'category_status_counts.csv',index=False)
 b.dump(dict(status='complete',context_rows=50000,fit_ids_sha256=b.ids_hash(original,ix['context']),default_labels_opened=False,outer_or_correction_customers_used=False,summary=summary,category_status=groups,source_raw_sha256=b.sha(raw),source_sha256=b.sha(Path(__file__)),seconds=time.time()-start,credit_model_training_admitted=False,qualification='Loweredgepileupisdescriptive; nooriginalgenerator/censorrule/latentdistributionisproven. Cancelledpricesarenotspending.'),R/'RESULT.json');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
