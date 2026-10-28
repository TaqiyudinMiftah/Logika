import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import numpy as np,pandas as pd
from utils import write,read
from prepare_features import build
from features import ExploreBank,Processor

class FeaturesTests(unittest.TestCase):
 def fixture(self,r):
  raw=r/'raw';raw.mkdir();bank=r/'bank';bank.mkdir();ids=['c'+str(i) for i in range(12)];apps=pd.DataFrame({'application_id':['a'+str(i) for i in range(12)],'customer_id':ids,'decision_datetime':['2026-01-15']*12,'loan_amount':100.,'proposed_monthly_installment':5.})
  apps.iloc[:8].assign(target=[0,1]*4).to_csv(raw/'train_applications.csv',index=False);apps.iloc[8:].to_csv(raw/'test_applications.csv',index=False);pd.DataFrame({'customer_id':ids,'declared_monthly_income':20.,'employment_type':'test'}).to_csv(raw/'customer_profiles.csv',index=False)
  pd.DataFrame({'customer_id':ids*2,'loan_id':['l'+str(i) for i in range(24)],'due_date':['2025-11-01']*12+['2025-12-01']*12,'amount_due':10.,'amount_paid':[5.,10.]*12,'max_days_past_due':[1,40]*12}).to_csv(raw/'repayment_history.csv',index=False)
  pd.DataFrame({'customer_id':ids*2,'bureau_provider':['a']*12+['b']*12,'available_datetime':'2025-12-10','report_datetime':'2025-12-01','score':[500.,600.]*12,'utilization':.5,'outstanding_balance':10.}).to_csv(raw/'bureau_observations.csv',index=False)
  pd.DataFrame({'customer_id':ids*2,'month':['2025-11-01']*12+['2025-12-01']*12,'income_like_credits':20.,'outflows':12.,'average_balance':[5.,10.]*12}).to_csv(raw/'account_behavior.csv',index=False)
  pd.DataFrame({'customer_id':ids*2,'transaction_datetime':'2025-12-01','order_id':['o'+str(i) for i in range(24)],'quantity':2,'item_total':4.,'order_total':4.,'product_category':['food','fashion']*12,'payment_method':'cash','order_status':['completed']*12+['cancelled']*12}).to_csv(raw/'ecommerce_transactions.csv',index=False)
  apps.assign(split=['train']*8+['test']*4).to_parquet(bank/'metadata.parquet',index=False);pd.DataFrame({'employment_type':['test']*12}).to_parquet(bank/'categorical.parquet',index=False)
  np.save(bank/'numeric.npy',np.arange(12,dtype='float32')[:,None]);np.save(bank/'development_rows.npy',np.arange(8));np.save(bank/'development_y.npy',np.array([0,1]*4));np.save(bank/'folds.npy',np.arange(8)%3)
  write(bank/'manifest.json',{'bank_id':'synthetic','columns':['loan_amount'],'categorical_columns':['employment_type'],'priors':[],'info':{'loan_amount':{'sources':['application']}},'views':{'v3':['loan_amount','employment_type']}})
  return raw,bank
 def test_point_in_time_and_no_label_features(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Path(tmp);raw,bank=self.fixture(r);build(raw,bank,r/'extras');m=read(r/'extras/manifest.json');assert 'target' not in m['columns'];assert 'application_id' not in m['columns'];seq=np.load(r/'extras/events.npy');assert seq.shape==(12,128,9);assert np.isfinite(seq).all()
   first=np.load(r/'extras/numeric.npy').copy();train=pd.read_csv(raw/'train_applications.csv');train.target=1-train.target;train.to_csv(raw/'train_applications.csv',index=False);build(raw,bank,r/'extras2');np.testing.assert_allclose(first,np.load(r/'extras2/numeric.npy'),equal_nan=True)
 def test_future_observation_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Path(tmp);raw,bank=self.fixture(r);d=pd.read_csv(raw/'bureau_observations.csv');d.available_datetime='2026-02-01';d.to_csv(raw/'bureau_observations.csv',index=False)
   with self.assertRaises(AssertionError):build(raw,bank,r/'extras')

if __name__=='__main__':unittest.main()

class TransformTests(FeaturesTests):
 def test_fold_local_new_transforms(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Path(tmp);raw,bank=self.fixture(r);build(raw,bank,r/'extras');b=ExploreBank(r,r/'extras')
   views=read(Path(__file__).resolve().parents[1]/'source/views.json');rep=views['repayment_05'];pre=Processor(b,rep,'elastic',2026);x=pre.fit_transform(np.arange(8),b.y);q=pre.transform(np.arange(8,12));assert x.shape[1]==q.shape[1] and np.isfinite(x).all() and np.isfinite(q).all();assert pre.audit['actual_selection']=='true_gain';assert pre.pre.fit_rows.tolist()==list(range(8))
