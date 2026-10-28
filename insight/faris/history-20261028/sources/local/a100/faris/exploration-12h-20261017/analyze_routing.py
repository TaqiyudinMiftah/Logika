"""Fixed routing diagnostics from EXISTING development predictions; no training."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,sys
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp
root=Path(sys.argv[1]);out=Path(sys.argv[2]);sys.path[:0]=[str(root),str(root/'source')]
from supervisor import records,read
bank=Path(read(root/'deployment.json')['hosts']['pc2']['bank']);y=np.load(bank/'development_y.npy');dev=np.load(bank/'development_rows.npy');fold=np.load(bank/'folds.npy');m=read(bank/'manifest.json');x=np.load(bank/'numeric.npy',mmap_mode='r');lookup={n:i for i,n in enumerate(m['columns'])}
no_repay=x[dev,lookup['has_repayments']]==0;no_account=x[dev,lookup['has_account']]==0
cohorts={'no_repayment':no_repay,'has_repayment':~no_repay,'no_account':no_account,'has_account':~no_account}
for r in [False,True]:
 for a in [False,True]:cohorts[f'repayment_missing_{r}_account_missing_{a}']=(no_repay==r)&(no_account==a)
ks=lambda t,p:float(ks_2samp(p[t==0],p[t==1]).statistic)
results=[];group_results=[]
for protocol in ['screen','reseed','repeat']:
 b=np.load(root/'evidence/champion_oof.npz')['existing_feature_ensemble'] if protocol=='screen' else np.load(root/'reports'/('champion_'+protocol+'.npy'))
 p=np.load(root/'reports'/('ens_e62cfb5b6d636145'+('' if protocol=='screen' else '_'+protocol)+'.npy'))
 ff=fold if protocol!='repeat' else np.asarray(read(root/'evidence/repeat.json')['folds'])
 tests={'new_if_no_repayment':np.where(no_repay,p,b),'new_if_no_account':np.where(no_account,p,b),
        'new_if_either_missing':np.where(no_repay|no_account,p,b)}
 for name,condition in [('repayment',no_repay),('account',no_account)]:
  w=np.where(condition,.75,.25);tests['soft_probability_'+name]=(1-w)*b+w*p
  bc=np.clip(b,1e-8,1-1e-8);pc=np.clip(p,1e-8,1-1e-8);tests['soft_logit_'+name]=1/(1+np.exp(-((1-w)*np.log(bc/(1-bc))+w*np.log(pc/(1-pc)))))
 for name,q in tests.items():results.append({'protocol':protocol,'rule':name,'ks':ks(y,q),'delta_vs_champion':ks(y,q)-ks(y,b),'delta_vs_new':ks(y,q)-ks(y,p),'fold_delta_vs_champion':[ks(y[ff==f],q[ff==f])-ks(y[ff==f],b[ff==f]) for f in range(3)],'posthoc_not_confirmed':True})
 for name,mask in cohorts.items():
  if min(int(y[mask].sum()),int((1-y[mask]).sum()))<30:continue
  group_results.append({'protocol':protocol,'cohort':name,'n':int(mask.sum()),'defaults':int(y[mask].sum()),'new_ks':ks(y[mask],p[mask]),'old_ks':ks(y[mask],b[mask]),'delta':ks(y[mask],p[mask])-ks(y[mask],b[mask]),'fold_delta':[ks(y[mask&(ff==f)],p[mask&(ff==f)])-ks(y[mask&(ff==f)],b[mask&(ff==f)]) for f in range(3)]})
(out/'fixed_routing_diagnostics.json').write_text(json.dumps(results,indent=2)+'\n');(out/'cohort_stability.json').write_text(json.dumps(group_results,indent=2)+'\n')
lineage=[]
rr=records()
for key in ['pc2_job_000623','pc3_job_000424','pc3_job_000846']:
 r=next(r for r in rr if r['id']==key)
 for fold in range(3):
  a=read(Path(r['folder'])/f'fold_{fold}/preprocessing.json');names=a['selected_features'];lineage.append({'id':key,'fold':fold,'selected':len(names),'output_features':a['output_features'],'new_selected':[n for n in names if n.startswith('new_')]})
(out/'finalist_feature_lineage.json').write_text(json.dumps(lineage,indent=2)+'\n')
print(json.dumps({'routing':results,'cohort_stability':group_results,'feature_lineage':lineage}))
