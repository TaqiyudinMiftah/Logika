"""Label-free feature magnitudes and train/test drift, not performance screening."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from scipy.stats import ks_2samp
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'));import baseline as b
def main():
 result=[]
 for f in range(5):
  root=R/f'data/fold{f}';changed=pd.read_parquet(root/'changed_features.parquet');cols=[c for c in changed if c not in ['application_id','customer_id']]
  old=pd.read_parquet(R.parent/f'fold-local-priors-20261021/fold{f}/features.parquet',columns=['application_id']+cols).set_index('application_id').loc[changed.application_id]
  context=pd.read_parquet(root/'context.parquet',columns=['application_id']).application_id;context_mask=changed.application_id.isin(context).to_numpy();rows=[]
  for col in cols:
   before=old[col].to_numpy(float);after=changed[col].to_numpy(float);delta=after[context_mask]-before[context_mask]
   rows.append(dict(column=col,context_rows=int(context_mask.sum()),context_correlation=float(np.corrcoef(before[context_mask],after[context_mask])[0,1]),context_change_quantiles=np.quantile(delta,[0,.1,.5,.9,1]).tolist(),context_fraction_abs_change_gt_005=float(np.mean(abs(delta)>.05)),old_train_test_distribution_KS=float(ks_2samp(before[:70000],before[70000:]).statistic),new_train_test_distribution_KS=float(ks_2samp(after[:70000],after[70000:]).statistic)))
  result.append(dict(fold=f,features=rows))
 b.dump(dict(status='complete',default_labels_opened=False,folds=result,source_sha256=b.sha(Path(__file__)),qualification='Descriptive magnitudes and train/test feature distributions only. No classifier metric, pruning, selection, adaptation or population weighting is performed.'),R/'FEATURE_CONTRASTS.json')
if __name__=='__main__':main()
