"""Inspect predefined related feature pairs in each50kcontext; no default labels."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import time,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
import baseline as b
R=Path(__file__).resolve().parents[1];A=R.parent;DATA=A/'kumo-temperature-adaptation-20261021/full_data';INPUT=A/'ranking-pipeline-20261020/input'
PAIRS=[
 ('v4_price_all_mean','v4_price_all_equal_order_mean'),
 ('v4_price_180d_mean','v4_price_180d_equal_order_mean'),
 ('v4_price_90d_mean','v4_price_90d_equal_order_mean'),
 ('v5_price_pooled_all_mean','v5_price_pooled_all_equal_order_mean'),
 ('v5_price_pooled_all_equal_category_mean','v5_price_pooled_all_equal_order_mean'),
 ('v4_price_all_equal_category_median','v4_price_all_equal_category_mean'),
 ('v3_repay_all_late_7_posterior_24','v3_repay_all_late_1_posterior_24'),
]
def main():
 assert not(R/'RESULT.json').exists();start=time.time();assert time.time()<1792585871.823748-10800
 ids=pd.read_parquet(INPUT/'data/train_features.parquet',columns=['application_id']);features=list(dict.fromkeys([c for pair in PAIRS for c in pair]));results=[]
 for f in range(5):
  directory=DATA/f'fold{f}';manifest=json.loads((directory/'manifest.json').read_text());assert b.sha(directory/'context.parquet')==manifest['files']['context.parquet']
  frame=pd.read_parquet(directory/'context.parquet',columns=['application_id']+features);ix=np.load(INPUT/f'artifacts/fold{f}/indices.npz')
  assert len(frame)==50000 and frame.application_id.tolist()==ids.application_id.iloc[ix['context']].tolist()
  pairs=[]
  for candidate,keeper in PAIRS:
   x=frame[candidate].to_numpy(float);y=frame[keeper].to_numpy(float);valid=np.isfinite(x)&np.isfinite(y);assert valid.sum()>1000
   scale=max(float(np.std(y[valid])),1e-12);pairs.append(dict(candidate=candidate,keeper=keeper,paired_rows=int(valid.sum()),Pearson=float(np.corrcoef(x[valid],y[valid])[0,1]),Spearman=float(spearmanr(x[valid],y[valid]).statistic),normalized_mean_abs_difference=float(abs(x[valid]-y[valid]).mean()/scale),missing_masks_equal=bool(np.array_equal(np.isfinite(x),np.isfinite(y))),exactly_equal_when_observed=bool(np.array_equal(x[valid],y[valid]))))
  results.append(dict(fold=f,context_ids_sha256=b.ids_hash(ids,ix['context']),pairs=pairs,number_Pearson_ge_098=sum(p['Pearson']>=.98 for p in pairs)))
 b.dump(dict(status='complete',default_labels_opened=False,pairs=PAIRS,folds=results,source_sha256=b.sha(Path(__file__)),seconds=time.time()-start,training_admitted=False,qualification='High correlation is not predictive equivalence or permission to drop columns. Any future selector must be fitted separately inside eachcontext; no global winner or recipe is selected by this audit. Tests no predictive model.'),R/'RESULT.json')
 print(json.dumps(dict(status='complete',fold_counts=[f['number_Pearson_ge_098']for f in results],seconds=time.time()-start)),flush=True)
if __name__=='__main__':main()
