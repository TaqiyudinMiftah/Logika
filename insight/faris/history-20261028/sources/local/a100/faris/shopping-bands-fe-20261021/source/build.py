from pathlib import Path
import time,json
import numpy as np,pandas as pd
import baseline as b
R=Path(__file__).resolve().parents[1];P=R.parent/'shopping-periodicity-fe-20261021'
EXTRA=['beh_spectral_'+channel+'_power_'+period for channel in ['spend','orders']for period in ['7d','14d','30d']]+['beh_spectral_window_supported']
def main():
 assert not(R/'BUILD_COMPLETE.json').exists()and time.time()<1792585871.823748-10800
 assert json.loads((P/'FINAL_STATUS.json').read_text())['status']=='passed';old=json.loads((P/'BUILD_COMPLETE.json').read_text());assert old['default_labels_opened']is False
 receipts={}
 for kind,source in [('bands','true'),('shuffled_bands','shuffled')]:
  path=P/(source+'_features.parquet');assert b.sha(path)==old['feature_hashes'][source];frame=pd.read_parquet(path);assert len(frame)==100000 and frame.application_id.is_unique
  subset=frame[['application_id','customer_id']+EXTRA].copy();subset.to_parquet(R/(kind+'_features.parquet'),index=False);replay=pd.read_parquet(R/(kind+'_features.parquet'));pd.testing.assert_frame_equal(subset,replay,check_exact=True)
  receipts[kind]=dict(source_sha256=b.sha(path),feature_sha256=b.sha(R/(kind+'_features.parquet')),exact_column_subset=True,rows=100000)
 b.dump(dict(status='complete',rows=100000,features=EXTRA,default_labels_opened=False,shared_parameters_fitted=False,source_sha256=b.sha(Path(__file__)),parent_data_audit_sha256=b.sha(P/'BUILD_COMPLETE.json'),parent_model_audit_sha256=b.sha(P/'FINAL_STATUS.json'),feature_hashes={k:v['feature_sha256']for k,v in receipts.items()},receipts=receipts,removed_fields=['beh_spectral_'+c+'_'+n for c in ['spend','orders']for n in ['periodicity_strength','dominant_cycle_days']],reference_errors={c:0. for c in EXTRA}),R/'BUILD_COMPLETE.json')
 print('Exact7fieldsubsets verified; no newFFT/seed/sharedfit.',flush=True)
if __name__=='__main__':main()
