from pathlib import Path
import json,joblib
import numpy as np,pandas as pd
import baseline as b
R=Path(__file__).resolve().parents[1]
def main():
 assert not(R/'FINAL_STATUS.json').exists();sample=pd.read_csv(R/'sample_submission.csv');combined=np.zeros(30000);checks=[]
 for f in range(5):
  root=R/f'models/fold{f}';meta=json.loads((root/'complete.json').read_text());assert meta['status']=='passed'and meta['every_training_row_used']and meta['spline_rows']==70000
  for name,h in meta['files'].items():assert b.sha(root/name)==h
  head=joblib.load(root/'head.joblib');sp=joblib.load(root/'spline.joblib');frame=pd.read_parquet(root/'test_features.parquet');assert'target'not in frame;raw=np.load(root/'raw_predictions.npz');pred=.75*b.corrected_probability(head,frame,raw['prediction'][6000:])[0]+.25*b.predict_spline(sp,frame);saved=np.load(root/'test_predictions.npz');assert np.array_equal(pred,saved['prediction'])and saved['application_id'].tolist()==frame.application_id.tolist();aligned=pd.Series(pred,index=frame.application_id).reindex(sample.application_id).to_numpy();assert np.isfinite(aligned).all();combined+=aligned/5;checks.append(dict(fold=f,models_replayed=True,seconds=meta['seconds']))
 assert np.isfinite(combined).all()and((combined>0)&(combined<1)).all()and len(np.unique(combined))>100;out=sample[['application_id']].copy();out['default_probability']=combined;dest=R/'submission_ks05309665_periodicity_maturity_full.csv';out.to_csv(dest,index=False,float_format='%.17g');back=pd.read_csv(dest,float_precision='round_trip');assert back.application_id.tolist()==sample.application_id.tolist()and np.array_equal(back.default_probability.to_numpy(),combined);b.dump(dict(status='passed',submission=dest.name,submission_sha256=b.sha(dest),rows=30000,columns=out.columns.tolist(),probability_min=float(combined.min()),probability_max=float(combined.max()),checks=checks,training_rows_per_run=70000,Kumo_context_per_run=64000,correction_per_run=6000,spline_per_run=70000,runs=5,selected_recipe_research_OOF_KS=.5309665314756384,selected_recipe_research_OOF_AUC=.8469295983174442,final_refit_validation_score=None,public_score=None,uploaded=False,qualification='UserselectedhighestKSdespiteAUCloss. Full-datarefitdoesnotinheritaguaranteedCVorpublicscore.'),R/'FINAL_STATUS.json');print(json.dumps(dict(status='passed',submission=str(dest),sha256=b.sha(dest))),flush=True)
if __name__=='__main__':main()
