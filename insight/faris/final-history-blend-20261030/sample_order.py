"""Export held candidates in exact official sample order without changing any ID/prediction pair."""
from pathlib import Path
import pandas as pd,numpy as np,hashlib,json,time
R=Path(__file__).resolve().parent;sample=Path('/mnt/home/faris/logika-faris-exploration/exploration-12h-20261017/raw/sample_submission(3).csv');S=pd.read_csv(sample);ids=S.application_id.astype(str);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();out=R/'exports';out.mkdir(exist_ok=True)
items=[(R/'paired/candidates/current_fusion_cann_kumo70_p0.003_w0.1/submission_HELD.csv','submission_localKS_0.5319679799870825.csv')]+[(p,p.name)for p in (R/'private/test-blends-qualified').glob('submission_UNSCORED*.csv')];records=[]
for p,n in items:
 d=pd.read_csv(p);assert len(d)==30000 and d.application_id.nunique()==30000;old=pd.Series(d.default_probability.to_numpy(),index=d.application_id.astype(str));assert set(old.index)==set(ids);values=old.reindex(ids).to_numpy();assert np.isfinite(values).all() and np.all((values>=0)&(values<=1))
 q=out/n;pd.DataFrame(dict(application_id=S.application_id,default_probability=values)).to_csv(q,index=False,float_format='%.17g');rb=pd.read_csv(q,float_precision='round_trip');assert rb.application_id.astype(str).tolist()==ids.tolist();assert np.max(abs(rb.default_probability.to_numpy()-values))==0
 restored=pd.Series(rb.default_probability.to_numpy(),index=rb.application_id.astype(str)).reindex(old.index).to_numpy();assert np.array_equal(restored,old.to_numpy());records.append(dict(path=str(q),sha256=sha(q),source_path=str(p),source_sha256=sha(p),exact_sample_row_order=True,all30000_ID_prediction_pairs_unchanged=True,max_abs_export_error=0.,submitted=False))
j=dict(status='PASS',epoch=time.time(),sample_sha256=sha(sample),source_sha256=sha(__file__),source_artifacts_unchanged=True,scored_local_predictions_unchanged=True,records=records);(out/'EXPORT_AUDIT.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j),flush=True)
