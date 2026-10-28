"""Collect finished artifacts and atomically install the requested submission."""
from pathlib import Path
import subprocess,json,hashlib,csv,shutil,os
R=Path(__file__).resolve().parent;PROJECT=R.parents[2];REMOTE='/mnt/home/logika-faris-exploration/'+R.name;HOST='faris@labkc-comm1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()
def fetch(name):
 p=R/name;p.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['scp',f'{HOST}:{REMOTE}/{name}',str(p)],check=True);return p
def main():
 state=json.loads(subprocess.check_output(['ssh',HOST,'cat',REMOTE+'/STATE.json'],text=True));assert state['status']=='complete',state;final=fetch('FINAL_STATUS.json');audit=json.loads(final.read_text());assert audit['status']=='passed'and audit['training_rows_per_run']==70000 and audit['runs']==5
 for n in ['ENVIRONMENT.json','PREPARED.json','STATE.json','FULL_DATA_ROLE_AUDIT.json','SDM_SOURCE.json','requirements-resolved.txt']:fetch(n)
 prediction=fetch(audit['submission']);assert sha(prediction)==audit['submission_sha256']
 with prediction.open()as f:
  rows=list(csv.DictReader(f));assert len(rows)==30000 and list(rows[0])==['application_id','default_probability'];assert len({x['application_id']for x in rows})==30000;assert all(0<float(x['default_probability'])<1 for x in rows)
 for fold in range(5):
  meta=json.loads(fetch(f'models/fold{fold}/complete.json').read_text());assert meta['every_training_row_used']and meta['serialized_models_replay_exact']
  for n in ['head.joblib','spline.joblib','maturity_transformer.json','kumo_model.joblib','test_predictions.npz']:
   p=fetch(f'models/fold{fold}/'+n);assert sha(p)==meta['files'][n]
  for n in ['complete.json','prior_audit.json']:fetch(f'feature_build/fold{fold}/'+n)
 target=PROJECT/'submission.csv';before=sha(target)if target.exists()else None
 if target.exists():
  backup=R/('previous_submission_'+before+'.csv')
  if not backup.exists():shutil.copy2(target,backup)
  assert sha(backup)==before
 named=PROJECT/audit['submission'];tmp=named.with_suffix('.csv.tmp');shutil.copyfile(prediction,tmp);assert sha(tmp)==audit['submission_sha256'];tmp.replace(named)
 tmp=target.with_suffix('.csv.tmp');shutil.copyfile(prediction,tmp);assert sha(tmp)==audit['submission_sha256'];tmp.replace(target)
 record=dict(status='passed',output=str(target),named_output=str(named),sha256=sha(target),previous_sha256=before,rows=30000,full_data_training_rows_per_run=70000,uploaded=False);(R/'LOCAL_DELIVERY.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
if __name__=='__main__':main()
