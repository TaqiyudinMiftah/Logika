"""Collect the one authorized candidate, validate it, and submit exactly once."""
from pathlib import Path
import csv,datetime,hashlib,json,math,subprocess,sys,time
R=Path(__file__).resolve().parents[1]
HOST='faris@100.68.29.126';REMOTE='/mnt/home/faris/logika-submit-highest-ks-20261029/'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(v,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def api(record,args):
 p=subprocess.run([sys.executable,str(R/'ops/kaggle_private_cli.py'),'--record',record,'--',*args],capture_output=True,text=True,timeout=150)
 print(p.stdout,flush=True)
 assert p.returncode==0,'Official Kaggle call failed; receipt retained for reconciliation, no blind retry'
 return json.loads((R/'kaggle_receipts'/record).read_text())
assert not (R/'kaggle_receipts/SUBMIT_INTENT.json').exists(), 'Never retry uncertain upload without reconciliation'
deadline=time.time()+2100
while time.time()<deadline:
 p=subprocess.run(['rsync','-az','--include=/predictions/***','--include=/*.json','--include=/submission_K*.csv','--exclude=*',HOST+':'+REMOTE,str(R)+'/'],capture_output=True,text=True,timeout=45)
 assert p.returncode==0,'Collection failed; preserve all remote jobs'
 if (R/'QUEUE_FAILURE.json').exists():raise RuntimeError('Remote queue needs execution review; no submission')
 if (R/'SUBMISSION_AUDIT.json').exists():break
 dump(dict(stage='awaiting_audited_CSV',UTC=utc(),completed=len(list((R/'predictions').glob('s*/fold*/RESULT.json')))),R/'private/SUBMIT_STATE.json')
 time.sleep(10)
else:raise TimeoutError('No audited file ready within bounded collection window')
audit=json.loads((R/'SUBMISSION_AUDIT.json').read_text());assert audit['status']=='PASS' and len(audit['members'])==10
assert audit['original_postprocessing_replayed'] and audit['anchor_B_replayed']
filename=audit['submission'];assert Path(filename).name==filename and filename=='submission_KumoLarge99_e2_stack_KS05312029.csv'
file=R/filename;assert sha(file)==audit['sha256']
with (R.parents[2]/'dataset/sample_submission(3).csv').open() as stream:
 reader=csv.DictReader(stream);sample_columns=reader.fieldnames;expected=[row['application_id']for row in reader]
with file.open() as stream:
 reader=csv.DictReader(stream);assert reader.fieldnames==sample_columns;rows=list(reader)
assert len(rows)==len(expected)==30000 and [r['application_id']for r in rows]==expected
assert len(set(expected))==30000
probability=[float(r['default_probability'])for r in rows]
assert all(math.isfinite(p)and 0<p<1 for p in probability) and max(probability)>min(probability)
for member in audit['members']:
 p=R/f"predictions/s{member['seed']}/fold{member['fold']}"
 result=json.loads((p/'RESULT.json').read_text());assert sha(p/'prediction.npz')==result['prediction_sha256']==member['raw_prediction_sha256']
 assert sha(p/'START.json')==result['start_sha256'] and not result['fitted_weights_saved'] and not result['KV_caches_saved']
 assert member['postprocessing_replay_max_abs']<1e-12
intent=dict(UTC=utc(),filename=filename,sha256=audit['sha256'],rows=30000,competition='data-science-competition-logika-ui-2026',
 authorized_by='Human explicitly requested submitting highestKSmodel with savedAPIkey',audit_sha256=sha(R/'SUBMISSION_AUDIT.json'),
 development_KS=.5312029153339166,supported_gain=False,exactly_one_upload=True)
dump(intent,R/'kaggle_receipts/SUBMIT_INTENT.json')
message='Logika fixed highest local KS: KumoLarge99 e2 stack; CV0.531202915; exploratory; sha '+audit['sha256'][:16]
api('SUBMIT_RESPONSE.json',['competitions','submit','-c',intent['competition'],'-f',str(file),'-m',message])
dump(dict(stage='uploaded_awaiting_Kaggle_evaluation',UTC=utc(),sha256=audit['sha256']),R/'private/SUBMIT_STATE.json')
api('POST_UPLOAD_LIST.json',['competitions','submissions','-c',intent['competition'],'--csv'])
