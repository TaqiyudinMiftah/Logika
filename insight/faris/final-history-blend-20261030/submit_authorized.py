"""One explicitly authorized submission of the audited final paired CSV."""
from pathlib import Path
import csv,datetime,fcntl,hashlib,io,json,math,os,subprocess,time
R=Path('/home/faris/logika-final-history-blend-20261030-submission');R.mkdir(exist_ok=True);P=R/'private';P.mkdir(exist_ok=True,mode=0o700);OLD=Path('/home/faris/logika-overnight-refit-20261030');COMP='data-science-competition-logika-ui-2026';SHA='f14cb4555575016828f656d722aa9cb9f9d85217bb3518ba18eddd59c7b10f62';MSG='LogikaFinalPaired_KS05319680_'+SHA[:12]
lock=(OLD/'private/submit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
tokenfile=OLD/'private/kaggle-token';assert tokenfile.stat().st_mode&0o077==0;token=tokenfile.read_text().strip()
def call(args,label):
 try:
  q=subprocess.run([str(OLD/'kaggle-env/bin/kaggle'),*args],capture_output=True,text=True,env={**os.environ,'KAGGLE_API_TOKEN':token},timeout=60);v=dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr)
 except subprocess.TimeoutExpired:v=dict(returncode=None,stdout='',stderr='Timed out; reconcile server before any retry')
 v={k:(x.replace(token,'[REDACTED]')if isinstance(x,str)else x)for k,x in v.items()};dump(P/(label+'.json'),v);return v
file=R/'submission_localKS_0.5319679799870825.csv';assert sha(file)==SHA
export=json.loads((R/'EXPORT_AUDIT.json').read_text());audit=json.loads((R/'FINAL_AUDIT.json').read_text());assert export['status']==audit['status']=='PASS' and audit['selected']['KS']==.5319679799870825;assert any(x['sha256']==SHA and x['exact_sample_row_order']for x in export['records'])
r=csv.DictReader(file.open());assert r.fieldnames==['application_id','default_probability'];rows=list(r);assert len(rows)==30000 and len({x['application_id']for x in rows})==30000;assert [x['application_id']for x in rows]==json.loads((OLD/'EXPECTED_TEST_IDS.json').read_text());assert all(math.isfinite(float(x['default_probability']))and 0<=float(x['default_probability'])<=1 for x in rows)
def listing(label):
 q=call(['competitions','submissions','-c',COMP,'--csv'],label);assert q['returncode']==0,'Authenticated server listing failed';return list(csv.DictReader(io.StringIO(q['stdout'])))
before=listing('before');found=next((x for x in before if x.get('description')==MSG),None)
if not found:
 intent=P/'SUBMISSION_INTENT.json';assert not intent.exists(),'Existing intent: reconcile first, never blind retry';dump(intent,dict(epoch=time.time(),authorization='User explicitly requested submit this annotated heldblend',CSV_sha256=SHA,message=MSG,development_KS=.5319679799870825));call(['competitions','submit','-c',COMP,'-f',str(file),'-m',MSG],'upload')
for i in range(12):
 after=listing('after'+str(i));found=next((x for x in after if x.get('description')==MSG),None)
 if found and ('complete'in str(found.get('status','')).lower()or i>=4):break
 time.sleep(5)
assert found,'Upload unconfirmed; inspect private response, no repeat'
v=dict(status='SERVER_OBSERVED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),competition=COMP,CSV_sha256=SHA,local_development_KS=.5319679799870825,receipt=found,new_base_training=False,authorization='Explicit user request: well submit it',message=MSG);dump(R/'SUBMITTED.json',v);print(json.dumps(v),flush=True)
