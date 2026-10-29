"""One authenticated, audited, idempotent deadline submission on the personal host."""
from pathlib import Path
import csv,datetime,fcntl,hashlib,json,math,os,re,subprocess,time
R=Path(__file__).resolve().parents[1];P=R/'private';CLI=R/'kaggle-env/bin/kaggle';COMP='data-science-competition-logika-ui-2026'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def call(args,record):
 token=(P/'kaggle-token').read_text().strip();assert (P/'kaggle-token').stat().st_mode&0o077==0
 deadline=json.load(open(R/'CLOCK.json'))['deadline_epoch'];remaining=deadline-time.time()-10
 assert remaining>0,'Absolute submission deadline reached'
 try:
  q=subprocess.run([str(CLI),*args],env={**os.environ,'KAGGLE_API_TOKEN':token},text=True,capture_output=True,timeout=min(60,max(1,remaining)))
  d=dict(epoch=time.time(),args=args,returncode=q.returncode,stdout=q.stdout.replace(token,'[REDACTED]'),stderr=q.stderr.replace(token,'[REDACTED]'))
 except subprocess.TimeoutExpired as e:
  clean=lambda x:(x.decode(errors='replace')if isinstance(x,bytes)else x or'').replace(token,'[REDACTED]')
  d=dict(epoch=time.time(),args=args,returncode=None,stdout=clean(e.stdout),stderr=clean(e.stderr),timeout=True,upload_may_be_ambiguous=True)
 target=R/'private'/record
 if target.exists():record=target.stem+'_'+str(time.time_ns())+target.suffix
 dump('private/'+record,d);return d
def validate(p,expected,ids=None):
 assert sha(p)==expected
 with p.open()as f:
  reader=csv.DictReader(f);assert reader.fieldnames==['application_id','default_probability'];rows=list(reader)
 assert len(rows)==30000 and len({x['application_id']for x in rows})==30000
 assert all(math.isfinite(float(x['default_probability']))and 0<=float(x['default_probability'])<=1 for x in rows)
 if ids is not None:assert [x['application_id']for x in rows]==ids
 return [x['application_id']for x in rows]
def main():
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--preflight',action='store_true');args=a.parse_args()
 clock=json.load(open(R/'CLOCK.json'));ids=json.load(open(R/'EXPECTED_TEST_IDS.json'))
 if args.preflight:
  q=call(['competitions','submissions','-c',COMP,'--csv'],'PREFLIGHT_SUBMISSIONS.json');assert q['returncode']==0;dump('SUBMISSION_PREFLIGHT.json',dict(status='PASS',rows=len(ids),authenticated=True,no_historical_fallback=True));return
 lock=(P/'submit.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if(R/'SUBMITTED.json').exists():return
 while time.time()<clock['submit_epoch']:time.sleep(min(30,clock['submit_epoch']-time.time()))
 assert time.time()<clock['deadline_epoch']
 selected=None
 ready=R/'READY_CANDIDATE.json'
 if ready.exists():
  v=json.load(open(ready));assert v['campaign']=='overnight-refit-20261030' and v['fresh_training'] is True;p=Path(v['path']);audit=json.load(open(Path(v['audit_path'])));assert audit['status']=='PASS' and audit['sha256']==v['sha256'];validate(p,v['sha256'],ids);selected=v
 if selected is None:
  dump('NO_SUBMISSION.json',dict(epoch=time.time(),reason='No newly trained audited candidate; human prohibits historical fallback'));return
 validate(Path(selected['path']),selected['sha256'],ids);dump('FROZEN_SUBMISSION.json',dict(epoch=time.time(),**selected))
 message='Logika0630_'+selected['sha256'][:12]
 prior=None
 for attempt in range(3):
  prior=call(['competitions','submissions','-c',COMP,'--csv'],f'BEFORE_SUBMISSION_{attempt}.json')
  if prior['returncode']==0:break
  if time.time()>clock['deadline_epoch']-120:break
  time.sleep(5)
 assert prior['returncode']==0,'Read-only pre-upload listing failed; no upload issued'
 if message in prior['stdout']:
  dump('SUBMITTED.json',dict(status='already_observed',message=message,selected=selected,epoch=time.time()));return
 dump('UPLOAD_INTENT.json',dict(epoch=time.time(),message=message,sha256=selected['sha256'],only_one_upload_allowed=True))
 response=call(['competitions','submit','-c',COMP,'-f',selected['path'],'-m',message],'SUBMISSION_RESPONSE.json')
 # Always reconcile the server before considering another upload; never blind retry.
 for i in range(6):
  if time.time()>clock['deadline_epoch']-15:break
  listing=call(['competitions','submissions','-c',COMP,'--csv'],f'AFTER_SUBMISSION_{i}.json')
  if listing['returncode']==0 and message in listing['stdout']:
   dump('SUBMITTED.json',dict(status='server_observed',message=message,selected=selected,epoch=time.time(),response_returncode=response['returncode']));return
  if time.time()>clock['deadline_epoch']-20:break
  time.sleep(10)
 dump('SUBMISSION_NEEDS_ATTENTION.json',dict(epoch=time.time(),message=message,upload_returncode=response['returncode'],no_blind_retry=True));raise RuntimeError('Upload not confirmed by authenticated server listing')
if __name__=='__main__':main()
