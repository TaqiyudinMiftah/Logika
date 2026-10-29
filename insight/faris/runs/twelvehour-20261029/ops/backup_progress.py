"""Workstation private compact-evidence backup; preserve512MiB; never fit or upload."""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,os,re,shlex,shutil,subprocess,time,traceback
R=Path(__file__).resolve().parents[1];REMOTE='faris@100.68.29.126';ROOT='/mnt/home/faris/logika-twelvehour-20261029';RESERVE=512*1024**2

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def run(args):return subprocess.run(args,capture_output=True,text=True,check=True,timeout=180)
def guarded_transfer(args):
 preview=run([*args[:-2],'--dry-run','--stats',*args[-2:]])
 m=re.search(r'Total transferred file size:\s*([\d,]+)',preview.stdout);assert m,'No byte forecast; do not assume zero'
 expected=int(m.group(1).replace(',',''));free=shutil.disk_usage(R).free
 if free-expected<RESERVE:raise RuntimeError(f'Storage reserve requires defer; free={free},forecast={expected},reserve={RESERVE}')
 run(args);return expected

def main(once=False):
 private=R/'private';private.mkdir(exist_ok=True);lock=(private/'backup.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out=R/'INCREMENTAL_EVIDENCE';out.mkdir(exist_ok=True)
 while True:
  try:
   # Small public state first. No labels/data/packets or modelweights are read.
   for name in ['CLOCK.json','LIVE_STATE.json','COMPLETION.json','ARTIFACT_CONSISTENCY_AUDIT.json']:
    p=subprocess.run(['rsync','-az',REMOTE+':'+ROOT+'/'+name,str(R)+'/'],capture_output=True,text=True,timeout=30)
    assert p.returncode in [0,23],p.stderr
   for name in ['runs','composed','evaluation']:
    target=out/name;target.mkdir(exist_ok=True);args=['rsync','-az','--include=*/','--include=prediction.npz','--include=*.json','--include=*.jsonl','--exclude=*',REMOTE+':'+ROOT+'/'+name+'/',str(target)+'/'];guarded_transfer(args)
   manifest={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.rglob('*') if p.is_file()}
   checked=0
   for p in (out/'runs').glob('*/attempt1/RESULT.json'):
    result=json.loads(p.read_text());assert result['status']=='complete';assert sha(p.parent/'prediction.npz')==result['prediction_sha256'];assert sha(p.parent/'START.json')==result['start_sha256'];assert result['weights_saved'] is False and result['outer_labels_loaded'] is False;checked+=1
   dump(private/'PRIVATE_BACKUP_MANIFEST.json',manifest);receipt=dict(status='HASH_VERIFIED_INCREMENTAL',UTC=utc(),files=len(manifest),bytes=sum(x['bytes'] for x in manifest.values()),completed_fit_records_checked=checked,manifest_sha256=sha(private/'PRIVATE_BACKUP_MANIFEST.json'),free_bytes=shutil.disk_usage(R).free,minimum_free_bytes=RESERVE,case_predictions_private=True,case_labels_collected=False,weights_or_KV_collected=False)
   dump(private/'INCREMENTAL_BACKUP_STATE.json',receipt)
   if (R/'COMPLETION.json').exists() and (R/'ARTIFACT_CONSISTENCY_AUDIT.json').exists():
    audit=json.loads((R/'ARTIFACT_CONSISTENCY_AUDIT.json').read_text());assert audit['status']=='PASS' and checked==audit['completed_fits'];dump(private/'FINAL_BACKUP_RECEIPT.json',dict(receipt,completion_sha256=sha(R/'COMPLETION.json'),audit_sha256=sha(R/'ARTIFACT_CONSISTENCY_AUDIT.json')));return
   if once:return
   if (R/'CLOCK.json').exists() and time.time()>json.loads((R/'CLOCK.json').read_text())['deadline_epoch']+600:dump(private/'BACKUP_NEEDS_REVIEW.json',dict(UTC=utc(),reason='No complete final collection withinboundedtail; canonicalevidencepreserved'));return
  except Exception as e:
   try:
    (private/'INCREMENTAL_BACKUP_FAILURE_PRIVATE.txt').write_text(traceback.format_exc());dump(private/'INCREMENTAL_BACKUP_STATE.json',dict(status='DEFERRED_OR_FAILED',UTC=utc(),exception_class=type(e).__name__,failure_sha256=sha(private/'INCREMENTAL_BACKUP_FAILURE_PRIVATE.txt'),minimum_free_bytes=RESERVE,canonical_evidence_retained=True))
   except OSError:pass
   if once:raise
  time.sleep(180)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args();main(a.once)
