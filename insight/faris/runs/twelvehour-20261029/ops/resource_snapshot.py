"""Read-only resource evidence on the persistent control host; never allocates."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime,json,os,re,subprocess,time,traceback
R=Path('/home/faris/logika-twelvehour-20261029');P=R/'private/colab';C=json.load(open(R/'CLOCK.json'));os.environ['PATH']='/home/faris/.local/bin:'+os.environ['PATH']
def usage(n):
 d=P/f'account{n}';p=subprocess.run(['/home/faris/.local/bin/colab','--account',str(n),'--config',str(d/'sessions.json'),'usage'],capture_output=True,text=True,timeout=90)
 (d/'USAGE_LATEST_PRIVATE.json').write_text(json.dumps(dict(epoch=time.time(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))+'\n')
 result=dict(account=n,released_flag=(d/'RELEASED.json').exists(),usage_read_ok=p.returncode==0)
 for key,pattern in [('balance_CU',r'Current balance:\s*([\d.]+)'),('rate_CU_hour',r'Usage rate:\s*([\d.]+)'),('assignments',r'Active assignments:\s*(\d+)')]:
  m=re.search(pattern,p.stdout);result[key]=float(m.group(1))if m and key!='assignments'else(int(m.group(1))if m else None)
 for name in ['FINAL_STATE.json','CONTROLLER_STATE.json']:
  f=d/name
  if f.exists():
   v=json.load(open(f));result['terminal'if name.startswith('FINAL')else'controller_stage']=v if name.startswith('FINAL')else v.get('stage')
 return result
while time.time()<C['deadline_epoch']-60:
 try:
  with ThreadPoolExecutor(max_workers=6)as pool:accounts=list(pool.map(usage,range(1,7)))
  value=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),accounts=accounts,original_clock_sha256=__import__('hashlib').sha256((R/'CLOCK.json').read_bytes()).hexdigest(),policy='Later human-authorized resource addendum supersedes initial no_paid_allocation flag; source, features and clock unchanged',observed_paid_account2_start_CU=337.28,resource_snapshot_only=True)
  all_released=all(a['released_flag']and a['usage_read_ok']and a['assignments']==0 for a in accounts);value['all_owned_sessions_released_and_no_account_assignments']=all_released
  p=R/'CURRENT_RESOURCE_STATE.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(p)
  subprocess.run(['scp','-q',str(p),'faris@100.68.29.126:/mnt/home/faris/logika-twelvehour-20261029/CURRENT_RESOURCE_STATE.json'],check=True,capture_output=True,timeout=90)
  with(P/'RESOURCE_HISTORY.jsonl').open('a')as f:f.write(json.dumps(value)+'\n')
  print(json.dumps(dict(UTC=value['UTC'],active_assignments=sum(a['assignments'] or 0 for a in accounts),all_released=all_released)),flush=True)
  if all_released:break
 except Exception:(P/'RESOURCE_SNAPSHOT_FAILURE_PRIVATE.log').write_text(traceback.format_exc())
 time.sleep(min(1800,max(1,C['deadline_epoch']-time.time()-60)))
