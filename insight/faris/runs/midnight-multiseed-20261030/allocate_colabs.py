"""Explicit authorized independent profiles; one bounded allocation attempt each.

Read-only inventory first. No quota churn, unknown-session shutdown or deadline reset.
"""
from pathlib import Path
import concurrent.futures,datetime,json,os,subprocess,time
R=Path('/home/faris/logika-midnight-multiseed-20261030');P=R/'private/colab';P.mkdir(parents=True,exist_ok=True)
os.environ['PATH']='/home/faris/.local/bin:'+os.environ['PATH'];CLI='/home/faris/.local/bin/colab'
C=json.load(open(R/'CLOCK.json'));assert time.time()<C['admission_end_epoch']
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def one(account):
    d=P/f'account{account}';d.mkdir(exist_ok=True);name=f'logika-midnight-a{account}-20261030';prefix=[CLI,'--account',str(account),'--config',str(d/'sessions.json')]
    assert not(d/'ALLOCATION_INTENT.json').exists(),'Never repeat allocation without reconciling real sessions'
    for op in ['usage','sessions']:
        p=subprocess.run([*prefix,op],capture_output=True,text=True,timeout=90);dump(d/f'INITIAL_{op}.json',dict(UTC=utc(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));p.check_returncode()
    gpu='A100'if account==2 else'T4';dump(d/'ALLOCATION_INTENT.json',dict(UTC=utc(),account=account,session=name,requested_gpu=gpu,original_deadline_epoch=C['deadline_epoch'],user_explicitly_requested_all_Colab_profiles=True))
    try:
        p=subprocess.run([*prefix,'new','--session',name,'--gpu',gpu],capture_output=True,text=True,timeout=240);dump(d/'ALLOCATION_RESPONSE.json',dict(UTC=utc(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        if p.returncode:
            dump(d/'FINAL_STATE.json',dict(status='allocation_refused_no_retry',account=account,session=name,UTC=utc(),scientific_conclusion=False));return
        code='''import threading,time,json
from pathlib import Path
r=Path('/root/logika-midnight-multiseed-20261030');r.mkdir(exist_ok=True)
deadline=DEADLINE-120;setup_deadline=time.time()+1800
def release():
 while time.time()<deadline:
  if time.time()>setup_deadline and not any((r/'runs').glob('*/attempt1/START.json')):break
  time.sleep(15)
 from google.colab import runtime
 runtime.unassign()
threading.Thread(target=release,daemon=True).start()
(r/'RELEASE_WATCHDOG.json').write_text(json.dumps({'owned_release_deadline':deadline,'original_campaign_deadline':DEADLINE,'setup_deadline':setup_deadline,'only_owned_session':True})+'\\n')
print('Independent owned-session deadline watchdog active')
'''.replace('DEADLINE',repr(C['deadline_epoch']))
        script=d/'release_watchdog.py';script.write_text(code)
        q=subprocess.run([*prefix,'exec','--session',name,'-f',str(script)],capture_output=True,text=True,timeout=120);dump(d/'WATCHDOG_RESPONSE.json',dict(UTC=utc(),returncode=q.returncode,stdout=q.stdout,stderr=q.stderr));q.check_returncode()
        dump(d/'ALLOCATED_READY.json',dict(UTC=utc(),account=account,session=name,requested_gpu=gpu,watchdog=True,deadline_epoch=C['deadline_epoch']))
    except Exception as e:
        dump(d/'NEEDS_RECONCILIATION.json',dict(UTC=utc(),account=account,session=name,exception_class=type(e).__name__,no_blind_retry=True))
        raise
    finally:
        q=subprocess.run([*prefix,'usage'],capture_output=True,text=True,timeout=90);dump(d/'POST_ALLOCATION_USAGE.json',dict(UTC=utc(),returncode=q.returncode,stdout=q.stdout,stderr=q.stderr))
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as executor:
    futures={executor.submit(one,a):a for a in range(1,12)}
    for f in concurrent.futures.as_completed(futures):
        a=futures[f]
        try:f.result();print(json.dumps({'account':a,'allocation_step':'finished','UTC':utc()}),flush=True)
        except Exception as e:print(json.dumps({'account':a,'error_class':type(e).__name__,'private_receipts_retained':True}),flush=True)
dump(P/'ALLOCATION_PASS_COMPLETE.json',dict(UTC=utc(),profiles_checked=11,deadline_epoch=C['deadline_epoch'],results={str(a):('allocated'if(P/f'account{a}/ALLOCATED_READY.json').exists()else'refused_or_needs_reconciliation')for a in range(1,12)}))
