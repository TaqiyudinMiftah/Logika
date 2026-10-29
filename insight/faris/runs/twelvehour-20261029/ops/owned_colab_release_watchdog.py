"""Persistent-host backstop for this invocation's six owned session names only."""
from pathlib import Path
import datetime,json,os,subprocess,time
R=Path('/home/faris/logika-twelvehour-20261029');P=R/'private/colab';C=json.load(open(R/'CLOCK.json'))
os.environ['PATH']='/home/faris/.local/bin:'+os.environ['PATH']
while True:
    active=[]
    for a in range(1,7):
        d=P/f'account{a}';intent=d/'ALLOCATION_INTENT.json';response=d/'ALLOCATION_RESPONSE.json'
        if not intent.exists()or not response.exists()or(d/'RELEASED.json').exists():continue
        if json.load(open(response))['returncode']!=0:continue
        active.append(a);started=datetime.datetime.fromisoformat(json.load(open(intent))['UTC']).timestamp()
        marker=d/'CONTROLLER_TRAINING_STARTED.json'
        if time.time()<C['deadline_epoch'] and (marker.exists()or time.time()<started+1800):continue
        name=json.load(open(intent))['session'];prefix=['/home/faris/.local/bin/colab','--account',str(a),'--config',str(d/'sessions.json')]
        p=subprocess.run([*prefix,'stop','--session',name],capture_output=True,text=True,timeout=150)
        (d/'WATCHDOG_RELEASE_RESPONSE.json').write_text(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))+'\n')
        if p.returncode==0:(d/'RELEASED.json').write_text(json.dumps(dict(epoch=time.time(),session=name,reason='originaldeadlineorsetupadmissiontimeout'))+'\n')
    if not active and (P/'ALLOCATION_PASS_COMPLETE.json').exists():break
    if time.time()>C['deadline_epoch']+300:break
    time.sleep(20)
