"""Read-only runtime gate collection; no scientific worker or controller restart."""
from pathlib import Path
import json,os,subprocess,shlex,time,traceback
R=Path('/home/faris/logika-overnight-refit-20261030/helma-book-context10k');P=R.parent/'private/helma-books/account2';D=P/'runtime_metadata';D.mkdir(exist_ok=True)
V='/root/logika-helma-books-20261030';name='logika-0630-helma-books-20261030';key='/home/faris/.ssh/id_ed25519_logika_lab';os.environ['PATH']='/home/faris/.local/bin:'+os.environ['PATH']
prefix=['/home/faris/.local/bin/colab','--account','2','--config',str(P/'sessions.json')]
opts=['-o','BatchMode=yes','-o','ConnectTimeout=30','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(P/'known_hosts'),'-o','HostKeyAlias='+name,'-i',key,'-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath='+str(Path.home()/'.cache/logika-colab-control/helma-books-a2'),'-o','ProxyCommand='+shlex.join(prefix+['ssh','--proxy-mode','--session',name,'--identity',key])]
clock=json.load(open(R/'CLOCK.json'));can='/mnt/home/faris/logika-overnight-refit-20261030/helma-book-context10k'
while time.time()<clock['model_deadline_epoch']+150:
 if(P/'RELEASED.json').exists():break
 try:
  code="from pathlib import Path;import json;r=Path("+repr(V)+");ps=list(r.glob('GATE_FAMILY_*.json'))+[r/n for n in ['BACKEND_STATE.json','BACKEND_FINAL.json','ENVIRONMENT_LIVE.json','DISPATCH.json','RELEASE_WATCHDOG.json','QUEUE_DISPATCH.json']];print(json.dumps({p.name:json.load(open(p))for p in ps if p.exists()}))"
  q=subprocess.run(['ssh','-T',*opts,'root@colab','python3 -'],input=code,capture_output=True,text=True,check=True,timeout=90);records=json.loads(q.stdout)
  for n,v in records.items():
   if n.startswith('GATE_FAMILY_'):
    assert v['clock_sha256']==__import__('hashlib').sha256((R/'CLOCK.json').read_bytes()).hexdigest()
   p=D/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
  subprocess.run(['ssh','faris@100.68.29.126','mkdir -p '+can+'/private/helma_runtime_metadata'],check=True,capture_output=True,timeout=30)
  subprocess.run(['rsync','-az',str(D)+'/', 'faris@100.68.29.126:'+can+'/private/helma_runtime_metadata/'],check=True,capture_output=True,timeout=90)
  gates={k:v for k,v in records.items()if k.startswith('GATE_FAMILY_')}
  projection=json.dumps(dict(epoch=time.time(),owner='helma_account2',runtime_gate_records=gates,source='exact same-session VM gate JSON; no reconstructed predicates or score-based changes'))
  remote="from pathlib import Path;p=Path("+repr(can+"/RUNTIME_ADMISSION_EVIDENCE.json")+");t=p.with_suffix('.tmp');t.write_text("+repr(projection+"\n")+");t.replace(p)"
  subprocess.run(['ssh','faris@100.68.29.126','python3 -'],input=remote,capture_output=True,text=True,check=True,timeout=45)
  (P/'RUNTIME_METADATA_COLLECTION.json').write_text(json.dumps(dict(epoch=time.time(),records=len(records),passing_design_gates=sum(v.get('status')=='PASS'for k,v in records.items()if k.startswith('GATE_FAMILY_')),preserved_before_owned_release=True))+'\n')
 except Exception:(P/'RUNTIME_METADATA_COLLECTION_FAILURE_PRIVATE.log').write_text(traceback.format_exc())
 time.sleep(30)
