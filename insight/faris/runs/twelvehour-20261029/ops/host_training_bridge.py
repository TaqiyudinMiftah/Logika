from pathlib import Path
import hashlib,json,subprocess,sys,time,traceback
R=Path('/home/faris/logika-twelvehour-20261029');P=R/'private/host119';P.mkdir(parents=True,exist_ok=True);C=json.load(open(R/'CLOCK.json'));LAB='faris@100.68.29.126';CANON='/mnt/home/faris/logika-twelvehour-20261029'
def call(a,timeout=180):return subprocess.run(a,check=True,capture_output=True,text=True,timeout=timeout)
def collect():
 count=0
 for p in(R/'runs').glob('*/attempt1/RESULT.json'):
  v=json.load(open(p));assert hashlib.sha256((p.parent/'prediction.npz').read_bytes()).hexdigest()==v['prediction_sha256'];assert hashlib.sha256((p.parent/'START.json').read_bytes()).hexdigest()==v['start_sha256'];count+=1
 if(R/'runs').exists():call(['rsync','-az','--ignore-existing',str(R/'runs')+'/',LAB+':'+CANON+'/runs/'])
 for n in ['BACKEND_GATE.json','BACKEND_STATE.json','BACKEND_FINAL.json']:
  if(R/n).exists():call(['scp','-q',str(R/n),LAB+':'+CANON+'/HOST119_'+n])
 (P/'COLLECTION_RECEIPT.json').write_text(json.dumps(dict(epoch=time.time(),verified_fits=count))+'\n')
 return count
unit='logika-host119-training-20261029.service'
call(['systemd-run','--user','--unit='+unit,'--property=MemoryMax=44G','--property=CPUQuota=800%','--property=RuntimeMaxSec='+str(int(C['model_deadline_epoch']-time.time()-60)),'--property=StandardOutput=append:'+str(R/'host119_queue.log'),'--property=StandardError=append:'+str(R/'host119_queue.log'),str(R/'env311/bin/python'),'-u',str(R/'ops/backend_queue.py'),'--owner','host119'])
while time.time()<C['model_deadline_epoch']-150:
 try:collect()
 except Exception:(P/'COLLECTION_FAILURE.log').write_text(traceback.format_exc())
 if(R/'BACKEND_FINAL.json').exists():break
 if(R/'BACKEND_GATE.json').exists()and json.load(open(R/'BACKEND_GATE.json'))['status']!='PASS':break
 state=call(['systemctl','--user','show',unit,'-p','ActiveState','--value']).stdout.strip()
 if state not in ['active','activating']:break
 time.sleep(20)
call(['systemctl','--user','stop',unit]);count=collect();p=P/'FINAL_STATE.json';p.write_text(json.dumps(dict(epoch=time.time(),collected=True,verified_fits=count,original_clock_preserved=True))+'\n');call(['scp','-q',str(p),LAB+':'+CANON+'/BACKEND_COLLECTION_host119_COMPLETE.json'])
