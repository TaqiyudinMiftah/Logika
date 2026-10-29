from pathlib import Path
import json,time
R=Path('/mnt/home/faris/logika-twelvehour-20261029');c=json.load(open(R/'CLOCK.json'))
flags=['BACKEND_COLLECTION_account'+str(n)+'_COMPLETE.json'for n in range(1,7)]+['BACKEND_COLLECTION_host119_COMPLETE.json']
while time.time()<c['model_deadline_epoch']-100:
 if all((R/f).exists()for f in flags):break
 time.sleep(10)
value=dict(epoch=time.time(),all_collected=all((R/f).exists()for f in flags),missing=[f for f in flags if not(R/f).exists()],original_clock_preserved=True)
p=R/'EXTERNAL_BACKENDS_COLLECTION_COMPLETE.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)
p=R/'DEFERRED_QUEUE_gpu2001_FINAL.json'
if p.exists():p.replace(R/'QUEUE_gpu2001_FINAL.json')
