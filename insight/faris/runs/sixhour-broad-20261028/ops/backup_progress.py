"""Incremental private compact-evidence backup; no data tables or model weights."""
from pathlib import Path
import json,subprocess,time,traceback,datetime
R=Path(__file__).resolve().parents[1];end=json.loads((R/'CLOCK.json').read_text())['deadline_epoch'];H='faris@100.68.29.126';remote='/mnt/home/faris/logika-sixhour-broad-20261028';out=R/'INCREMENTAL_EVIDENCE';out.mkdir(exist_ok=True)
while time.time()<end:
 try:
  for name in ['runs','composed','evaluation','dispatch']:
   target=out/name;target.mkdir(exist_ok=True)
   p=subprocess.run(['rsync','-az','--include=*/','--include=prediction.npz','--include=predictions.npz','--include=*.json','--include=*.jsonl','--include=*.md','--exclude=*',H+':'+remote+'/'+name+'/',str(target)+'/'],text=True,capture_output=True,timeout=max(30,min(900,int(end-time.time()))))
   if p.returncode not in [0,23]:raise RuntimeError(p.stderr)
  (R/'private/INCREMENTAL_BACKUP_STATE.json').write_text(json.dumps(dict(epoch=time.time(),stage='compact_private_backup_current',file_count=sum(p.is_file()for p in out.rglob('*')),no_fitted_weights_or_KV=True))+'\n')
 except Exception:
  (R/'private/INCREMENTAL_BACKUP_FAILURE.txt').write_text(traceback.format_exc())
 time.sleep(max(1,min(300,end-time.time())))
