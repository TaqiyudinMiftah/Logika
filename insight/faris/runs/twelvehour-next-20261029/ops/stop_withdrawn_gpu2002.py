"""Shutdown-only watcher for this campaign's withdrawn node2002 worker."""
from pathlib import Path
import json,time,subprocess,hashlib,datetime
R=Path(__file__).resolve().parents[1];clock=json.load(open(R/'CLOCK.json'));key='/mnt/home/faris/.ssh/id_ed25519_logika_backup119';unit='logika-next-extra-gpu2002-20261029.service'
assert (R/'NODE2002_RESOURCE_WITHDRAWAL.json').exists()
while time.time()<clock['deadline_epoch']-10:
 cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-i',key,'-p2002','faris@gpu-labkc.ub.ac.id','systemctl --user stop '+unit+'; systemctl --user show '+unit+' -p ActiveState -p MainPID']
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
 # Inactive or unloaded after stop is sufficient; never touch other units.
 success=p.returncode==0 and ('ActiveState=inactive'in p.stdout or'MainPID=0'in p.stdout)
 v=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='OWNED_WORKER_STOPPED'if success else'SHUTDOWN_PENDING_UNREACHABLE',owned_unit=unit,returncode=p.returncode,diagnostic_sha256=hashlib.sha256((p.stdout+p.stderr).encode()).hexdigest(),no_training_or_allocation=True,original_absolute_deadline_epoch=clock['deadline_epoch'])
 q=R/'NODE2002_SHUTDOWN_STATE.json';t=q.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(q)
 if success:break
 time.sleep(120)
