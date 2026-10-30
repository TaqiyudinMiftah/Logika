from pathlib import Path
import subprocess,time,json,shlex,sys
R=Path(__file__).resolve().parent/'gatedmeta';ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-p2003','faris@gpu-labkc.ub.ac.id'];target=str(R)
while time.time()<1793364605:
 q=subprocess.run(ssh+['python3 -c '+shlex.quote('from pathlib import Path;import json;p=Path('+repr(target)+');print(json.dumps([p.joinpath("GPU_OWNER_FINAL_"+str(i)+".json").exists()for i in [0,1]]))')],capture_output=True,text=True,timeout=20,check=True)
 if all(json.loads(q.stdout)):break
 time.sleep(10)
R.mkdir(exist_ok=True);subprocess.run(['rsync','-az','--exclude=bank/','--exclude=__pycache__/','-e',shlex.join(ssh[:-1]),ssh[-1]+':'+target+'/',str(R)+'/'],check=True)
for name in ['bank','BANK_FROZEN.json']:
 if not (R/name).exists():(R/name).symlink_to('../'+name)
(R/'PLAN.json').write_bytes((R/'GPU_PLAN.json').read_bytes());(R/'controls').mkdir(exist_ok=True)
predictions=list(R.glob('gpu_results/*/s*/fold*/FROZEN.json'));p=dict(terminal_UTC=time.strftime('%FT%TZ',time.gmtime()),completed_fit_packets=len(predictions),planned_fit_slots=160,no_automatic_retry=True);(R/'COLLECTION.json').write_text(json.dumps(p,indent=2)+'\n')
remaining=1793365060-time.time();assert remaining>0
subprocess.run(['/mnt/home/faris/logika-sixhour-broad-20261028/env311/bin/python','-u',str(R/'audit_all.py')],check=True,timeout=remaining)
