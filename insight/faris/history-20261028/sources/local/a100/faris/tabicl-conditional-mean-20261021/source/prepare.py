"""Ordinary queued admission; never reads performance to decide admission."""
from pathlib import Path
import os,json,time,hashlib,subprocess,sys
R=Path(__file__).resolve().parents[1];D=1792585871.823748
from huggingface_hub import HfApi,hf_hub_download

def write(p,x):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
assert time.time()<D-10800
plan=json.loads((R/'plan.json').read_text())
for n,h in plan['source_sha256'].items():assert hashlib.sha256((R/'source'/n).read_bytes()).hexdigest()==h
info=HfApi(token=False).model_info('jingang/TabICL');path=Path(hf_hub_download('jingang/TabICL','tabicl-regressor-v2-20260212.ckpt',revision=info.sha,cache_dir=R/'hf-cache',token=False))
h=hashlib.sha256(path.read_bytes()).hexdigest();write(R/'WEIGHT.json',dict(path=str(path),sha256=h,revision=info.sha,repo='jingang/TabICL'))
plan['checkpoint_sha256']=h;write(R/'plan.json',plan)
parent=R.parent/'tabicl-compact-a100-20261021';launch=json.loads((parent/'LAUNCH.json').read_text());owner_pid=launch['pid']
while True:
 assert time.time()<D-10800,'Original admission deadline; do not restart clock'
 assert not(R/'STOP').exists()
 try:
  stat=Path(f'/proc/{owner_pid}/stat').read_text();live=stat[stat.rfind(')')+2:].split()[0]!='Z'
 except FileNotFoundError:live=False
 free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
 if not live and free>=40000:break
 write(R/'QUEUE.json',dict(status='waiting_for_safe_GPU_admission',parent_pid=owner_pid,parent_live=live,free_VRAM_MiB=free,
   admission_depends_on_performance=False,original_deadline=D))
 time.sleep(10)
write(R/'QUEUE.json',dict(status='preflight',original_deadline=D))
code=subprocess.call([sys.executable,str(R/'source/experiment.py'),'preflight']);assert code==0,code
assert json.loads((R/'PREFLIGHT.json').read_text())['status']=='passed'
plan['training_admitted']=True;write(R/'plan.json',plan)
write(R/'QUEUE.json',dict(status='admitted',original_deadline=D,admission_depends_on_performance=False))
code=subprocess.call([sys.executable,str(R/'source/runner.py')]);write(R/'QUEUE.json',dict(status='complete'if code==0 else'failed_or_partial',exit_code=code,original_deadline=D))
