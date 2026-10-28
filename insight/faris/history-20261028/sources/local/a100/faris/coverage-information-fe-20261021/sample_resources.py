"""Finite read-only campaign resource sampling; never signals any process."""
from pathlib import Path
import json,time,subprocess,os,hashlib
R=Path(__file__).resolve().parent
def write(value,path):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(path)
def main():
 deadline=json.loads((R/'training-plan.json').read_text())['original_deadline'];unit='logika-coverage-information-gpu-20261021.service'
 cg=subprocess.check_output(['systemctl','--user','show',unit,'--property=ControlGroup','--value'],text=True).strip();base=Path('/sys/fs/cgroup')/cg.lstrip('/')
 started=time.time();peak=0;gpu_peak={};count=0
 while time.time()<deadline:
  state=json.loads((R/'TRAINING_STATUS.json').read_text());observed=dict(time=time.time(),status=state['status'],workers=[])
  for a in state.get('active',[]):
   proc=Path('/proc')/str(a['pid']);cmd=proc/'cmdline'
   if cmd.exists()and str(R/'source/worker.py').encode()in cmd.read_bytes():
    values=(proc/'status').read_text().splitlines();rss=[v for v in values if v.startswith('VmRSS:')];observed['workers'].append(dict(**a,rss_kB=int(rss[0].split()[1])if rss else None))
  for name in ['memory.current','memory.peak','cpu.stat']:
   p=base/name
   if p.exists():observed[name]=p.read_text().strip()
  if 'memory.peak'in observed:peak=max(peak,int(observed['memory.peak']))
  try:
   raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
   owned={w['pid']:w['gpu']for w in observed['workers']};observed['owned_gpu_memory_MiB']={}
   for line in raw.splitlines():
    pid,memory=[v.strip()for v in line.split(',')]
    if int(pid)in owned and memory.isdigit():
     gpu=owned[int(pid)];observed['owned_gpu_memory_MiB'][str(gpu)]=int(memory);gpu_peak[str(gpu)]=max(gpu_peak.get(str(gpu),0),int(memory))
  except subprocess.CalledProcessError:observed['GPU_observation_failed']=True
  with(R/'resource_samples.jsonl').open('a')as f:f.write(json.dumps(observed)+'\n')
  count+=1;write(dict(status='sampling'if state['status']in ['running','reporting']else'complete',started=started,observed_at=time.time(),samples=count,cgroup_memory_peak_bytes=peak,observed_owned_GPU_peak_MiB=gpu_peak,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Observation started after initial worker launch. GPU values are sampled owned-process memory, not guaranteed lifetime peaks. memory.peak is the campaign cgroup kernel high-water mark while available. No unrelated process resource use is attributed.'),R/'RESOURCE_SUMMARY.json')
  if state['status']in ['complete','failed','partial']:return
  time.sleep(10)
if __name__=='__main__':main()
