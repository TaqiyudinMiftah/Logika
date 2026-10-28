"""Observational only: samples campaign PIDs and their own resource group."""
from pathlib import Path
import json,time,os,subprocess,sys
R=Path(__file__).resolve().parent;unit=sys.argv[1];D=1792585871.823748;peaks={};group_peak=0;last_status=None
with(R/'resource_samples.jsonl').open('a',buffering=1)as log:
 while time.time()<D:
  try:state=json.loads((R/'TRAINING_STATUS.json').read_text())
  except(FileNotFoundError,ValueError):state={}
  gpu={}
  try:
   for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines():
    a,b=row.split(',');gpu[int(a)]=int(b.strip())
  except(Exception):pass
  workers=[]
  for a in state.get('active',[]):
   pid=a['pid'];p=Path('/proc')/str(pid)
   try:
    if p.stat().st_uid!=os.getuid()or str(R/'source/worker.py').encode()not in(p/'cmdline').read_bytes():continue
    fields={line.split(':',1)[0]:line.split(':',1)[1].strip()for line in(p/'status').read_text().splitlines()if':'in line};rss=int(fields.get('VmHWM','0').split()[0]);entry=peaks.setdefault(str(a['job']),dict(pid=pid,observed_peak_RSS_KiB=0,observed_peak_GPU_MiB=0));entry['observed_peak_RSS_KiB']=max(entry['observed_peak_RSS_KiB'],rss);entry['observed_peak_GPU_MiB']=max(entry['observed_peak_GPU_MiB'],gpu.get(pid,0));workers.append(dict(job=a['job'],pid=pid,RSS_highwater_KiB=rss,VRAM_MiB=gpu.get(pid,0)))
   except(FileNotFoundError,ProcessLookupError):pass
  try:
   gate=json.loads((R/'TRAINING_RESOURCE_GATE.json').read_text());cg=Path(gate['cgroup']);group_peak=max(group_peak,int((cg/'memory.peak').read_text()));current=int((cg/'memory.current').read_text())
  except(FileNotFoundError,ValueError):current=None
  row=dict(time=time.time(),workers=workers,cgroup_memory_current=current,cgroup_memory_peak=group_peak);log.write(json.dumps(row)+'\n')
  summary=dict(status='sampling',sampled_worker_peaks=peaks,cgroup_memory_peak=group_peak,qualification='Samples began after initialjobs; perjob peaks observational, not proof of exactVRAMpeak. Cgroupmemory.peak coverswholeownedtraininggroup lifetime when available. No process terminated or modified.')
  t=R/'RESOURCE_SUMMARY.tmp';t.write_text(json.dumps(summary,indent=2));t.replace(R/'RESOURCE_SUMMARY.json')
  active=subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True).stdout.strip()
  if active not in['active','activating','reloading']:break
  time.sleep(10)
summary['status']='finished';summary['controller_unit_state']=active;(R/'RESOURCE_SUMMARY.json').write_text(json.dumps(summary,indent=2))
