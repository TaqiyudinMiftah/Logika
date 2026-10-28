"""Watch only our worker; stop it before shared memory pressure becomes critical."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];cg=Path('/sys/fs/cgroup')
def current():return int((cg/'memory.current').read_text())
def events():return dict(line.split() for line in (cg/'memory.events').read_text().splitlines())
before=events();maximum=int((cg/'memory.max').read_text());initial=current()
assert maximum-initial>2.5*2**30,'Less than 2.5 GiB initial host headroom'
assert float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<256
with (root/'pilot.log').open('x') as log:
    child=subprocess.Popen(sys.argv[1:],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    peak=initial;peak_rss=0;reason=None;start=time.time()
    while child.poll() is None:
        used=current();peak=max(peak,used)
        try:
            status=Path(f'/proc/{child.pid}/status').read_text()
            rss=int(next(l.split()[1] for l in status.splitlines() if l.startswith('VmRSS:')))*1024
        except (OSError,StopIteration):rss=0
        peak_rss=max(peak_rss,rss)
        if used>maximum-0.75*2**30:reason='shared_cgroup_headroom_below_0.75GiB'
        elif rss>2.25*2**30:reason='worker_RSS_above_2.25GiB'
        elif time.time()-start>600:reason='600_second_pilot_limit'
        if reason:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            break
        time.sleep(.2)
record={'returncode':child.wait(),'stop_reason':reason,'initial_cgroup_GiB':initial/2**30,
 'peak_cgroup_GiB':peak/2**30,'peak_worker_RSS_GiB':peak_rss/2**30,
 'events_before':before,'events_after':events(),'seconds':time.time()-start}
(root/'RESOURCE_PILOT.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
