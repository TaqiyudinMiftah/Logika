"""Small destructive-to-owned-test-units-only checks for aggregate cgroup limits."""
from pathlib import Path
import json
import subprocess
import sys
import time
import shlex
from .common import write

def exercise(root,host):
    root=Path(root);records={}
    tests={
      'memory':"import subprocess,sys; ps=[subprocess.Popen([sys.executable,'-c','import time; x=bytearray(48*1024**2); time.sleep(3)']) for _ in range(2)]; [p.wait() for p in ps]",
      'cpu':"import subprocess,sys; code='import time; t=time.process_time(); exec(\"while time.process_time()-t<0.7: pass\")'; ps=[subprocess.Popen([sys.executable,'-c',code]) for _ in range(4)]; [p.wait() for p in ps]"}
    for name,code in tests.items():
        unit=f'metaexp-limitcheck-{host}-{name}-{int(time.time())}'
        prefix="import json;from pathlib import Path;rel=Path('/proc/self/cgroup').read_text().strip().split('::')[-1];cg=Path('/sys/fs/cgroup')/rel.lstrip('/');print(json.dumps({n:(cg/n).read_text().strip() for n in ['cpu.max','memory.max']}),flush=True);"
        code=prefix+code
        args=['systemd-run','--user','--wait','--pipe','--unit='+unit,'--property=RuntimeMaxSec=15',
              '--property=MemorySwapMax=0','--property=MemoryMax='+('64M' if name=='memory' else '256M'),
              '--property=CPUQuota=100%',sys.executable,'-c',code]
        start=time.time();p=subprocess.run(args,capture_output=True,text=True,timeout=25);elapsed=time.time()-start
        show=subprocess.run(['systemctl','--user','show',unit,'-p','Result','-p','MemoryPeak','-p','MemoryMax','-p','CPUQuotaPerSecUSec','-p','CPUUsageNSec'],capture_output=True,text=True,check=True)
        values=dict(line.split('=',1) for line in show.stdout.splitlines() if '=' in line)
        actual=json.loads(p.stdout.splitlines()[0]);quota,period=map(int,actual['cpu.max'].split())
        if name=='memory':
            if actual['memory.max']!=str(64*1024**2) or values.get('Result')!='oom-kill':raise RuntimeError('Aggregate memory OOM test did not enforce the limit: '+show.stdout+p.stderr)
        elif quota/period>1.00001 or elapsed<2.4:raise RuntimeError('Aggregate CPU quota test failed: '+show.stdout+p.stdout)
        records[name]={'elapsed_seconds':elapsed,'returncode':p.returncode,'effective_cgroup':actual,'properties':values,'stdout':p.stdout,'stderr':p.stderr}
        subprocess.run(['systemctl','--user','reset-failed',unit],capture_output=True)
    unit=f'metaexp-disconnect-{host}-{int(time.time())}';marker=root/'preflight'/f'{host}_disconnect_marker.json';marker.parent.mkdir(exist_ok=True)
    code="import time,json;from pathlib import Path;time.sleep(2);Path("+repr(str(marker))+").write_text(json.dumps({'survived_parent_disconnect':True}))"
    condition=sys.executable+' -c '+shlex.quote('import time,sys;sys.exit(0 if time.time()<'+repr(time.time()+60)+' else 1)')
    subprocess.run(['systemd-run','--user','--unit='+unit,'--property=MemoryMax=64M','--property=CPUQuota=100%','--property=OOMScoreAdjust=1000','--property=ExecCondition='+condition,sys.executable,'-c',code],capture_output=True,check=True)
    # The launching command has returned. A subsequent invocation reads the result.
    time.sleep(3)
    if not marker.exists():raise RuntimeError('Detached service did not produce its completion marker')
    records['detached_service']={'passed':True,'marker':str(marker)}
    write(root/'preflight'/f'{host}_resource_exercise.json',records)
    return records
