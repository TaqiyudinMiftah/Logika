"""Finite capacity wait for the prepared replay, using no ML imports or extra kernel."""
from pathlib import Path
import datetime,json,os,subprocess,sys,time
root=Path(__file__).resolve().parents[1]
deadline=1793020580.849942
cg=Path('/sys/fs/cgroup');limit=int((cg/'memory.max').read_text())
stable_since=None
def state(stage,**kw):
    record={'stage':stage,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'deadline_epoch':deadline,**kw}
    (root/'CAPACITY_STATE.json').write_text(json.dumps(record,indent=2)+'\n')
while time.time()<deadline and not (root/'STOP').exists():
    used=int((cg/'memory.current').read_text());headroom=(limit-used)/2**30
    try:
        used_gpu=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=15).strip())
    except (ValueError,subprocess.SubprocessError):used_gpu=999999
    available=headroom>=3.0 and used_gpu<256
    stable_since=stable_since or time.time() if available else None
    state('waiting_for_capacity',RAM_headroom_GiB=headroom,GPU_used_MiB=used_gpu,
          required_RAM_headroom_GiB=3.0,required_stable_seconds=30)
    if stable_since is not None and time.time()-stable_since>=30:
        py='/home/fariselhakim/gipta/projects/logika-ui-26/.venv/bin/python'
        checkpoint='/home/fariselhakim/gipta/projects/logika-ui-26/artifacts/hf_cache/hub/models--nvidia--Kumo-Tabular/snapshots/3c3e10bbdb590ace29e7026847f92db3c603096d/small/classifier.pt'
        env=os.environ.copy();env['PYTHONPATH']=str(root/'dependencies')
        args=[py,str(root/'source/a100_guard.py'),py,str(root/'source/a100_pilot.py'),
              '--packet',str(root/'a100_packet'),'--checkpoint',checkpoint,'--out',str(root/'pilot_result')]
        state('guarded_pilot_running',RAM_headroom_GiB=headroom)
        with (root/'capacity_pilot_guard.log').open('xb') as log:
            rc=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=650).returncode
        state('pilot_complete' if rc==0 else 'pilot_failed',returncode=rc)
        sys.exit(rc)
    time.sleep(30)
state('stopped' if (root/'STOP').exists() else 'capacity_wait_expired')
