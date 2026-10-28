"""Persistent local queues and bounded process execution; no model decisions here."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import psutil
from .common import read,write,event,GIB,sha,identity,inside,code_hash

def terminate(process):
    if process.poll() is not None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()

def local_cgroup():
    rel=Path('/proc/self/cgroup').read_text().strip().split('::')[-1]
    cg=Path('/sys/fs/cgroup')/rel.lstrip('/')
    out={'path':str(cg)}
    for n in ['memory.max','memory.current','memory.peak','memory.swap.max','memory.events','cpu.max']:
        if (cg/n).exists():out[n]=(cg/n).read_text().strip()
    return out

def enqueue(root,tasks):
    root=Path(root);(root/'inbox').mkdir(exist_ok=True)
    catalogue=read(root/'catalogue.json');known={r['id']:r for r in catalogue['recipes']}
    allowed={'cv','reference','reference_confirmation','nested','repeat','temporal','confirmation','ensemble_confirmation'}
    count=0
    for task in tasks:
        if task.get('mode','cv') not in allowed:raise ValueError('Unsupported task mode')
        if not isinstance(task['id'],str) or not all(c.isalnum() or c=='_' for c in task['id']):raise ValueError('Invalid task ID')
        if task.get('mode','cv') in ['cv','repeat','temporal','confirmation']:
            if task['recipe']['id'] not in known or task['recipe']!=known[task['recipe']['id']]:raise ValueError('Unregistered pipeline recipe')
        path=inside(root,'inbox/'+task['id']+'.json')
        if path.exists():
            if read(path)!=task:raise ValueError('Changed queued task')
        else:write(path,task);count+=1
    return count

def daemon(root,host):
    root=Path(root).resolve();cfg=read(root/'campaign.json');machine=cfg['machines'][host]
    if code_hash(root)!=cfg['code_hash'] and cfg.get('id')!='synthetic_runtime_test':raise ValueError('Worker source differs from the frozen campaign')
    lock=(root/'daemon.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={};done=set();stopping=False;campaign_size=0;last_disk_scan=0
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for p in (root/'runs').glob('*/status.json'):
        if read(p).get('state') in ['completed','failed','partial','timed_out','resource_limited']:done.add(p.parent.name)
    # Orphaned running tasks from a process-group shutdown resume their saved folds.
    write(root/'daemon_identity.json',{'pid':os.getpid(),'create_time':psutil.Process().create_time(),'host':host,'cgroup':local_cgroup()})
    while not stopping and time.time()<cfg['deadline']:
        now=time.time();total=0;gpu_active=False;gpu_memory={}
        if any(v['gpu'] for v in active.values()) and machine.get('gpu_uuid') and int(now)%5==0:
            try:
                query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
                for line in query.stdout.splitlines():
                    pid,memory=line.split(',');gpu_memory[int(pid)]=float(memory.strip())*1024**2
            except Exception as e:event(root/'runtime.jsonl',event='gpu_telemetry_unavailable',error=str(e))
        for tid,item in list(active.items()):
            p=item['process'];folder=root/'runs'/tid
            try:
                proc=psutil.Process(p.pid);owned=[proc]+proc.children(recursive=True)
                rss=sum(v.memory_info().rss for v in owned if v.is_running())
                cpu_seconds=sum(v.cpu_times().user+v.cpu_times().system for v in owned if v.is_running())
                gpu_bytes=sum(gpu_memory.get(v.pid,0) for v in owned)
            except (psutil.NoSuchProcess,psutil.AccessDenied):rss=0;cpu_seconds=0;gpu_bytes=0
            total+=rss;item['peak']=max(item['peak'],rss);gpu_active|=item['gpu']
            if now-item['last_sample']>=2:
                event(folder/'resources.jsonl',pid=p.pid,rss_bytes=rss,cpu_seconds=cpu_seconds,gpu_bytes=gpu_bytes if gpu_memory else None,elapsed_seconds=now-item['started'])
                item['last_sample']=now
            if p.poll() is None and now>item['until']:
                terminate(p);write(folder/'status.json',{'state':'timed_out','error':'Recipe or phase deadline reached','elapsed_seconds':now-item['started']})
            if p.poll() is not None:
                item['log'].close();status=read(folder/'status.json') if (folder/'status.json').exists() else {'state':'failed','error':'Child exited without status'}
                if status['state']=='running':status={'state':'partial','error':'Worker interrupted','returncode':p.returncode};write(folder/'status.json',status)
                write(folder/'resources.json',{'peak_rss_bytes':item['peak'],'returncode':p.returncode,'elapsed_seconds':now-item['started']})
                done.add(tid);del active[tid]
                event(root/'runtime.jsonl',event='finished',task=tid,state=status['state'])
        # RSS admission stays below the hard service limit and includes a reserve.
        host_available=psutil.virtual_memory().available
        if (total>machine['admission_gib']*GIB or host_available<2*GIB) and active:
            tid=max(active,key=lambda t:active[t]['peak']);item=active[tid];terminate(item['process'])
            write(root/'runs'/tid/'status.json',{'state':'resource_limited','error':'Campaign memory admission or host OS reserve reached'})
        pending=sorted((root/'inbox').glob('*.json'),key=lambda p:p.stat().st_mtime)
        free_disk=psutil.disk_usage(root).free
        if now-last_disk_scan>=30:
            campaign_size=sum(p.stat().st_size for p in (root/'runs').rglob('*') if p.is_file());last_disk_scan=now
        disk_ok=free_disk>2*GIB and campaign_size<machine.get('run_disk_gib',25)*GIB and host_available>4*GIB
        if disk_ok:
            for path in pending:
                if len(active)>=machine['workers'] or total+2*GIB>machine['admission_gib']*GIB:break
                tid=path.stem
                if tid in done or tid in active:continue
                task=read(path);until=min(task['until'],cfg['deadline'])
                if until-now<max(20,task.get('expected_seconds',20)):
                    (root/'runs'/tid).mkdir(parents=True,exist_ok=True)
                    write(root/'runs'/tid/'status.json',{'state':'partial','error':'No time remaining before start'});done.add(tid);continue
                gpu=task.get('device') in ['cuda','opencl']
                if gpu and gpu_active:continue
                folder=root/'runs'/tid;folder.mkdir(parents=True,exist_ok=True)
                log=(folder/'worker.log').open('ab');env=dict(os.environ)
                for directory in ['.cache','.tmp']:(root/directory).mkdir(exist_ok=True)
                env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg',XDG_CACHE_HOME=str(root/'.cache'),TMPDIR=str(root/'.tmp'),CUDA_CACHE_PATH=str(root/'.cache/cuda'))
                env['CUDA_VISIBLE_DEVICES']=machine.get('gpu_uuid','') if task.get('device')=='cuda' else ''
                p=subprocess.Popen([str(root/'.venv/bin/python'),'-u',str(root/'research_worker.py'),str(root),str(path)],cwd=root,
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
                active[tid]={'process':p,'log':log,'started':now,'until':until,'peak':0,'last_sample':0,'gpu':gpu}
                total+=2*GIB;gpu_active|=gpu;event(root/'runtime.jsonl',event='started',task=tid,pid=p.pid,device=task.get('device'))
        write(root/'heartbeat.json',{'time':time.time(),'host':host,'pid':os.getpid(),'active':list(active),'done':len(done),
                                    'rss_bytes':total,'free_disk_bytes':free_disk,'deadline':cfg['deadline'],'cgroup':local_cgroup()})
        time.sleep(1)
    for tid,item in active.items():
        terminate(item['process']);item['log'].close()
        folder=root/'runs'/tid
        if (folder/'complete.json').exists():
            write(folder/'status.json',{'state':'completed','finished_during_shutdown':True})
        else:write(folder/'status.json',{'state':'partial','error':'Campaign stopped or hard deadline reached'})
        write(folder/'resources.json',{'peak_rss_bytes':item['peak'],'returncode':item['process'].poll(),'elapsed_seconds':time.time()-item['started']})
    write(root/'daemon_stopped.json',{'time':time.time(),'deadline':cfg['deadline']})
    lock.close()

def start_daemon(root,host):
    root=Path(root).resolve();cfg=read(root/'campaign.json');machine=cfg['machines'][host]
    record=root/'daemon_identity.json'
    if record.exists():
        old=read(record)
        try:
            if abs(psutil.Process(old['pid']).create_time()-old['create_time'])<.01:return {'already_running':True,**old}
        except psutil.NoSuchProcess:pass
    if time.time()>=cfg['deadline']:raise ValueError('Campaign deadline expired')
    if host=='pc2':
        with (root/'daemon.log').open('ab') as log:
            p=subprocess.Popen([str(root/'.venv/bin/python'),'-u','-m','metaexp','daemon','--root',str(root),'--host',host],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        return {'pid':p.pid}
    unit='metaexp-'+cfg['id']+'-'+host
    state=subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True)
    if state.stdout.strip()=='active':return {'unit':unit,'already_running':True}
    subprocess.run(['systemctl','--user','reset-failed',unit],capture_output=True)
    import shlex
    condition=str(root/'.venv/bin/python')+' -c '+shlex.quote('import time,sys;sys.exit(0 if time.time()<'+repr(cfg['deadline'])+' else 1)')
    args=['systemd-run','--user','--unit='+unit,'--property=CPUQuota='+str(machine['cpu_quota'])+'%',
          '--property=OOMScoreAdjust=1000',
          '--property=MemoryMax='+str(machine['memory_gib'])+'G','--property=MemorySwapMax=0','--property=Restart=on-failure',
          '--property=RestartSec=5','--property=KillMode=control-group','--property=WorkingDirectory='+str(root),
          '--property=ExecCondition='+condition,
          '--property=RuntimeMaxSec='+str(int(cfg['deadline']-time.time()+5)),str(root/'.venv/bin/python'),'-u','-m','metaexp','daemon','--root',str(root),'--host',host]
    p=subprocess.run(args,capture_output=True,text=True,timeout=20)
    if p.returncode:raise RuntimeError(p.stderr)
    return {'unit':unit,'message':p.stderr}

def status(root):
    root=Path(root);runs={}
    for p in (root/'runs').glob('*/status.json'):
        s=read(p)
        if s.get('state')!='running':runs[p.parent.name]={'state':s['state'],'error':s.get('error')}
    return {'heartbeat':read(root/'heartbeat.json') if (root/'heartbeat.json').exists() else None,'runs':runs,
            'stopped':(root/'daemon_stopped.json').exists()}

def stop_daemon(root):
    root=Path(root);p=root/'daemon_identity.json'
    if p.exists():
        d=read(p)
        try:
            proc=psutil.Process(d['pid'])
            if abs(proc.create_time()-d['create_time'])<.01:proc.send_signal(signal.SIGTERM)
        except psutil.NoSuchProcess:pass

def rpc(root):
    """Forced SSH command: JSON requests only, no shell or arbitrary filesystem access."""
    root=Path(root).resolve();request=json.load(sys.stdin);action=request['action']
    if action=='status':answer=status(root)
    elif action=='push':answer={'accepted':enqueue(root,request['tasks'])}
    elif action=='start':answer=start_daemon(root,request['host'])
    elif action=='configure':
        cfg=request['campaign']
        if cfg['id']!=root.name or cfg['deadline']-cfg['started']>43200.1:raise ValueError('Invalid campaign configuration')
        if (root/'campaign.json').exists() and read(root/'campaign.json')!=cfg:raise ValueError('Immutable campaign configuration changed')
        write(root/'campaign.json',cfg);answer={'configured':True}
    elif action=='freeze':
        if (root/'finalists.json').exists() and read(root/'finalists.json')!=request['finalists']:raise ValueError('Finalists are already frozen')
        write(root/'finalists.json',request['finalists']);answer={'frozen':True}
    elif action=='bank_export':
        import tarfile,gzip
        with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1) as compressed:
            with tarfile.open(fileobj=compressed,mode='w|') as tar:
                for p in sorted((root/'bank').iterdir()):
                    if p.is_file() and not p.is_symlink():tar.add(p,arcname='bank/'+p.name,recursive=False)
        return
    elif action=='pack':
        import tarfile
        tid=request['id']
        if not all(c.isalnum() or c=='_' for c in tid):raise ValueError('Unsafe task ID')
        folder=inside(root,'runs/'+tid)
        with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
            for p in sorted(folder.rglob('*')):
                if p.is_file() and not p.is_symlink():tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
        return
    elif action=='stop':stop_daemon(root);answer={'stopping':True}
    elif action=='hold':
        cfg=read(root/'campaign.json');print(json.dumps({'holding':True}),flush=True)
        while time.time()<cfg['deadline']+1800:time.sleep(30)
        return
    elif action=='cleanup_key':
        tag='metaexp-'+root.name;p=Path.home()/'.ssh/authorized_keys'
        if p.exists():
            text=p.read_text();p.write_text('\n'.join(line for line in text.splitlines() if not line.endswith(' '+tag))+'\n');p.chmod(0o600)
        answer={'removed_campaign_key':True}
    else:raise ValueError('Unsupported RPC action')
    print(json.dumps(answer),flush=True)
