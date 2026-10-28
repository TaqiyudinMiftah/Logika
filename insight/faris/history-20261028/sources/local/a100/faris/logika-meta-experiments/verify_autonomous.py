"""Verify local queue progress while pausing only this campaign's collector."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import time

from metaexp.common import write
from metaexp.deployment import machine_config, remote


def execute(host, source):
    root = machine_config()['machines'][host]['root']
    command = 'cd ' + shlex.quote(root) + ' && .venv/bin/python -c ' + shlex.quote(source)
    return json.loads(remote(host, command, capture_output=True, text=True, timeout=45).stdout)


def snapshot(host):
    return execute(host, '''
import json,time,psutil
from pathlib import Path
from metaexp.common import read,sha,code_hash
r=Path.cwd();cfg=read(r/'campaign.json');h=read(r/'heartbeat.json');s=read(r/'local_catalogue_shard.json')
starts=[]
for line in (r/'runtime.jsonl').read_text().splitlines():
 try:
  event=json.loads(line)
  if event.get('event')=='started':starts.append(event)
 except json.JSONDecodeError:pass
active=[read(r/'inbox'/(tid+'.json')) for tid in h['active']]
proc=psutil.Process(h['pid']);rel=Path('/proc')/str(proc.pid)/'cgroup'
cg=Path('/sys/fs/cgroup')/rel.read_text().strip().split('::')[-1].lstrip('/')
limits={k:(cg/k).read_text().strip() for k in ['cpu.max','memory.max','memory.swap.max','memory.current','memory.peak','memory.events']}
print(json.dumps({'observed':time.time(),'heartbeat':h,'starts':starts,'shard_tasks':len(s['tasks']),
 'shard_sha256':sha(r/'local_catalogue_shard.json'),'shard_clock':[s['started'],s['deadline']],
 'active_devices':{d:sum(t.get('device','cpu')==d for t in active) for d in ['cpu','cuda','opencl']},
 'limits':limits,'executor_command':proc.cmdline(),'training_code_hash':code_hash(r),
 'configuration_sha256':sha(r/'campaign.json'),'started':cfg['started'],'deadline':cfg['deadline']}))
''')


def snapshots():
    with ThreadPoolExecutor(max_workers=3) as pool:
        return dict(zip(['pc1', 'pc2', 'pc3'], pool.map(snapshot, ['pc1', 'pc2', 'pc3'])))


def signal_collector(action, expected=None):
    return execute('pc2', '''
import os,signal,subprocess,json,time
from pathlib import Path
r=Path.cwd();unit='metaexp-meta-20261016-v1-coordinator'
pid=int(subprocess.check_output(['systemctl','--user','show',unit,'--property=MainPID','--value'],text=True))
cmd=Path('/proc',str(pid),'cmdline').read_bytes().decode().split(chr(0))
expected=''' + repr(expected) + '''
if pid<=1 or str(r/'scheduler_parallel.py') not in cmd or (expected is not None and pid!=expected):raise ValueError('Collector ownership guard failed')
action=''' + repr(action) + '''
if action=='pause':os.kill(pid,signal.SIGSTOP)
elif action=='resume':os.kill(pid,signal.SIGCONT)
else:raise ValueError(action)
print(json.dumps({'pid':pid,'action':action,'command':cmd,'time':time.time()}))
''')


def main():
    root = Path(__file__).resolve().parent
    before = snapshots()
    pause = signal_collector('pause')
    try:
        time.sleep(35)
        after = snapshots()
    finally:
        resumed = signal_collector('resume', pause['pid'])
    hosts = {}
    for host in before:
        first, last = before[host], after[host]
        new = [e for e in last['starts'] if e['time'] >= pause['time']]
        hosts[host] = {'before': {k:v for k,v in first.items() if k != 'starts'},
                       'after': {k:v for k,v in last.items() if k != 'starts'},
                       'new_job_starts': new,
                       'completed_delta': last['heartbeat']['done'] - first['heartbeat']['done']}
        assert len(last['heartbeat']['active']) <= machine_config()['machines'][host]['workers']
        assert last['active_devices']['cuda'] + last['active_devices']['opencl'] <= 1
        assert first['shard_sha256'] == last['shard_sha256']
        assert first['configuration_sha256'] == last['configuration_sha256']
        assert last['configuration_sha256'] == '8a848f72f51dc546431d22ea2ddcf70e699499675622f2ea81675b2a354d7695'
        assert last['shard_clock'] == [last['started'], last['deadline']]
        assert first['deadline'] == last['deadline'] == 1792189779.8887079
        assert first['training_code_hash'] == last['training_code_hash'] == '520f9b40d751d5bc83243f2ddf6c93a976cc249e5e7faeecf199cb2d05fd4808'
        assert last['heartbeat']['time'] > first['heartbeat']['time']
        machine = machine_config()['machines'][host]
        assert last['limits']['cpu.max'] == str(machine['cpu_quota']*1000)+' 100000'
        assert int(last['limits']['memory.max']) == machine['memory_gib']*1024**3
        assert last['limits']['memory.swap.max'] == '0'
    proof = {'observed': time.time(), 'collector_paused':pause, 'collector_resumed':resumed,
             'pause_seconds':35, 'hosts':hosts}
    write(root/'autonomous_execution_verification.json', proof)
    print(json.dumps({h: {'new_job_starts':len(v['new_job_starts']), 'completed_delta':v['completed_delta'],
                         'active':len(v['after']['heartbeat']['active']), 'shard_tasks':v['after']['shard_tasks']}
                      for h,v in hosts.items()}))


if __name__ == '__main__':
    main()
