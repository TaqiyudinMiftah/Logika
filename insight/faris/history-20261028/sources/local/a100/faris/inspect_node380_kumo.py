"""Read and save node380 run status without interrupting the launcher."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from connection_profiles import activated, load_profile


def main():
    with activated(load_profile('node380')) as remote:
        client = remote.JupyterClient(remote.DEFAULT_BASE_URL, remote.load_token(Path(remote.DEFAULT_SYSINFO)))
        snippet = '''
from pathlib import Path
import hashlib,json,subprocess,time
root=Path('/home/taqiyudinmiftah/faris/kumo-large-node380-20261016')
state=json.loads((root/'artifacts/node380/status.json').read_text())
bootstrap=json.loads((root/'bootstrap.json').read_text())
stat=Path('/proc/%s/stat'%bootstrap['pid'])
state.update(remote_root=str(root),checked_epoch=time.time(),identity_matches=stat.exists() and stat.read_text().split()[21]==bootstrap['proc_start_ticks'])
for name,rel in [('environment','artifacts/node380/environment.json'),('pilot','artifacts/node380/pilot/summary.json'),('full','artifacts/node380/full/status.json'),('metrics','artifacts/node380/full/fold_metrics.json'),('summary','artifacts/node380/full/summary.json')]:
 p=root/rel
 if p.exists(): state[name]=json.loads(p.read_text())
if 'full' in state:
 pid=state['full'].get('worker_pid')
 if pid:
  p=Path('/proc/%s/status'%pid)
  if p.exists(): state['worker_memory']=[line for line in p.read_text().splitlines() if line.startswith(('VmRSS:','VmHWM:','State:'))]
state['gpu_processes']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=15).stdout.strip().splitlines()
manifest=json.loads((root/'data/kumo_v3/manifest.json').read_text())
state['feature_sha256']=manifest['feature_sha256']
for name in ['pilot','full']:
 log=root/('artifacts/node380/%s/fold_01/worker.log'%name)
 if log.exists(): state[name+'_worker_log_tail']=log.read_text()[-3000:]
state['completed_folds']=len(list((root/'artifacts/node380/full').glob('fold_*/complete.json')))
print(json.dumps(state))
'''
        code, output = client.run_python(client.session_kernel('term'), snippet, timeout=30)
        if code:
            raise RuntimeError(output)
        state = json.loads(output)
        (HERE / 'node380_last_status.json').write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps({key: state.get(key) for key in [
            'state', 'stage', 'pid', 'identity_matches', 'full', 'completed_folds',
            'worker_memory', 'gpu_processes', 'full_worker_log_tail']}, indent=2))


if __name__ == '__main__':
    main()
