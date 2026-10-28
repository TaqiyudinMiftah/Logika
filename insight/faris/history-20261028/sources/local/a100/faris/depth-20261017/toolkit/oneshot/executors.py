"""Process isolation and optional existing-Ray execution on a shared filesystem."""
import os, subprocess, sys, time
from pathlib import Path
import psutil
from .common import atomic, read, probe, validate_bundle, check_dependencies

def terminate_identity(pid, created, grace=2):
    try:
        p = psutil.Process(pid)
        if abs(p.create_time() - created) > .01: return False
        children = p.children(recursive=True); targets = children + [p]
        for target in targets:
            try: target.terminate()
            except psutil.NoSuchProcess: pass
        _, alive = psutil.wait_procs(targets, timeout=grace)
        for target in alive:
            try: target.kill()
            except psutil.NoSuchProcess: pass
        return True
    except psutil.NoSuchProcess: return False

class LocalExecutor:
    def __init__(self, root): self.root = Path(root); self.running = {}
    def submit(self, key, spec, directory):
        directory = Path(directory); env = os.environ.copy()
        for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[k] = str(spec['resources']['cpus'])
        if 'gpu_ids' in spec: env['CUDA_VISIBLE_DEVICES'] = ','.join(spec['gpu_ids'])
        elif not spec['resources'].get('gpus'): env['CUDA_VISIBLE_DEVICES'] = ''
        env.update({'PYTHONUNBUFFERED': '1', 'PYTHONPATH': str(self.root / 'runtime'), 'WANDB_MODE': 'disabled', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
        out = (directory / 'stdout.log').open('ab'); err = (directory / 'stderr.log').open('ab')
        cmd = [sys.executable, '-m', 'oneshot.worker', str(self.root), str(directory / 'spec.json'), str(directory)]
        try: p = subprocess.Popen(cmd, cwd=directory, env=env, stdout=out, stderr=err, start_new_session=True)
        finally: out.close(); err.close()
        created = psutil.Process(p.pid).create_time()
        self.running[key] = {'p': p, 'created': created, 'directory': directory, 'peak_ram_mb': 0, 'cpu_seconds_sampled': 0, 'peak_vram_mb': 0, 'gpu_last_sample': 0, 'started': time.time()}
        atomic(directory / 'process.json', {'pid': p.pid, 'created': created, 'command': cmd})
        return {'pid': p.pid, 'created': created, 'command': cmd}
    def poll(self, key):
        item = self.running[key]; p = item['p']; rc = p.poll()
        try:
            process = psutil.Process(p.pid); processes = [process] + process.children(recursive=True)
            ram = 0; cpu = 0
            for proc in processes:
                try:
                    ram += proc.memory_info().rss / 1048576; t = proc.cpu_times(); cpu += t.user + t.system
                except psutil.Error: pass
            item['peak_ram_mb'] = max(item['peak_ram_mb'], ram)
            item['cpu_seconds_sampled'] = max(item['cpu_seconds_sampled'], cpu)
            spec = read(item['directory'] / 'spec.json')
            if spec['resources'].get('gpus') and time.time() - item['gpu_last_sample'] > 1:
                item['gpu_last_sample'] = time.time()
                try:
                    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2)
                    pids = {pr.pid for pr in processes}; memory = 0
                    for row in out.stdout.splitlines():
                        pid, mb = row.split(',')
                        if int(pid.strip()) in pids: memory += float(mb.strip())
                    item['peak_vram_mb'] = max(item['peak_vram_mb'], memory)
                except (OSError, ValueError, subprocess.TimeoutExpired): pass
        except psutil.Error: pass
        return rc, {k: item[k] for k in ('peak_ram_mb', 'cpu_seconds_sampled', 'peak_vram_mb')}
    def cancel(self, key):
        item = self.running[key]; return terminate_identity(item['p'].pid, item['created'])
    def forget(self, key): self.running.pop(key, None)
    def close(self): pass

def ray_trial(root, key, spec, directory):
    # Ray assigns visible GPUs. Never overwrite that assignment with driver IDs.
    spec = dict(spec); spec.pop('gpu_ids', None)
    ex = LocalExecutor(root); identity = ex.submit(key, spec, directory)
    while True:
        rc, usage = ex.poll(key); atomic(Path(directory) / 'usage.json', usage)
        cancel = Path(directory) / 'cancel.json'
        if cancel.exists() and time.time() - cancel.stat().st_mtime > 2: ex.cancel(key)
        if rc is not None: return {'returncode': rc, 'usage': usage, 'identity': identity}
        time.sleep(.2)

def ray_node_probe(root):
    validate_bundle(root)
    check_dependencies(root)
    return probe(root)

class RayExecutor:
    def __init__(self, root, address):
        import ray
        self.ray = ray; self.root = Path(root); self.running = {}; self.directories = {}
        self.runtime_env = {'env_vars': {'PYTHONPATH': str(self.root / 'runtime')}}
        if not ray.is_initialized(): ray.init(address=address, ignore_reinit_error=True)
        self.remote = ray.remote(ray_trial)
        self.nodes = {}
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        self.affinity = NodeAffinitySchedulingStrategy
        for node in ray.nodes():
            if node['Alive']:
                ref = ray.remote(ray_node_probe).options(num_cpus=0, runtime_env=self.runtime_env, scheduling_strategy=self.affinity(node['NodeID'], soft=False)).remote(str(root))
                self.nodes[node['NodeID']] = ray.get(ref, timeout=30)
    def submit(self, key, spec, directory):
        r = spec['resources']; node = spec['node_id']
        ref = self.remote.options(num_cpus=r['cpus'], num_gpus=r.get('gpus', 0), memory=int(r['ram_mb'] * 1048576),
            runtime_env=self.runtime_env, scheduling_strategy=self.affinity(node, soft=False)).remote(str(self.root), key, spec, str(directory))
        self.running[key] = ref; self.directories[key] = Path(directory)
        return {'ray_task': ref.hex(), 'node_id': node}
    def poll(self, key):
        ready, _ = self.ray.wait([self.running[key]], timeout=0)
        if ready:
            try:
                result = self.ray.get(ready[0]); return result['returncode'], result['usage']
            except Exception: return 1, {}
        return None, {}
    def cancel(self, key):
        # Cooperative marker lets node-local supervisor kill the process tree first.
        if not (self.directories[key] / 'process.json').exists(): self.ray.cancel(self.running[key], force=False)
    def forget(self, key): self.running.pop(key, None)
    def close(self): self.ray.shutdown()
