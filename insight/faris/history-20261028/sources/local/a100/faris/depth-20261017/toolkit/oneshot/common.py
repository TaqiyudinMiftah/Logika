"""Versioned campaign inputs, immutable snapshots, and host inspection."""
from __future__ import annotations
import hashlib, importlib.util, json, math, os, platform, shutil, subprocess, sys, time
from pathlib import Path

VERSION = 1
TERMINAL = {'complete', 'pruned', 'failed', 'interrupted', 'timeout', 'invalid'}

def read(path):
    return json.loads(Path(path).read_text())

def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def inside(root, name):
    root = Path(root).resolve(); p = (root / name).resolve()
    if p == root or not p.is_relative_to(root): raise ValueError(f'path escapes artifact root: {name}')
    return p

def files(root):
    root = Path(root)
    if root.is_file(): return {str(root.resolve()): sha(root)}
    if not root.is_dir(): raise ValueError(f'missing input: {root}')
    found = {str(p.resolve()): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    if not found: raise ValueError(f'empty input: {root}')
    return found

def size_mb(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file()) / 1048576

def positive(x, name):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
        raise ValueError(f'{name} must be positive and finite')

def validate_config(c):
    if c.get('version') != VERSION: raise ValueError('unsupported campaign version')
    for key in ('id', 'competition_root', 'adapter', 'metric', 'inputs', 'protocols', 'families', 'brief', 'evidence'):
        if not c.get(key): raise ValueError(f'missing {key}')
    if not c['id'].replace('-', '').replace('_', '').isalnum(): raise ValueError('invalid id')
    if c.get('direction') not in ('minimize', 'maximize'): raise ValueError('direction must be minimize or maximize')
    for k in ('budget_seconds', 'storage_limit_mb', 'max_trials', 'max_workers'): positive(c.get(k), k)
    for k in ('max_trials', 'max_workers'):
        if type(c[k]) is not int: raise ValueError(f'{k} must be integer')
    if not 0 < c.get('reserve_fraction', .15) < .5: raise ValueError('invalid verification reserve')
    if not 0 < c.get('breadth_fraction', .4) < 1 - c.get('reserve_fraction', .15): raise ValueError('invalid breadth fraction')
    if not c.get('seeds') or any(type(s) is not int for s in c['seeds']) or len(set(c['seeds'])) != len(c['seeds']):
        raise ValueError('seeds must be distinct integers')
    if 'screen' not in c['protocols']: raise ValueError('screen protocol required')
    for name, protocol in c['protocols'].items():
        if not protocol.get('split'): raise ValueError(f'{name}: frozen split required')
    for key in ('description', 'rules', 'compute_access', 'external_data_policy', 'pretrained_policy'):
        if not c['brief'].get(key): raise ValueError(f'brief missing {key}')
    for key in ('research', 'eda', 'hypotheses', 'applicability'):
        if not c['evidence'].get(key): raise ValueError(f'evidence missing {key}')
    names = set()
    for family in c['families']:
        name = family.get('name')
        if not name or name in names: raise ValueError('family names must be unique')
        names.add(name)
        if not family.get('hypothesis'): raise ValueError(f'{name}: hypothesis required')
        r = family.get('resources', {})
        for key in ('cpus', 'ram_mb', 'disk_mb', 'timeout_seconds'): positive(r.get(key), f'{name}.{key}')
        if type(r['cpus']) is not int or (type(r.get('gpus', 0)) is not int or r.get('gpus', 0) < 0): raise ValueError('positive integer CPUs and nonnegative integer GPUs required')
        if r.get('gpus') and r.get('vram_mb', 0) <= 0: raise ValueError('GPU jobs require vram_mb')
        previous = set()
        for key, spec in family.get('parameters', {}).items():
            if spec.get('type') not in ('categorical', 'int', 'float'): raise ValueError(f'{key}: invalid parameter type')
            if not set(spec.get('when', {})).issubset(previous): raise ValueError('conditions must reference earlier parameters')
            if spec['type'] == 'categorical':
                vals = spec.get('choices', [])
                if not vals or len({digest(v) for v in vals}) != len(vals): raise ValueError('choices must be nonempty and unique')
                if any(type(v) not in (str, int, float, bool, type(None)) for v in vals): raise ValueError('choices must be scalar')
            else:
                lo, hi = spec.get('low'), spec.get('high')
                if not isinstance(lo, (float, int)) or not isinstance(hi, (float, int)) or not math.isfinite(lo) or not math.isfinite(hi) or lo > hi: raise ValueError('invalid parameter bounds')
                if spec.get('log') and lo <= 0: raise ValueError('log bounds must be positive')
                if spec['type'] == 'int' and (int(lo) != lo or int(hi) != hi): raise ValueError('integer bounds required')
            previous.add(key)
        fidelity = family.get('fidelity')
        if fidelity:
            for key in ('max_steps', 'min_steps', 'reduction_factor'): positive(fidelity.get(key), key)
            if any(type(fidelity[k]) is not int for k in ('max_steps', 'min_steps', 'reduction_factor')): raise ValueError('fidelity steps and reduction factor must be integers')
            if fidelity['min_steps'] > fidelity['max_steps'] or fidelity['reduction_factor'] < 2: raise ValueError('invalid fidelity rungs')
    if c.get('executor', 'local') not in ('local', 'ray'): raise ValueError('unsupported executor')
    if c.get('executor') == 'ray' and not c.get('ray_address'): raise ValueError('existing Ray address required')
    if c.get('max_retries', 1) not in range(0, 6): raise ValueError('max_retries must be 0..5')
    if c.get('executor') == 'ray' and c.get('ray_shared_filesystem_verified') is not True:
        raise ValueError('Ray requires verified shared campaign paths')
    if not isinstance(c.get('coverage_pool_limit', 10000), int) or c.get('coverage_pool_limit', 10000) < 1: raise ValueError('invalid coverage pool limit')
    from .space import pool
    for family in c['families']:
        rows, _ = pool(family, min(c.get('coverage_pool_limit', 10000), 10000), c.get('search_seed', 42))
        if not rows: raise ValueError(f"empty admissible space: {family['name']}")
    return c

def check_dependencies(root):
    import importlib.metadata
    root = Path(root); seal = read(root / 'seal.json')
    lock = root / 'snapshots' / seal['snapshot'] / 'code' / 'requirements.lock'
    for line in lock.read_text().splitlines():
        if not line.strip() or line.startswith('#'): continue
        package, expected = line.strip().split('==', 1)
        try: actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: raise ValueError(f'missing pinned dependency: {package}')
        if actual != expected: raise ValueError(f'dependency drift: {package}: expected {expected}, got {actual}')

def probe(path='.'):
    import psutil
    v = psutil.virtual_memory(); process = psutil.Process()
    cpus = len(process.cpu_affinity()) if hasattr(process, 'cpu_affinity') else (os.cpu_count() or 1)
    limits = {}
    # Resolve this process's nested cgroup as well as the mount root.
    roots = [Path('/sys/fs/cgroup')]
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            if line.startswith('0::'):
                p = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
                roots += [p, *[a for a in p.parents if a.is_relative_to('/sys/fs/cgroup')]]
    except OSError: pass
    ram = v.available
    for root in set(roots):
        try:
            quota, period = (root / 'cpu.max').read_text().split()
            if quota != 'max': cpus = min(cpus, max(1, math.floor(int(quota) / int(period))))
        except (OSError, ValueError): pass
        try:
            maximum = (root / 'memory.max').read_text().strip()
            current = int((root / 'memory.current').read_text())
            if maximum != 'max':
                ram = min(ram, max(0, int(maximum) - current)); limits[str(root)] = {'max': int(maximum), 'current': current}
        except (OSError, ValueError): pass
    gpus = []
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            for row in out.stdout.splitlines():
                idx, uid, total, free, util = [v.strip() for v in row.split(',')]
                gpus.append({'id': idx, 'uuid': uid, 'total_mb': float(total), 'free_mb': float(free), 'utilization': float(util)})
    except (OSError, subprocess.TimeoutExpired, ValueError): pass
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        permitted = visible.split(',')
        gpus = [g for g in gpus if g['id'] in permitted or g['uuid'] in permitted]
    return {'time': time.time(), 'host': platform.node(), 'python': sys.version, 'cpus': cpus, 'ram_available_mb': ram / 1048576,
            'host_ram_mb': v.total / 1048576, 'cgroup_limits': limits, 'disk_free_mb': shutil.disk_usage(path).free / 1048576,
            'gpus': gpus, 'load_average': list(os.getloadavg()) if hasattr(os, 'getloadavg') else None}

def load_adapter(campaign):
    seal = read(Path(campaign) / 'seal.json'); code = Path(campaign) / 'snapshots' / seal['snapshot'] / 'code'
    c = read(Path(campaign) / 'config.json'); path = inside(code, c['adapter'])
    sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location('campaign_task', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for name in ('run_trial', 'evaluate'):
        if not callable(getattr(module, name, None)): raise ValueError(f'adapter lacks {name}')
    return module

def bundle(config, source, output):
    c = validate_config(read(config)); source = Path(source).resolve(); output = Path(output).resolve()
    root = Path(c['competition_root']).resolve(strict=True)
    if not output.is_relative_to(root) or output == root: raise ValueError('campaign must be inside competition_root')
    if output.exists(): raise ValueError('output already exists; use validate/resume')
    inside(source, c['adapter']).resolve(strict=True)
    source_files = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts}
    if not source_files: raise ValueError('empty source directory')
    if any(p.is_symlink() for p in source.rglob('*')): raise ValueError('source symlinks are not permitted in portable snapshots')
    inputs = {}
    for path in c['inputs'].values(): inputs.update(files(Path(path).resolve(strict=True)))
    evidence = {**c['evidence'], 'rules': c['brief']['rules']}
    evidence.update({f'split_{key}': val['split'] for key, val in c['protocols'].items()})
    for name, path in evidence.items():
        if not Path(path).is_file() or not Path(path).stat().st_size: raise ValueError(f'nonempty evidence required: {name}')
    lock = source / 'requirements.lock'
    if not lock.is_file() or not lock.stat().st_size: raise ValueError('source requires requirements.lock with exact dependency pins')
    for line in lock.read_text().splitlines():
        if line.strip() and not line.startswith('#') and '==' not in line: raise ValueError('requirements.lock must contain exact pins')
    output.mkdir(parents=True)
    runtime = output / 'runtime'
    shutil.copytree(Path(__file__).resolve().parents[1], runtime, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    snap_id = digest(source_files); snapshot = output / 'snapshots' / snap_id
    for name in source_files:
        dest = inside(snapshot / 'code', name); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source / name, dest)
    evidence_hashes = {}
    for name, path in evidence.items():
        dest = snapshot / 'evidence' / (name + Path(path).suffix); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
        evidence_hashes[str(dest.relative_to(output))] = sha(dest)
        if name.startswith('split_'): c['protocols'][name[6:]]['split'] = str(dest)
    atomic(output / 'config.json', c)
    runtime_hashes = {str(p.relative_to(output)): sha(p) for p in runtime.rglob('*') if p.is_file()}
    atomic(output / 'seal.json', {'version': VERSION, 'config_sha': sha(output / 'config.json'), 'snapshot': snap_id,
        'code': source_files, 'inputs': inputs, 'evidence': evidence_hashes, 'runtime': runtime_hashes})
    validate_bundle(output)
    return output

def validate_bundle(root, check_inputs=True):
    root = Path(root); c = validate_config(read(root / 'config.json')); seal = read(root / 'seal.json')
    if sha(root / 'config.json') != seal['config_sha']: raise ValueError('campaign config changed')
    code = root / 'snapshots' / seal['snapshot'] / 'code'
    actual_files = {str(p.relative_to(code)) for p in code.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if actual_files != set(seal['code']): raise ValueError('source file set changed')
    compile(inside(code, c['adapter']).read_text(), c['adapter'], 'exec')
    checks = dict(seal['runtime']); checks.update(seal['evidence'])
    checks.update({f"snapshots/{seal['snapshot']}/code/{p}": h for p, h in seal['code'].items()})
    for name, expected in checks.items():
        if sha(inside(root, name)) != expected: raise ValueError(f'changed frozen artifact: {name}')
    if check_inputs:
        actual = {}
        for path in c['inputs'].values(): actual.update(files(path))
        if actual != seal['inputs']: raise ValueError('input files changed')
    return c
