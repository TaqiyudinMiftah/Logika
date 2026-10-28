#!/usr/bin/env python3
"""Self-contained campaign CLI; bundle this directory with each campaign."""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
from oneshot.common import atomic, read, bundle, validate_bundle, probe, inside, check_dependencies

def emit(value): print(json.dumps(value, indent=2, allow_nan=False))

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('bundle'); b.add_argument('--config', required=True); b.add_argument('--source', required=True); b.add_argument('--output', required=True)
    pr = sub.add_parser('probe'); pr.add_argument('path', nargs='?', default='.')
    for cmd in ('validate', 'run', 'resume', 'status', 'stop', 'report'):
        s = sub.add_parser(cmd); s.add_argument('campaign')
        if cmd in ('run', 'resume'): s.add_argument('--detach', action='store_true')
    rep = sub.add_parser('replay'); rep.add_argument('campaign'); rep.add_argument('job'); rep.add_argument('--output', required=True)
    verify = sub.add_parser('verify'); verify.add_argument('campaign'); verify.add_argument('job')
    a = p.parse_args(argv)
    if a.command in ('run','resume','status','stop','report'):
        import horizontal_runtime
        return horizontal_runtime.main()

    if a.command == 'probe': emit(probe(a.path)); return 0
    if a.command == 'bundle': emit({'campaign': str(bundle(a.config, a.source, a.output))}); return 0
    root = Path(a.campaign).resolve()
    if a.command == 'validate':
        c = validate_bundle(root); check_dependencies(root); emit({'valid': True, 'id': c['id'], 'note': 'Structural/hash validation; adapter execution is separately smoke-tested.'}); return 0
    if a.command == 'status':
        state = read(root / 'state.json') if (root / 'state.json').exists() else {'status': 'prepared'}
        result = {k: v for k, v in state.items() if k != 'jobs'}
        counts = {}
        for job in state.get('jobs', {}).values(): counts[job['status']] = counts.get(job['status'], 0) + 1
        import psutil
        alive = False
        if (root / 'owner.json').exists():
            owner = read(root / 'owner.json')
            try:
                process = psutil.Process(owner['pid'])
                alive = abs(process.create_time() - owner['created']) < .01 and process.status() != psutil.STATUS_ZOMBIE
            except psutil.Error: pass
        result['controller_alive'] = alive
        if state.get('status') == 'running' and not alive: result['status'] = 'controller_lost'
        result['job_counts'] = counts; result['heartbeat_age_seconds'] = time.time() - (root / 'heartbeat.json').stat().st_mtime if (root / 'heartbeat.json').exists() else None
        emit(result); return 0
    if a.command == 'stop':
        if not (root / 'state.json').exists(): raise ValueError('campaign has not started')
        atomic(root / 'stop.json', {'requested_at': time.time()}); emit({'stop_requested': True}); return 0
    if a.command == 'report':
        from oneshot.reporting import report
        emit(report(root)); return 0
    if a.command in ('replay', 'verify'):
        c = validate_bundle(root); check_dependencies(root); state = read(root / 'state.json'); job = state['jobs'][a.job]; old = inside(root, job['directory'])
        if a.command == 'verify':
            # Verification also refuses altered validation/model artifacts.
            from oneshot.reporting import checked_records
            records = [r for r in checked_records(root) if r['job_id'] == a.job]
            if not records or records[-1]['status'] != 'complete': raise ValueError('trial is not verified complete')
            env = os.environ.copy(); env['PYTHONPATH'] = str(root / 'runtime')
            rc = subprocess.call([sys.executable, '-m', 'oneshot.worker', str(root), str(old / 'spec.json'), str(old), '--verify'], env=env)
            if rc: return rc
            value = read(old / 'reevaluation.json')
            if abs(value['score'] - records[-1]['score']) > 1e-9 * max(1, abs(value['score'])): raise ValueError('recomputed metric differs')
            emit({'verified': True, 'evaluation': value}); return 0
        output = Path(a.output).resolve()
        if output.exists() or output.is_relative_to(root): raise ValueError('replay needs a new directory outside original campaign')
        spec = read(old / 'spec.json'); host = probe(output.parent); resources = spec['resources']
        if resources['cpus'] > host['cpus'] or resources['ram_mb'] > host['ram_available_mb'] * .85 or resources['disk_mb'] + c.get('disk_headroom_mb', 64) > host['disk_free_mb']:
            raise ValueError('insufficient current replay resources')
        if resources.get('gpus'):
            available = [g for g in host['gpus'] if g['free_mb'] * .85 >= resources['vram_mb']]
            if len(available) < resources['gpus']: raise ValueError('insufficient replay GPUs')
            spec['gpu_ids'] = [g['id'] for g in available[:resources['gpus']]]
        output.mkdir(parents=True); spec['deadline'] = time.time() + spec['resources']['timeout_seconds']; spec.pop('resume_from', None)
        spec['replay'] = True; atomic(output / 'spec.json', spec)
        from oneshot.executors import LocalExecutor
        ex = LocalExecutor(root); ex.submit('replay', spec, output)
        while True:
            rc, usage = ex.poll('replay')
            if time.time() >= spec['deadline']: ex.cancel('replay')
            if rc is not None: emit({'exit_code': rc, 'output': str(output), 'usage': usage}); return rc
            time.sleep(.1)
    if a.command in ('run', 'resume'):
        if a.detach:
            log = (root / 'controller.log').open('ab')
            try:
                proc = subprocess.Popen([sys.executable, str(root / 'runtime' / 'campaign.py'), a.command, str(root)], stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
            finally: log.close()
            for _ in range(150):
                if (root / 'state.json').exists():
                    s = read(root / 'state.json')
                    if s.get('ever_launched') and any((root / j['directory'] / 'started.json').exists() for j in s.get('jobs', {}).values() if j.get('directory')) and (root / 'owner.json').exists() and read(root / 'owner.json')['pid'] == proc.pid:
                        emit({'pid': proc.pid, 'campaign': str(root), 'status': s['status'], 'initial_worker_activity_verified': True, 'deadline': s['deadline']}); return 0
                if proc.poll() is not None: raise RuntimeError(f'controller exited {proc.returncode}; inspect {root / "controller.log"}')
                time.sleep(.1)
            emit({'pid': proc.pid, 'campaign': str(root), 'initial_worker_activity_verified': False, 'note': 'Process launched; worker startup not yet verified.'}); return 2
        from oneshot.controller import Controller
        state = Controller(root, resume=a.command == 'resume').run(); emit({k: v for k, v in state.items() if k != 'jobs'}); return 0

if __name__ == '__main__':
    try: sys.exit(main())
    except (ValueError, OSError, RuntimeError, KeyError) as e:
        print(f'{type(e).__name__}: {e}', file=sys.stderr); sys.exit(1)
