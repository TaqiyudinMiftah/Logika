"""Single-trial worker. No search, credentials, external services, or shared DB writes."""
import argparse, json, math, os, signal, sys, threading, time, traceback
from pathlib import Path
from .common import atomic, read, sha, inside, load_adapter

class TrialStopped(Exception): pass

class Context:
    def __init__(self, campaign, spec, directory):
        self.campaign = Path(campaign); self.spec = spec; self.directory = Path(directory)
        self.config = read(self.campaign / 'config.json')
        self.inputs = self.config['inputs']; self.params = spec['params']; self.seed = spec['seed']
        self.protocol = self.config['protocols'][spec['protocol']]
        self.split = Path(self.protocol['split']); self.family = spec['family']
        self.resume_from = spec.get('resume_from'); self.max_steps = spec.get('max_steps', 1)
        self._models = []; self._last_step = -1

    def path(self, name):
        p = inside(self.directory, name); p.parent.mkdir(parents=True, exist_ok=True); return p

    def save_json(self, name, value):
        p = self.path(name); atomic(p, value); return str(p.relative_to(self.directory))

    def model(self, name):
        p = self.path(name)
        if not p.is_file() or p.stat().st_size == 0: raise ValueError('model must exist before registering it')
        if name not in self._models: self._models.append(name)
        atomic(self.directory / 'models.json', self._models)
        return name

    def check_stop(self):
        if (self.directory / 'cancel.json').exists(): raise TrialStopped(read(self.directory / 'cancel.json')['reason'])
        if time.time() >= self.spec['deadline']: raise TrialStopped('deadline')
        heartbeat = self.campaign / 'heartbeat.json'
        if not self.spec.get('replay') and heartbeat.exists() and time.time() - heartbeat.stat().st_mtime > self.config.get('orphan_timeout_seconds', 60):
            raise TrialStopped('controller_lost')

    def report(self, step, metric):
        if type(step) is not int or step <= self._last_step or step > self.max_steps: raise ValueError('steps must increase within max_steps')
        if not math.isfinite(float(metric)): raise ValueError('nonfinite intermediate metric')
        self._last_step = step
        with (self.directory / 'progress.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, 'metric': float(metric), 'time': time.time()}) + '\n'); f.flush()
        self.check_stop()

def check_evaluation(value):
    if not isinstance(value, dict) or not isinstance(value.get('score'), (int, float)) or not math.isfinite(value['score']):
        raise ValueError('evaluator must return finite score')
    for x in value.get('per_fold', []):
        if not math.isfinite(x): raise ValueError('nonfinite fold score')
    json.dumps(value, allow_nan=False)
    return value

def execute(campaign, spec_path, directory, verify=False):
    spec = read(spec_path); directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    ctx = Context(campaign, spec, directory); start = time.time(); result = None
    def stopped(sig, frame): raise TrialStopped('signal')
    signal.signal(signal.SIGTERM, stopped)
    def watchdog():
        while True:
            time.sleep(.25)
            try: ctx.check_stop()
            except TrialStopped:
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(ctx.config.get('cancel_grace_seconds', 2) + 1)
                if os.getpgrp() == os.getpid(): os.killpg(os.getpgrp(), signal.SIGKILL)
                else: os.kill(os.getpid(), signal.SIGKILL)
                return
    if not verify: threading.Thread(target=watchdog, daemon=True).start()
    try:
        adapter = load_adapter(campaign)
        if not verify: atomic(directory / 'started.json', {'pid': os.getpid(), 'time': time.time(), 'snapshot': spec.get('snapshot')})
        if verify:
            result = read(directory / 'result.json')
        else:
            ctx.check_stop()
            result = adapter.run_trial(ctx)
            if not isinstance(result, dict) or not result.get('validation'): raise ValueError('run_trial must identify saved validation outputs')
            # Caller-supplied scores are ignored: evaluate only reads persisted output.
            for name in result['validation']:
                p = inside(directory, name)
                if not p.is_file() or not p.stat().st_size: raise ValueError('missing validation artifact')
            if not ctx._models: raise ValueError('no retained model/solution artifact')
            atomic(directory / 'result.json', result)
        score = check_evaluation(adapter.evaluate(ctx, result))
        if verify: atomic(directory / 'reevaluation.json', score); return 0
        model_files = read(directory / 'models.json')
        artifacts = {name: sha(inside(directory, name)) for name in set(result['validation'] + model_files + ['result.json', 'models.json'])}
        atomic(directory / 'outcome.json', {'status': 'complete', 'evaluation': score, 'artifacts': artifacts,
            'started': start, 'finished': time.time(), 'models': model_files})
        return 0
    except TrialStopped as e:
        atomic(directory / 'outcome.json', {'status': 'interrupted', 'reason': str(e), 'started': start, 'finished': time.time()})
        return 2
    except Exception as e:
        traceback.print_exc()
        atomic(directory / 'outcome.json', {'status': 'failed', 'reason': f'{type(e).__name__}: {e}', 'started': start, 'finished': time.time()})
        return 1

def main():
    p = argparse.ArgumentParser(); p.add_argument('campaign'); p.add_argument('spec'); p.add_argument('directory'); p.add_argument('--verify', action='store_true')
    a = p.parse_args(); return execute(a.campaign, a.spec, a.directory, a.verify)

if __name__ == '__main__': sys.exit(main())
