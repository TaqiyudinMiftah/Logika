"""Durable, single-writer, bounded adaptive campaign controller."""
from __future__ import annotations
import fcntl, importlib.metadata, json, math, os, pickle, random, signal, sys, time
from pathlib import Path
import psutil
from .common import atomic, read, sha, digest, probe, size_mb, validate_bundle, inside, check_dependencies
from .space import pool, tokens, canonical, suggest
from .executors import LocalExecutor, RayExecutor, terminate_identity
from .reporting import report

class Controller:
    def __init__(self, root, resume=False):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.optuna = optuna; self.root = Path(root).resolve(); self.lock = (self.root / 'controller.lock').open('a+')
        try: fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('campaign already has an active controller')
        self.c = validate_bundle(self.root); check_dependencies(self.root); self.families = {f['name']: f for f in self.c['families']}
        self.rng = random.Random(self.c.get('search_seed', 42)); self.active = {}; self.studies = {}; self.stopping = False
        self.executor = RayExecutor(self.root, self.c['ray_address']) if self.c.get('executor') == 'ray' else LocalExecutor(self.root)
        self.hosts = self.executor.nodes if isinstance(self.executor, RayExecutor) else {'local': probe(self.root)}
        now = time.time(); statepath = self.root / 'state.json'
        if resume:
            if not statepath.exists(): raise ValueError('no campaign state to resume')
            self.state = read(statepath)
            if self.state['status'] == 'finished': raise ValueError('campaign is finished; create a new campaign to extend work')
        else:
            if statepath.exists(): raise ValueError('campaign already started; use resume')
            self.state = {'version': 1, 'id': self.c['id'], 'start': now, 'deadline': now + self.c['budget_seconds'],
                'status': 'running', 'jobs': {}, 'admission_failures': [], 'confirm_planned': False, 'ever_launched': False}
        if self.c.get('external_clock'):
            clock = read(self.c['external_clock']); self.state['start'] = clock['start']; self.state['deadline'] = clock['deadline']
        if resume and (self.root / 'scheduler_rng.pkl').exists():
            with (self.root / 'scheduler_rng.pkl').open('rb') as f: self.rng.setstate(pickle.load(f))
        self.state['status'] = 'running'; self.state.pop('finished', None); self.state.pop('reason', None)
        env = {'python': sys.version, 'executable': sys.executable, 'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}, 'hosts': self.hosts}
        if (self.root / 'environment.json').exists():
            old = read(self.root / 'environment.json')
            if old['packages'] != env['packages'] or old['python'] != env['python']: raise ValueError('environment changed on resume')
        else: atomic(self.root / 'environment.json', env)
        self.pools = {}; self.exact = {}
        for name, family in self.families.items():
            self.pools[name], self.exact[name] = pool(family, self.c.get('coverage_pool_limit', 10000), self.c.get('search_seed', 42))
            sampler_path = self.root / f'sampler_{digest(name)[:12]}.pkl'
            if resume and sampler_path.exists():
                with sampler_path.open('rb') as f: sampler = pickle.load(f)
            else: sampler = optuna.samplers.TPESampler(seed=self.c.get('search_seed', 42), constant_liar=True)
            fidelity = family.get('fidelity')
            pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=int(fidelity['min_steps']), reduction_factor=int(fidelity['reduction_factor'])) if fidelity else optuna.pruners.NopPruner()
            self.studies[name] = optuna.create_study(study_name=name, storage='sqlite:///' + str(self.root / 'search.sqlite3'),
                direction=self.c['direction'], sampler=sampler, pruner=pruner, load_if_exists=True)
        if resume: self.recover()
        self.save(); atomic(self.root / 'owner.json', {'pid': os.getpid(), 'created': psutil.Process().create_time(), 'campaign': self.c['id']})
        self.last_probe = 0; self.last_storage_check = 0

    def save(self):
        atomic(self.root / 'state.json', self.state)
        atomic(self.root / 'heartbeat.json', {'time': time.time(), 'pid': os.getpid()})

    def checkpoint_samplers(self):
        path = self.root / 'scheduler_rng.pkl'; tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as f: pickle.dump(self.rng.getstate(), f)
        os.replace(tmp, path)
        for name, study in self.studies.items():
            path = self.root / f'sampler_{digest(name)[:12]}.pkl'; temp = path.with_suffix('.tmp')
            with temp.open('wb') as f: pickle.dump(study.sampler, f)
            os.replace(temp, path)

    def recover(self):
        for job in self.state['jobs'].values():
            if job['status'] == 'unstarted' and time.time() < self.state['deadline']:
                job['status'] = 'pending'; job.pop('reason', None)
            if job['status'] == 'running':
                d = self.root / job['directory']; atomic(d / 'cancel.json', {'reason': 'controller_recovery'})
                if job.get('identity', {}).get('pid'): terminate_identity(job['identity']['pid'], job['identity']['created'])
                if (d / 'outcome.json').exists() and read(d / 'outcome.json')['status'] == 'complete':
                    self.finish(job, 0, {})
                else:
                    self.finish(job, 1, {}, forced='interrupted')
            if job['status'] in ('interrupted', 'timeout', 'failed') and job['attempt'] <= self.c.get('max_retries', 1) and time.time() < self.state['deadline']:
                self.retry(job)
        # An interrupted ask() may exist without a committed job record.
        known = {(j['family'], j.get('trial_number')) for j in self.state['jobs'].values()}
        for name, study in self.studies.items():
            for trial in study.get_trials(states=(self.optuna.trial.TrialState.RUNNING,)):
                if (name, trial.number) not in known: study.tell(trial.number, state=self.optuna.trial.TrialState.FAIL)

    def family_seen(self, name):
        return {digest(j['params']) for j in self.state['jobs'].values() if j['family'] == name and j['protocol'] == 'screen'}

    def choose(self, breadth):
        names = [n for n in self.families if not self.families[n].get('stage_only') and (not self.exact[n] or len(self.family_seen(n)) < len(self.pools[n])) and
            sum(j['family'] == n and j['status'] in ('failed', 'invalid', 'unschedulable') for j in self.state['jobs'].values()) < self.c.get('family_failure_limit', 3)]
        if not names: return None
        unprobed = [n for n in names if not self.family_seen(n)]
        if unprobed: names = unprobed
        elif not breadth and self.rng.random() > self.c.get('exploration_fraction', .30):
            best = {}
            for n in names:
                vals = [j['score'] for j in self.state['jobs'].values() if j['family'] == n and j['status'] == 'complete' and j['protocol'] == 'screen']
                if vals: best[n] = min(vals) if self.c['direction'] == 'minimize' else max(vals)
            if best: names = sorted(best, key=best.get, reverse=self.c['direction'] == 'maximize')[:max(1, len(best) // 2)]
        # Calibrate each family once without serializing unrelated CPU/GPU queues.
        names = [n for n in names if not any(j['family']==n and j['status']=='running' for j in self.state['jobs'].values()) or any(j['family']==n and j['status']=='complete' for j in self.state['jobs'].values())]
        if not names: return None
        name = min(names, key=lambda n: sum(j['family'] == n and j['status'] in ('pending', 'running') for j in self.state['jobs'].values()))
        f = self.families[name]; seen = self.family_seen(name); study = self.studies[name]
        if breadth or unprobed:
            candidates = [p for p in self.pools[name] if digest(p) not in seen]
            if candidates:
                observed = set()
                for j in self.state['jobs'].values():
                    if j['family'] == name: observed |= tokens(f, j['params'])
                anchors = [a['params'] for a in f.get('anchors', []) if digest(canonical(f, a['params'])) not in seen]
                p = canonical(f, anchors[0]) if anchors else max(candidates, key=lambda p: len(tokens(f, p) - observed))
                study.enqueue_trial(p); trial = study.ask(); params = suggest(trial, f)
                return name, params, trial
        for _ in range(50):
            trial = study.ask()
            try: params = suggest(trial, f)
            except ValueError:
                study.tell(trial, state=self.optuna.trial.TrialState.FAIL); continue
            if digest(params) not in seen: return name, params, trial
            study.tell(trial, state=self.optuna.trial.TrialState.PRUNED)
        candidates = [p for p in self.pools[name] if digest(p) not in seen]
        if candidates:
            study.enqueue_trial(candidates[0]); trial = study.ask(); return name, suggest(trial, f), trial
        return None

    def new_job(self, name, params, trial=None, protocol='screen', seed=None):
        family = self.families[name]; idx = len(self.state['jobs'])
        job_id = f'job_{idx:06d}'; seed = self.c['seeds'][idx % len(self.c['seeds'])] if seed is None else seed
        r = dict(family['resources']); previous = [j for j in self.state['jobs'].values() if j['family'] == name and j['status'] == 'complete']
        if previous:
            r['ram_mb'] = max(r['ram_mb'], max(j.get('peak_ram_mb', 0) for j in previous) * 1.25)
            r['disk_mb'] = max(r['disk_mb'], max(j.get('artifact_mb', 0) for j in previous) * 1.25)
        job = {'job_id': job_id, 'family': name, 'params': params, 'hypothesis': family['hypothesis'], 'seed': seed, 'protocol': protocol,
            'resources': r, 'status': 'pending', 'attempt': 1, 'trial_number': trial.number if trial else None,
            'fidelity_group': family.get('fidelity', {}).get('group', 'full'), 'max_steps': family.get('fidelity', {}).get('max_steps', 1)}
        for a in family.get('anchors', []):
            if canonical(family, a['params']) == params: job['hypothesis'] = a.get('hypothesis', job['hypothesis'])
        self.state['jobs'][job_id] = job; self.save(); self.checkpoint_samplers(); return job

    def admit(self, job):
        r = job['resources']; required_disk = r['disk_mb']
        reserved_disk = sum(self.state['jobs'][k]['resources']['disk_mb'] for k in self.active)
        if size_mb(self.root) + reserved_disk + required_disk > self.c['storage_limit_mb']: return None, 'storage_budget'
        if r.get('opencl') and any(self.state['jobs'][k]['resources'].get('opencl') for k in self.active): return None, 'opencl_busy'
        live = probe(self.root)
        if live['disk_free_mb'] < reserved_disk + required_disk + self.c.get('disk_headroom_mb', 64): return None, 'disk_free'
        for node, host in self.hosts.items():
            active = [self.state['jobs'][k] for k in self.active if self.state['jobs'][k].get('node_id', 'local') == node]
            used_cpu = sum(j['resources']['cpus'] for j in active); used_ram = sum(j['resources']['ram_mb'] for j in active)
            cpu_cap = min(host['cpus'], self.c.get('cpu_limit', host['cpus']))
            ram_cap = min(host['ram_available_mb'] * .85, self.c.get('ram_limit_mb', host['ram_available_mb']))
            if used_cpu + r['cpus'] > cpu_cap or used_ram + r['ram_mb'] > ram_cap: continue
            if node == 'local' and r['ram_mb'] > live['ram_available_mb'] * .85: continue
            if r.get('gpus'):
                used_ids = {gid for j in active for gid in j.get('gpu_ids', [])}
                available = live['gpus'] if node == 'local' else host['gpus']
                gpus = [g for g in available if g['id'] not in used_ids and g['free_mb'] * .85 >= r['vram_mb']]
                if len(gpus) < r['gpus']: continue
                return {'node_id': node, 'gpu_ids': [g['id'] for g in gpus[:r['gpus']]]}, None
            return {'node_id': node}, None
        return None, 'cpu_ram_or_gpu_capacity'

    def launch(self, job, allocation):
        key = job['job_id']; d = self.root / 'runs' / key / f"attempt_{job['attempt']:03d}"; d.mkdir(parents=True)
        job.update(allocation); job['directory'] = str(d.relative_to(self.root)); job['started'] = time.time()
        spec = {**job, 'deadline': min(self.state['deadline'], job.get('stage_deadline',self.state['deadline']), time.time() + job['resources']['timeout_seconds']),
            'snapshot': read(self.root / 'seal.json')['snapshot']}
        atomic(d / 'spec.json', spec); atomic(d / 'environment.json', read(self.root / 'environment.json'))
        job['identity'] = self.executor.submit(key, spec, d); job['status'] = 'running'; job['progress_offset'] = 0
        self.active[key] = None; self.state['ever_launched'] = True; self.save()

    def progress(self, job):
        path = self.root / job['directory'] / 'progress.jsonl'
        if not path.exists(): return
        with path.open() as f:
            f.seek(job.get('progress_offset', 0))
            while True:
                start = f.tell(); line = f.readline()
                if not line or not line.endswith('\n'): f.seek(start); break
                try: event = json.loads(line)
                except ValueError: raise ValueError('invalid worker event')
                if job['protocol'] == 'screen' and self.families[job['family']].get('fidelity'):
                    trial = self.optuna.trial.Trial(self.studies[job['family']], self.studies[job['family']]._storage.get_trial_id_from_study_id_trial_number(self.studies[job['family']]._study_id, job['trial_number']))
                    trial.report(event['metric'], event['step'])
                    # Calibration anchors finish; subsequent compatible trials may be pruned.
                    if job['trial_number'] > 0 and trial.should_prune(): self.cancel(job, 'pruned')
            job['progress_offset'] = f.tell()

    def cancel(self, job, reason):
        if 'cancel_reason' in job: return
        job['cancel_reason'] = reason; job['cancel_time'] = time.time()
        atomic(self.root / job['directory'] / 'cancel.json', {'reason': reason})

    def finish(self, job, rc, usage, forced=None):
        directory = self.root / job['directory']; outcome_path = directory / 'outcome.json'
        outcome = read(outcome_path) if outcome_path.exists() else {'status': 'failed', 'reason': f'worker exited {rc} without outcome'}
        status = forced or job.get('cancel_reason') or outcome['status']
        if status not in ('complete', 'pruned', 'interrupted', 'timeout', 'failed', 'invalid'): status = 'interrupted'
        if status == 'complete':
            try:
                for name, expected in outcome['artifacts'].items():
                    if sha(inside(directory, name)) != expected: raise ValueError('worker artifact mismatch')
                score = outcome['evaluation']['score']
                if not math.isfinite(score) or not outcome.get('models'): raise ValueError('invalid evidence')
                job['score'] = score
            except (KeyError, ValueError, OSError): status = 'invalid'
        job.update(usage); job['status'] = status; job['finished'] = time.time(); job['artifact_mb'] = size_mb(directory)
        reason = job.get('cancel_reason') or outcome.get('reason')
        record = {**job, 'reason': reason, 'duration_seconds': job['finished'] - job['started'], 'metric': self.c['metric'],
            'direction': self.c['direction'], 'split_sha': sha(self.c['protocols'][job['protocol']]['split']),
            'snapshot': read(self.root / 'seal.json')['snapshot'], 'models': outcome.get('models', read(directory / 'models.json') if (directory / 'models.json').exists() else []),
            'evaluation': outcome.get('evaluation'), 'artifacts': outcome.get('artifacts', {}),
            'outcome_sha': sha(outcome_path) if outcome_path.exists() else None,
            'missing_model_reason': None if (directory / 'models.json').exists() else 'worker did not produce a model before termination'}
        # Preserve logs, execution specs and any unregistered intermediate checkpoints as well.
        for p in directory.rglob('*'):
            if p.is_file() and p.name != 'outcome.json': record['artifacts'][str(p.relative_to(directory))] = sha(p)
        dest = self.root / 'records' / f"{job['job_id']}_attempt_{job['attempt']:03d}.json"
        if not dest.exists(): atomic(dest, record)
        if job.get('trial_number') is not None:
            study = self.studies[job['family']]
            if study.trials[job['trial_number']].state == self.optuna.trial.TrialState.RUNNING:
                if status == 'complete': study.tell(job['trial_number'], job['score'])
                elif status == 'pruned': study.tell(job['trial_number'], state=self.optuna.trial.TrialState.PRUNED)
                elif not self.can_retry(job, reason): study.tell(job['trial_number'], state=self.optuna.trial.TrialState.FAIL)
        self.save()

    def can_retry(self, job, reason):
        # Ordinary code/validation exceptions are deterministic and are not retried.
        return job['status'] in ('interrupted', 'timeout', 'failed') and job['attempt'] <= self.c.get('max_retries', 1) and (
            job['status'] == 'interrupted' or 'without outcome' in str(reason)) and time.time() < self.state['deadline'] and not self.stopping

    def retry(self, job):
        previous = str(self.root / job['directory']); job.update({'status': 'pending', 'attempt': job['attempt'] + 1, 'resume_from': previous})
        for k in ('cancel_reason', 'cancel_time', 'score', 'identity'): job.pop(k, None)
        if job.get('trial_number') is not None:
            study = self.studies[job['family']]
            if study.trials[job['trial_number']].state != self.optuna.trial.TrialState.RUNNING:
                study.enqueue_trial(job['params']); trial = study.ask(); suggest(trial, self.families[job['family']]); job['trial_number'] = trial.number
        self.save()

    def confirmation(self):
        if self.state['confirm_planned']: return
        self.state['confirm_planned'] = True
        if 'confirm' not in self.c['protocols']: return
        rows = sorted([j for j in self.state['jobs'].values() if j['protocol'] == 'screen' and j['status'] == 'complete'], key=lambda j: j['score'], reverse=self.c['direction'] == 'maximize')
        selected = set(); group_counts = {}
        for j in rows:
            identity = digest([j['family'], j['params']])
            if identity in selected or group_counts.get(j['fidelity_group'], 0) >= self.c.get('confirmation_top_k', 3): continue
            selected.add(identity); group_counts[j['fidelity_group']] = group_counts.get(j['fidelity_group'], 0) + 1
            for seed in self.c.get('confirmation_seeds', self.c['seeds']): self.new_job(j['family'], j['params'], protocol='confirm', seed=seed)
        self.save()

    def run(self):
        def stop(sig, frame): self.stopping = True
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        reserve_start = self.state['deadline'] - self.c['budget_seconds'] * self.c.get('reserve_fraction', .15)
        breadth_end = self.state['start'] + self.c['budget_seconds'] * self.c.get('breadth_fraction', .4)
        reason = 'search_exhausted'; failed = None
        try:
            while True:
                now = time.time(); self.save()
                if now-self.last_probe>=10:
                    self.last_probe=now; sample=probe(self.root);sample['active_jobs']=list(self.active)
                    sample['amd_busy_percent']={}
                    for card in Path('/sys/class/drm').glob('card*/device/gpu_busy_percent'):
                        try:sample['amd_busy_percent'][str(card)]=float(card.read_text().strip())
                        except (OSError,ValueError):pass
                    with (self.root/'resources.jsonl').open('a') as handle:handle.write(json.dumps(sample)+'\n')
                if now - self.last_storage_check >= 1:
                    self.last_storage_check = now
                    if size_mb(self.root) >= self.c['storage_limit_mb']:
                        self.stopping = True; reason = 'storage_budget'
                    elif __import__('shutil').disk_usage(self.root).free / 1048576 < self.c.get('disk_headroom_mb', 64):
                        self.stopping = True; reason = 'disk_free'
                if (self.root / 'stop.json').exists(): self.stopping = True
                if self.stopping and reason not in ('storage_budget', 'disk_free'): reason = 'user_stop'
                elif now >= self.state['deadline']: reason = 'deadline'
                if self.stopping or now >= self.state['deadline']:
                    for key in list(self.active): self.cancel(self.state['jobs'][key], 'interrupted')
                for key in list(self.active):
                    job = self.state['jobs'][key]; self.progress(job); rc, usage = self.executor.poll(key)
                    if now - job['started'] >= job['resources']['timeout_seconds']: self.cancel(job, 'timeout')
                    if usage.get('peak_ram_mb', 0) > job['resources']['ram_mb'] * 1.25: self.cancel(job, 'interrupted')
                    if usage.get('peak_vram_mb', 0) > job['resources'].get('vram_mb', float('inf')) * max(1, job['resources'].get('gpus', 0)) * 1.25: self.cancel(job, 'interrupted')
                    if job.get('cancel_time') and now - job['cancel_time'] >= self.c.get('cancel_grace_seconds', 2): self.executor.cancel(key)
                    if rc is not None:
                        self.finish(job, rc, usage); self.executor.forget(key); self.active.pop(key)
                        if self.can_retry(job, read(self.root / 'records' / f"{key}_attempt_{job['attempt']:03d}.json").get('reason')): self.retry(job)
                if (self.stopping or now >= self.state['deadline']) and not self.active: break
                if self.stopping or now >= self.state['deadline']: time.sleep(.05); continue
                stage_path = Path(self.c.get('stage_tasks', str(self.root/'no_tasks.json')))
                if stage_path.exists():
                    submitted = {j.get('external_id') for j in self.state['jobs'].values()}
                    for request in read(stage_path):
                        if request['id'] in submitted: continue
                        key = 'stage_' + request['id']; f=self.families.get(request['family'])
                        if f is None: continue
                        j=self.new_job(request['family'],request['params'],protocol=request['protocol'],seed=request.get('seed',2026));j['external_id']=request['id'];j['stage_deadline']=request.get('deadline',self.state['deadline']);submitted.add(request['id']);self.save()
                if now >= reserve_start:
                    # Close screening before selecting confirmations; no adaptive feedback from confirm.
                    for key in self.active:
                        if self.state['jobs'][key]['protocol'] == 'screen': self.cancel(self.state['jobs'][key], 'interrupted')
                    if not self.active: self.state['confirm_planned']=True
                if (self.root/'close.json').exists() and not self.active: reason='stages_complete'; break
                slots = 0 if (self.root/'pause_admission.json').exists() else self.c['max_workers'] - len(self.active)
                # Calibrate unknown families serially.
                calibrated = {j['family'] for j in self.state['jobs'].values() if j['status'] == 'complete'}
                # Unknown families get one representative each, on independent resources.
                launched = False; no_candidate = False; blocked_ids = set(); scheduled = 0
                for _ in range(max(0, slots) * 4):
                    if len(self.active) >= self.c['max_workers']: break
                    pending = [j for j in self.state['jobs'].values() if j['status'] == 'pending' and j['job_id'] not in blocked_ids and (now < reserve_start or j['protocol'] != 'screen')]
                    job = pending[0] if pending else None
                    screen_count = sum(j['protocol'] == 'screen' for j in self.state['jobs'].values())
                    if job is None and now < reserve_start and screen_count < self.c['max_trials']:
                        chosen = self.choose(now < breadth_end)
                        if chosen: job = self.new_job(*chosen)
                    if job is None: no_candidate = True; break
                    allocation, blocked = self.admit(job)
                    if allocation is None:
                        self.state['admission_failures'].append({'job': job['job_id'], 'reason': blocked, 'time': now})
                        if self.active:
                            blocked_ids.add(job['job_id'])
                            continue
                        job['status'] = 'unschedulable'; job['reason'] = blocked
                        if blocked in ('disk_free', 'storage_budget'): reason = blocked; self.stopping = True; break
                        continue
                    durations = [j['finished'] - j['started'] for j in self.state['jobs'].values() if j['family'] == job['family'] and j['status'] == 'complete']
                    estimate = max(durations) * 1.25 if durations else min(job['resources']['timeout_seconds'], 5)
                    cutoff = reserve_start if job['protocol'] == 'screen' else min(self.state['deadline'], job.get('stage_deadline',self.state['deadline']))
                    if time.time() + estimate >= cutoff:
                        job['status'] = 'budget_skipped'; job['reason'] = 'insufficient_runtime'; continue
                    self.launch(job, allocation); launched = True; scheduled += 1
                    if scheduled >= slots: break
                screen_count = sum(j['protocol'] == 'screen' for j in self.state['jobs'].values())
                if self.c.get('external_clock') and not self.active and not launched:
                    time.sleep(1); continue
                if not self.active and not launched and (no_candidate or screen_count >= self.c['max_trials']):
                    if not self.state['confirm_planned']:
                        self.confirmation()
                        # When the bounded search ends early, use the released time for confirmation.
                        if any(j['status'] == 'pending' and j['protocol'] == 'confirm' for j in self.state['jobs'].values()): continue
                    if not any(j['status'] == 'pending' for j in self.state['jobs'].values()): break
                    if now >= reserve_start and not any(j['status'] == 'pending' and j['protocol'] == 'confirm' for j in self.state['jobs'].values()): break
                self.save(); time.sleep(self.c.get('poll_seconds', .1))
        except BaseException as e:
            failed = e; reason = f'controller_error: {type(e).__name__}: {e}'
        finally:
            for key in list(self.active):
                j = self.state['jobs'][key]; self.cancel(j, 'interrupted'); self.executor.cancel(key)
                self.finish(j, 1, {}, forced='interrupted')
            self.active.clear(); self.checkpoint_samplers()
            self.state.update({'status': 'stopped' if self.stopping or failed else 'finished', 'reason': reason, 'finished': time.time()})
            for job in self.state['jobs'].values():
                if job['status'] == 'pending': job.update({'status': 'unstarted', 'reason': reason})
            self.save(); self.executor.close()
            if (self.root / 'stop.json').exists(): (self.root / 'stop.json').unlink()
            report(self.root)
            fcntl.flock(self.lock, fcntl.LOCK_UN); self.lock.close()
        if failed: raise failed
        return self.state
