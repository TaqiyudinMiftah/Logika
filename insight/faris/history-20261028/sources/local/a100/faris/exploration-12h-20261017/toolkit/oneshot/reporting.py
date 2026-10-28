"""Evidence readback and machine-readable, searchable campaign reports."""
import csv, json, math, statistics, time
from pathlib import Path
from .common import read, atomic, sha, inside, digest, validate_bundle
from .space import coverage

def checked_records(root):
    root = Path(root); config = validate_bundle(root, check_inputs=False); records = []
    for p in sorted((root / 'records').glob('*.json')):
        r = read(p)
        if r['status'] == 'complete':
            directory = inside(root, r['directory'])
            try:
                if r['metric'] != config['metric'] or r['direction'] != config['direction']: raise ValueError('metric mismatch')
                expected_split = sha(config['protocols'][r['protocol']]['split'])
                if r['split_sha'] != expected_split: raise ValueError('protocol mismatch')
                if not math.isfinite(r['score']): raise ValueError('nonfinite score')
                if not r.get('models'): raise ValueError('missing retained models')
                for name, expected in r['artifacts'].items():
                    path=inside(directory,name)
                    if not path.exists() and (directory/'archive_receipt.json').exists():
                        receipt=read(directory/'archive_receipt.json')
                        if receipt['verified_files'].get(name)!=expected:raise ValueError('Unverified archive reference')
                        r.setdefault('archived_model_locations',{})[name]=receipt['archive']+'/'+name
                    elif sha(path) != expected: raise ValueError(f'artifact changed: {name}')
                outcome = read(directory / 'outcome.json')
                if outcome['evaluation']['score'] != r['score']: raise ValueError('score changed')
                if sha(directory / 'outcome.json') != r['outcome_sha']: raise ValueError('outcome changed')
            except (ValueError, OSError, KeyError, TypeError) as e:
                r = {**r, 'status': 'invalid', 'verification_error': str(e)}
        records.append(r)
    return records

def report(root):
    root = Path(root); c = read(root / 'config.json'); state = read(root / 'state.json')
    records = checked_records(root); jobs = list(state['jobs'].values())
    # Retry attempts remain in the archive; only the final attempt counts toward outcomes.
    latest = {r['job_id']: r for r in records}
    final = list(latest.values()); valid = [r for r in final if r['status'] == 'complete']
    boards = {}
    for r in valid:
        key = f"{r['protocol']}:{r['split_sha']}:{r['fidelity_group']}"
        boards.setdefault(key, []).append({k: r.get(k) for k in ('job_id', 'family', 'params', 'seed', 'score', 'directory', 'duration_seconds', 'peak_ram_mb', 'hypothesis')})
    for rows in boards.values(): rows.sort(key=lambda x: x['score'], reverse=c['direction'] == 'maximize')
    cov = coverage(c, final)
    counts = {s: sum(r['status'] == s for r in final) for s in ('complete', 'pruned', 'failed', 'interrupted', 'timeout', 'invalid')}
    counts.update({'attempts': len(records), 'retries': max(0, len(records) - len(latest)), 'jobs': len(jobs),
        'unique_configurations': len({digest([r['family'], r['params']]) for r in final}),
        'screen_jobs': sum(r['protocol'] == 'screen' for r in final), 'confirmation_jobs': sum(r['protocol'] == 'confirm' for r in final)})
    for status in ('unschedulable', 'budget_skipped', 'unstarted', 'pending', 'running'):
        counts[status] = sum(j['status'] == status for j in jobs)
    exact = all(v['denominator_kind'] == 'exact_finite_space' for v in cov.values())
    total = sum(v['candidate_count'] for v in cov.values())
    completed = sum(v['reference_pool_configs_completed'] for v in cov.values())
    untested = {'kind': 'exact' if exact else 'reference_pool_only', 'candidate_count': total,
        'not_completed': max(0, total - completed), 'note': 'Pruned/failed configurations are not completed answers; continuous spaces have no finite global denominator.'}
    ablations = []
    for f in c['families']:
        groups = {}
        for anchor in f.get('anchors', []):
            if anchor.get('ablation'): groups.setdefault(anchor['ablation']['group'], []).append(anchor)
        for group, anchors in groups.items():
            matched = [r for r in valid if r['family'] == f['name'] and any(r['params'] == a['params'] for a in anchors)]
            # Report matched seed/protocol pairs only; no causal claims for an unmatched search result.
            controls = [a for a in anchors if a['ablation'].get('control')]
            for a in anchors:
                if a['ablation'].get('control') or not controls: continue
                for r in matched:
                    if r['params'] != a['params']: continue
                    control = next((x for x in matched if x['params'] == controls[0]['params'] and x['seed'] == r['seed'] and x['protocol'] == r['protocol'] and x['fidelity_group'] == r['fidelity_group']), None)
                    if control:
                        ablations.append({'group': group, 'factor': a['ablation']['factor'], 'control': control['job_id'], 'treatment': r['job_id'], 'delta': r['score'] - control['score'], 'protocol': r['protocol']})
    grouped = {}
    for r in valid:
        identity = digest([r['family'], r['params'], r['protocol'], r['split_sha'], r['fidelity_group']])
        grouped.setdefault(identity, []).append(r)
    seed_summary = [{'family': rows[0]['family'], 'params': rows[0]['params'], 'protocol': rows[0]['protocol'],
        'seeds': [r['seed'] for r in rows], 'mean': statistics.mean(r['score'] for r in rows),
        'population_std': statistics.pstdev(r['score'] for r in rows), 'job_ids': [r['job_id'] for r in rows]} for rows in grouped.values()]
    findings = []
    for group, rows in boards.items():
        best = rows[0]
        same_seed = sorted([r for r in valid if r['job_id'] in {v['job_id'] for v in rows} and r['seed'] == best['seed']], key=lambda r: r['started'])
        baseline = same_seed[0]
        findings.append({'comparison_group': group, 'best_job': best['job_id'], 'baseline_job': baseline['job_id'],
            'score_delta': best['score'] - baseline['score'], 'best_score': best['score'], 'interpretation': 'Observed search comparison; not a causal ablation or an unbiased generalization estimate.'})
    limitations = ['Search scores are selection-biased; confirmation results are reported separately.',
        'CPU seconds are sampled process-tree lower bounds; GPU allocation-seconds are not measured busy GPU time.',
        'Pair coverage concerns declared categorical factors within families; it does not prove higher-order or global optimality.']
    if 'confirm' not in c['protocols']: limitations.append('No protected confirmation protocol supplied; no independent generalization claim.')
    if c.get('executor', 'local') == 'ray': limitations.append('Ray requires shared durable paths and compatible environments on every participating node.')
    summary = {'campaign': c['id'], 'state': state['status'], 'reason': state.get('reason'), 'start': state['start'], 'deadline': state['deadline'],
        'finished': state.get('finished'), 'counts': counts, 'coverage': cov, 'untested': untested, 'scoreboards': boards, 'ablations': ablations,
        'seed_summary': seed_summary, 'findings': findings,
        'unstarted_jobs': [{k: j.get(k) for k in ('job_id','family','params','protocol','status','reason')} for j in jobs if j['status'] in ('unschedulable','budget_skipped','unstarted','pending')],
        'limitations': limitations, 'admission_failures': state.get('admission_failures', []),
        'cpu_seconds_sampled': sum(r.get('cpu_seconds_sampled', 0) for r in records),
        'gpu_allocated_seconds': sum(r.get('duration_seconds', 0) * r['resources'].get('gpus', 0) for r in records),
        'code_snapshot': read(root / 'seal.json')['snapshot'], 'generated_at': time.time()}
    dest = root / 'reports'; dest.mkdir(exist_ok=True); atomic(dest / 'summary.json', summary)
    with (dest / 'trials.jsonl').open('w') as f:
        for r in records: f.write(json.dumps(r, allow_nan=False) + '\n')
    with (dest / 'trials.csv').open('w', newline='') as f:
        columns = ['job_id', 'attempt', 'family', 'protocol', 'seed', 'status', 'score', 'duration_seconds', 'directory', 'reason']
        w = csv.DictWriter(f, columns, extrasaction='ignore'); w.writeheader(); w.writerows(records)
    lines = [f"# Experiment report: {c['id']}", '', f"State: **{state['status']}**. Stop reason: {state.get('reason', 'running')}.", '',
        '## Outcomes', '', json.dumps(counts), '', '## Coverage', '', '| Family | Complete configs | Candidate pool | Pair coverage | Denominator |', '|---|---:|---:|---:|---|']
    for name, v in cov.items(): lines.append(f"| {name} | {v['unique_completed_configs']} | {v['candidate_count']} | {v['pairs_covered']}/{v['pairs_total']} | {v['denominator_kind']} |")
    lines += ['', '## Verified scoreboards', '']
    for key, rows in boards.items():
        lines += [f'### {key}', '', '| Job | Family | Score | Seconds | Evidence |', '|---|---|---:|---:|---|']
        for r in rows[:20]: lines.append(f"| {r['job_id']} | {r['family']} | {r['score']:.9g} | {r['duration_seconds']:.3f} | `{r['directory']}` |")
    lines += ['', '## Matched ablations', '', 'See summary.json for matched control/treatment differences. Unmatched experiments are not presented as ablations.', '',
        '## Limitations and unresolved questions', ''] + [f'- {x}' for x in limitations]
    lines += [f"- {name}: {v['unresolved_hypothesis']}" for name, v in cov.items() if v['unresolved_hypothesis']]
    lines += ['', '## Reproduction', '', 'Each attempt contains spec.json, process.json, environment.json, and immutable source references. Replay one with:', '',
        f'`{__import__("sys").executable} {root}/runtime/campaign.py replay {root} JOB_ID --output NEW_DIRECTORY`', '',
        'All retained model and validation artifacts are indexed in records/*.json. No submission files are produced.']
    (dest / 'report.md').write_text('\n'.join(lines) + '\n')
    return summary
