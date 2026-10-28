"""Apply the existing two-failure numerical admission rule, never model scoring.

Only original ResNet recipes with a separately registered FP32 counterpart are
eligible. No attempt is deleted/retried, no threshold changes, and active fits
are never interrupted. Whole-design ownership is the queue's admission control.
"""
from pathlib import Path
import argparse
import fcntl
import json
import os
import socket
import sys
import time

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'source'))
from common import catalog, check_seal, dump, load, sha, utc


def reconcile(dry_run=False):
    designs = catalog()
    owners = load('TASK_OWNERS.json')
    policy = load('NUMERICAL_ADMISSION.json')
    changes = []
    for name, spec in designs.items():
        if (spec.get('family') != 'neural'
                or spec.get('parameters', {}).get('kind') != 'resnet'
                or name.endswith('_fp32') or name + '_fp32' not in designs
                or name in policy['blocks']):
            continue
        counterpart = designs[name + '_fp32']
        assert counterpart['execution_precision'] == 'ieee_fp32_no_tf32'
        assert counterpart['parameters'] == spec['parameters']
        assert counterpart['view'] == spec['view']
        evidence = {}
        for path in sorted((R / 'runs').glob(name + '__s*/attempt*/FAILURE.json')):
            record = json.loads(path.read_text())
            error = record.get('error', '')
            if ('assert replay_error <= 5e-6' in error
                    and 'AssertionError:' in error
                    and 'models_neural.py' in error):
                evidence[record['task_id']] = dict(
                    path=str(path.relative_to(R)), sha256=sha(path))
        if len(evidence) < 2:
            continue
        changes.append(dict(design=name, previous_owner=owners['designs'][name],
                            observed_replay_failures=len(evidence),
                            evidence=evidence,
                            counterpart=name + '_fp32'))
    if dry_run or not changes:
        return changes
    previous = sha(R / 'NUMERICAL_ADMISSION.json')
    for change in changes:
        name = change['design']
        owners['designs'][name] = -1
        policy['blocks'][name] = dict(
            previous_owner=change['previous_owner'],
            observed_replay_failures=change['observed_replay_failures'],
            evidence=change['evidence'], counterpart=change['counterpart'],
            reason='Existing >=2 strict5e-6 replay-failure rule; block future '
                   'admission only, preserve active fits and every attempt. '
                   'Separately registered FP32 counterpart is unchanged. '
                   'Numerical validation limit, not negative predictive evidence.')
    policy['UTC'] = utc()
    policy['execution_rule_unchanged'] = True
    dump(owners, R / 'TASK_OWNERS.json')
    dump(policy, R / 'NUMERICAL_ADMISSION.json')
    event = dict(UTC=utc(), host=socket.gethostname(),
                 previous_policy_sha256=previous,
                 policy_sha256=sha(R / 'NUMERICAL_ADMISSION.json'),
                 operations_source_sha256=sha(Path(__file__)), changes=changes,
                 active_workers_unchanged=True, scientific_source_unchanged=True)
    with (R / 'NUMERICAL_ADMISSION_EVENTS.jsonl').open('a') as stream:
        stream.write(json.dumps(event, sort_keys=True) + '\n')
    return changes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    arguments = parser.parse_args()
    clock = check_seal()
    with (R / 'numerical_admission.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.time() < clock['model_deadline_epoch']:
            result = reconcile(arguments.dry_run)
            if not arguments.dry_run:
                dump(dict(UTC=utc(), pid=os.getpid(), host=socket.gethostname(),
                          stage='running', rule='unchanged >=2 strict replay failures',
                          model_deadline_epoch=clock['model_deadline_epoch']),
                     R / ('NUMERICAL_ADMISSION_WATCHER_' + socket.gethostname() + '_STATE.json'))
            if result:
                print(json.dumps(dict(UTC=utc(), dry_run=arguments.dry_run,
                                      blocked=[c['design'] for c in result])), flush=True)
            if arguments.once or arguments.dry_run:
                break
            time.sleep(min(10, max(0, clock['model_deadline_epoch'] - time.time())))
