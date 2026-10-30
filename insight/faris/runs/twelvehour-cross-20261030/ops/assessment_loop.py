"""Independent composition/evaluation, so collection never waits on a long score pass."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'source'))
from common import check_seal, load, dump, sha, utc


def module(name, extra=()):
    clock = load('CLOCK.json')
    remaining = clock['deadline_epoch'] - time.time() - 15
    if remaining < 1:
        raise TimeoutError('Original finalization ceiling reached')
    with (R / 'logs' / (name + '.log')).open('ab') as stream:
        subprocess.run([sys.executable, str(R / 'source' / (name + '.py')), *extra],
                       check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=remaining)


def main(mode):
    clock = check_seal()
    (R / 'logs').mkdir(exist_ok=True)
    if (R / 'COMPLETION.json').exists():
        return
    while time.time() < clock['deadline_epoch'] - 20:
        if mode == 'compose':
            module('compose')
            if (R / 'COMPUTATION_FINAL.json').exists():
                dump(dict(status='complete', UTC=utc(), clock_sha256=sha(R / 'CLOCK.json')),
                     R / 'COMPOSITION_FINAL.json')
                return
        else:
            if (R / 'COMPOSITION_FINAL.json').exists():
                module('evaluate', ['--final'])
                module('audit')
                dump(dict(status='completed_audited_research', UTC=utc(),
                          final_evaluation_sha256=sha(R / 'FINAL_EVALUATION.json'),
                          artifact_audit_sha256=sha(R / 'ARTIFACT_CONSISTENCY_AUDIT.json'),
                          clock_sha256=sha(R / 'CLOCK.json'), automatic_submission=False,
                          automatic_refit=False), R / 'COMPLETION.json')
                return
            module('evaluate')
        time.sleep(30 if mode == 'compose' else 180)
    dump(dict(status='finalization_ceiling_reached', UTC=utc(), no_clock_extension=True,
              completed=(R / 'COMPLETION.json').exists()), R / ('ASSESSMENT_LIMIT_' + mode + '.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['compose', 'evaluate'], required=True)
    main(parser.parse_args().mode)

