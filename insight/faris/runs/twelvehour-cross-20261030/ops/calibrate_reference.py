"""One metered registered full-size control fit before any broad admission."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import time
import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'source'))
sys.path.insert(0, str(R / 'ops'))
from common import load, dump, sha, utc, check_seal
from coordinator import setup, SPECS, collect


def main():
    clock = check_seal()
    assert load('GATE.json')['status'] == 'PASS'
    if (R / 'BASELINE_REPLAY_GATE.json').exists():
        assert load('BASELINE_REPLAY_GATE.json')['status'] == 'PASS'
        return
    setup('cpu2001', start=False)
    root = Path(SPECS['cpu2001']['root'])
    task = load('CONFIG.json')['baseline_pilot_task']
    folder = root / 'runs' / task / 'attempt1'
    if not (folder / 'RESULT.json').exists():
        assert not folder.exists(), 'Existing calibration attempt needs reconciliation; no implicit retry'
        with (R / 'logs/baseline_calibration.log').open('ab') as stream:
            subprocess.run([SPECS['cpu2001']['python'], str(root / 'source/worker.py'),
                            '--task', task, '--pilot'], check=True, stdout=stream,
                           stderr=subprocess.STDOUT,
                           timeout=min(1900, clock['model_deadline_epoch'] - time.time() - 30))
    old = Path('/mnt/home/faris/logika-overnight-refit-20261030/localglm-cann/runs/'
               'localglm_actuarial99_intercept_w64_r0p1_raw__s2026__f0/attempt1')
    current, reference = np.load(folder / 'prediction.npz'), np.load(old / 'prediction.npz')
    assert np.array_equal(current['rows'], reference['rows'])
    error = float(np.max(np.abs(current['raw'] - reference['raw'])))
    old_start = json.loads((old / 'START.json').read_text())
    new_start = json.loads((folder / 'START.json').read_text())
    for key in ['context_rows_sha256', 'prediction_rows_sha256', 'declared_inner_train_rows_sha256',
                'declared_inner_stop_rows_sha256', 'seed']:
        assert old_start[key] == new_start[key]
    value = dict(status='PASS' if error <= 5e-6 else 'FAIL_BASELINE_REPLAY', UTC=utc(),
                 task=task, rows=20000, probability_tolerance=5e-6,
                 max_probability_error=error, reference_prediction_sha256=sha(old / 'prediction.npz'),
                 current_prediction_sha256=sha(folder / 'prediction.npz'),
                 current_start_sha256=sha(folder / 'START.json'),
                 current_result_sha256=sha(folder / 'RESULT.json'),
                 clock_sha256=sha(R / 'CLOCK.json'), source_seal_sha256=sha(R / 'SOURCE_SEAL.json'),
                 config_sha256=sha(R / 'CONFIG.json'), contract_sha256=sha(R / 'VALIDATION_CONTRACT.json'),
                 new_fullsize_fits=1, included_in_registered_6160_slots=True,
                 outer_labels_read=False, no_independent_gain_claim=True)
    dump(value, R / 'BASELINE_REPLAY_GATE.json')
    shutil.copy2(R / 'BASELINE_REPLAY_GATE.json', root / 'BASELINE_REPLAY_GATE.json')
    collect('cpu2001')
    assert value['status'] == 'PASS', 'Broad experiments blocked by actual-data baseline reproduction'
    print(json.dumps(value))


if __name__ == '__main__':
    main()
