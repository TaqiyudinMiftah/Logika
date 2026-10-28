"""Route only the sealed logical checkpoint to its identical Colab-local bytes.

Original scientific modules remain byte-for-byte sealed. This adapter changes no
data, parameters, preprocessing, model code, seed, precision or batch size.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'source'))
import common
import worker
import kumo_engine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--attempt', type=int, default=2)
    args = parser.parse_args()
    clock = common.check_seal()
    assert time.time() < min(clock['admission_end_epoch'], clock['deadline_epoch'] - 7200)
    environment = common.load('COLAB_ENVIRONMENT.json')
    canonical = common.load('PATHS.json')['checkpoint']
    physical = environment['checkpoint']
    expected = 'cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613'
    assert common.sha(physical) == expected == environment['checkpoint_sha256']
    assert common.load('CONFIG.json')['tasks'][args.task]['family'] == 'kumo'
    original_predict, original_sha = kumo_engine.predict, worker.sha

    def predict(*positional, **keywords):
        positional = list(positional)
        assert str(positional[4]) == canonical
        positional[4] = physical
        return original_predict(*positional, **keywords)

    def mapped_sha(path):
        return original_sha(physical if str(path) == canonical else path)

    kumo_engine.predict = predict
    worker.sha = mapped_sha
    folder = R / f'runs/{args.task}/attempt{args.attempt}'
    assert not folder.exists()
    try:
        worker.main(args.task, args.attempt)
    finally:
        if folder.exists():
            common.dump({
                'execution_backend': 'Colab account2 A100',
                'adapter_sha256': common.sha(__file__),
                'original_source_seal_sha256': common.sha(R / 'SOURCE_SEAL.json'),
                'original_clock_sha256': common.sha(R / 'CLOCK.json'),
                'environment_sha256': common.sha(R / 'COLAB_ENVIRONMENT.json'),
                'checkpoint_logical_path': canonical,
                'checkpoint_physical_path': physical,
                'checkpoint_sha256': expected,
                'scientific_parameters_changed': False,
                'GPU_bitwise_equivalence_claimed': False,
            }, folder / 'EXECUTION_BACKEND.json')

if __name__ == '__main__':
    main()
