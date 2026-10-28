from __future__ import annotations

import gc
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from logika_ui_26.experiment_io import sha256_file, write_json
from logika_ui_26.kumo import supervise


def _predict_foundation(folder: Path, root: Path) -> pd.DataFrame:
    identity = json.loads((folder / 'identity.json').read_text())
    cache = json.loads((folder / 'feature_cache.json').read_text())
    parquet = root / 'data/processed/feature_cache' / (cache['fingerprint'] + '.parquet')
    if sha256_file(parquet) != identity['feature_hash']:
        raise ValueError('Foundation inference feature cache differs')
    for name, expected in cache['source']['files'].items():
        if sha256_file(root / 'data/raw' / name) != expected:
            raise ValueError('Foundation raw source differs from training')
    for name, expected in identity['source_hashes'].items():
        if sha256_file(folder / 'source' / name) != expected:
            raise ValueError('Historical foundation source snapshot differs')
    sample = pd.read_csv(root / 'data/raw/sample_submission.csv')
    plan = json.loads((folder / 'plan.json').read_text())
    work = root / 'artifacts/prediction_rebuild' / (folder.name + '-' + uuid.uuid4().hex[:8])
    environment = work / 'source_environment'
    environment.mkdir(parents=True)
    (environment / 'src').symlink_to((folder / 'source/src').resolve(), target_is_directory=True)
    (environment / 'artifacts').mkdir()
    (environment / 'artifacts/hf_cache').symlink_to((root / 'artifacts/hf_cache').resolve(), target_is_directory=True)
    write_json(work / 'status.json', {'state': 'running', 'source_run': str(folder)})
    predictions = []
    try:
        for split in plan:
            number = split['fold']
            original = folder / f'fold_{number}'
            destination = work / f'fold_{number}'
            destination.mkdir()
            complete = json.loads((original / 'complete.json').read_text())
            for name, expected in complete['hashes'].items():
                if sha256_file(original / name) != expected:
                    raise ValueError('Trained foundation fold artifact differs')
            request = json.loads((original / 'request.json').read_text())
            request.update(parquet=str(parquet.resolve()), output=str(destination.resolve()))
            if request['config'].get('feature_limit') is not None:
                for name in ('selection.json', 'selection_model.txt'):
                    shutil.copyfile(original / name, destination / name)
            write_json(destination / 'request.json', request)
            gc.collect()
            import pyarrow as pa

            pa.default_memory_pool().release_unused()
            try:
                import ctypes

                ctypes.CDLL(None).malloc_trim(0)
            except (AttributeError, OSError):
                pass
            resources = supervise(destination / 'request.json', environment,
                                  request['config']['process_budget_gib'])
            write_json(destination / 'resources.json', resources)
            rebuilt = json.loads((destination / 'model.json').read_text())
            fitted = json.loads((original / 'model.json').read_text())
            if rebuilt['asset_sha256'] != fitted['asset_sha256']:
                raise ValueError('Reconstructed foundation checkpoint assets differ')
            predicted = pd.read_csv(destination / 'test.csv')
            if predicted.application_id.tolist() != request['test_ids']:
                raise ValueError('Foundation test prediction IDs differ')
            probability = predicted.default_probability.to_numpy()
            if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
                raise ValueError('Invalid reconstructed probabilities')
            predictions.append(probability)
        averaged = pd.Series(np.mean(predictions, axis=0), index=plan[0]['test_ids'])
        result = sample[['application_id']].copy()
        result['default_probability'] = result.application_id.map(averaged)
        if result.default_probability.isna().any():
            raise ValueError('Foundation sample coverage differs')
        result.to_csv(work / 'submission.csv', index=False)
        write_json(work / 'status.json', {'state': 'completed', 'source_run': str(folder),
                   'submission_sha256': sha256_file(work / 'submission.csv')})
        return result
    except BaseException as error:
        write_json(work / 'status.json', {'state': 'failed', 'error': str(error), 'source_run': str(folder)})
        raise


def predict_foundation(folder: Path, root: Path) -> pd.DataFrame:
    from logika_ui_26.gpu_runtime import gpu_slot

    with gpu_slot(root):
        return _predict_foundation(folder, root)
