"""Pinned isolated CUDA environment; no pretrained checkpoint downloads."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

R = Path('/root/logika-twelvehour-cross-20261030')
clock = json.loads((R / 'CLOCK.json').read_text())


def call(args, maximum):
    remaining = clock['admission_end_epoch'] - time.time() - 20
    assert remaining > 0
    subprocess.run(args, check=True, timeout=min(maximum, remaining))


try:
    if shutil.which('uv') is None:
        call([sys.executable, '-m', 'pip', 'install', '--quiet', 'uv'], 240)
    uv = shutil.which('uv')
    if not (R / 'env311/bin/python').exists():
        call([uv, 'venv', '--python', '3.11', '--seed', str(R / 'env311')], 300)
    python = str(R / 'env311/bin/python')
    call([uv, 'pip', 'install', '--python', python, '--index-url',
          'https://download.pytorch.org/whl/cu126', 'torch==2.7.1'], 1200)
    call([uv, 'pip', 'install', '--python', python, 'numpy==2.2.6', 'pandas==2.3.3',
          'pyarrow==24.0.0', 'scipy==1.16.3', 'scikit-learn==1.7.2'], 600)
    code = '''import json,torch,importlib.metadata,hashlib
from pathlib import Path
r=Path('/root/logika-twelvehour-cross-20261030');assert torch.cuda.is_available()
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in json.load(open(r/'SOURCE_SEAL.json')).items())
v=dict(status='PASS',GPU=torch.cuda.get_device_name(0),GPU_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
versions={n:importlib.metadata.version(n)for n in ['torch','numpy','pandas','pyarrow','scipy','scikit-learn']},pretrained_checkpoint_used=False)
(r/'ENVIRONMENT_LIVE.json').write_text(json.dumps(v,indent=2)+'\\n');print(json.dumps(v))
'''
    call([python, '-c', code], 120)
except Exception:
    (R / 'SETUP_FAILURE.json').write_text(json.dumps(dict(epoch=time.time(),
        traceback=traceback.format_exc(),no_scientific_fit_started=True), indent=2) + '\n')
    raise
