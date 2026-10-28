"""Verify the exact SDK, CUDA/BF16, checkpoint identity and source adaptation."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import socket

import torch
from huggingface_hub import hf_hub_download
from sdm.models import KumoTabular
from kumo_cuda_cache import PINNED_FIT_SHA256
import inspect
import textwrap

revision = '329a707bbcf9ca147aa45a63fef8b2029f4ee37b'
direct = json.loads(metadata.distribution('structured-data-models').read_text('direct_url.json'))
assert direct['vcs_info']['commit_id'] == revision
assert torch.__version__.startswith('2.7.1')
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
assert hashlib.sha256(textwrap.dedent(inspect.getsource(KumoTabular.fit)).encode()).hexdigest() == PINNED_FIT_SHA256
x = torch.arange(32, device='cuda', dtype=torch.float32)
assert (x * 2).sum().item() == 992
assets = {}
for name in ['config.json', 'large/classifier.pt']:
    path = Path(hf_hub_download('nvidia/Kumo-Tabular', name, revision='v1.0.1', local_files_only=True))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    assets[name] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}
result = dict(hostname=socket.gethostname(), gpu=torch.cuda.get_device_name(),
              memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),
              sdk_revision=revision, assets=assets,
              versions={name: metadata.version(name) for name in ['torch','structured-data-models','numpy','pandas','pyarrow','scikit-learn']})
root = Path(__file__).resolve().parents[1]
(root / 'artifacts/node380/environment.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
