"""Synthetic GPU gradient and cached-inference acceptance test; no borrower rows."""
import os,torch,numpy as np,pyarrow as pa
from sdm import TableTensor
from sdm.models import KumoTabular
from kumo_cuda_cache import keep_fit_cache_on_cuda
rng=np.random.default_rng(6)
def x(n):return TableTensor.from_arrow(pa.table({f'f{i}':pa.array(rng.normal(size=n)) for i in range(8)}),stypes={f'f{i}':'numerical' for i in range(8)},device='cuda')
def y(n):return TableTensor.from_arrow(pa.table({'target':pa.array(np.arange(n)%2,type=pa.int64())}),stypes={'target':'categorical'},device='cuda')
m=KumoTabular(task='classification',size='large',pretrained=True,device='cuda');keep_fit_cache_on_cuda(m);m.train()
with torch.amp.autocast('cuda',dtype=torch.bfloat16):
 o=m(x(64),y(64),x(16),num_estimators=1);p=o[['1']].numerical.squeeze(-1).float();assert p.requires_grad
loss=torch.nn.functional.binary_cross_entropy(p.clamp(1e-6,1-1e-6),torch.as_tensor(np.arange(16)%2,device='cuda',dtype=torch.float32))
loss.backward();assert any(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters());m.zero_grad(set_to_none=True);m.eval()
with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.bfloat16):m.fit(x(64),y(64),num_estimators=2);o=m.predict(x(16))
import cloudpickle,tempfile
with tempfile.TemporaryFile() as f:
 cloudpickle.dump(m._cache['recipe_execution'],f);f.seek(0);restored=cloudpickle.load(f);assert type(restored)==type(m._cache['recipe_execution'])
assert torch.isfinite(o.numerical).all();print('PASS synthetic Kumo Large GPU gradient, backward and two-estimator cached inference',torch.cuda.get_device_name(),torch.cuda.max_memory_allocated())
