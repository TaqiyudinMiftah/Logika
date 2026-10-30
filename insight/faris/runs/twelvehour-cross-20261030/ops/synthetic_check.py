"""Algebra, derivative, kernel and matched-control checks on synthetic inputs."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'source'))
from cross_models import CrossLayer, Network

torch.set_num_threads(2)
torch.manual_seed(193)
layer = CrossLayer(2, 1, 1).double()
with torch.no_grad():
    layer.down[0].weight.copy_(torch.tensor([[.2, .3]], dtype=torch.float64))
    layer.up[0].weight.copy_(torch.tensor([[.5], [.7]], dtype=torch.float64))
    layer.up[0].bias.copy_(torch.tensor([.1, -.2], dtype=torch.float64))
answer = layer(torch.tensor([[1., 2.]], dtype=torch.float64),
               torch.tensor([[.4, -.5]], dtype=torch.float64))
assert torch.max(torch.abs(answer - torch.tensor([[.465, -.998]], dtype=torch.float64))) < 1e-12
for experts in [1, 4]:
    layer = CrossLayer(3, 2, experts).double()
    inputs = tuple(torch.randn(2, 3, dtype=torch.float64, requires_grad=True) for _ in range(2))
    assert torch.autograd.gradcheck(layer, inputs, atol=1e-6, rtol=1e-4)

base = dict(width=64, rank=8, cross_layers=2, rff_components=2048, gamma_multiplier=1.)
for kind in ['cross_linear', 'cross_moe', 'rff', 'mlp', 'localglm']:
    torch.manual_seed(19)
    model = Network(5, dict(base, kind=kind), 19)
    model.eval()
    x = torch.randn(128, 5)
    assert torch.equal(model(x), torch.zeros(128))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), (torch.arange(128) % 2).float())
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert torch.isfinite(model(x)).all()
    assert torch.max(torch.abs(model(x) - torch.cat([model(x[:64]), model(x[64:])]))) < 1e-5

spec = dict(base, kind='rff')
first, second = Network(5, spec, 101), Network(5, spec, 101)
assert torch.equal(first.frequencies, second.frequencies) and torch.equal(first.phases, second.phases)
small=Network(5,dict(spec,rff_components=128),101)
assert torch.equal(small.frequencies,first.frequencies[:,:128]) and torch.equal(small.phases,first.phases[:128])
x = torch.tensor([[0., 0., 0., 0., 0.], [.2, -.1, .3, -.4, .5], [1., 0., 1., 0., 1.]])
z = torch.cos(x @ first.frequencies + first.phases) * first.feature_scale
expected = torch.exp(-torch.cdist(x, x).square() / 5.)
kernel_error = float(torch.max(torch.abs(z @ z.T - expected)))
assert kernel_error < .08

# The LocalGLM control must reproduce the previous architecture and one optimizer step.
old = Path('/mnt/home/faris/logika-overnight-refit-20261030/localglm-cann/source/cpu_models.py')
assert old.is_file()
loader = importlib.util.spec_from_file_location('previous_localglm_control', old)
module = importlib.util.module_from_spec(loader)
loader.loader.exec_module(module)
torch.manual_seed(31)
previous = module.Network(5, 64)
torch.manual_seed(31)
current = Network(5, dict(base, kind='localglm'), 31)
assert set(previous.state_dict()) == set(current.state_dict())
assert all(torch.equal(previous.state_dict()[k], current.state_dict()[k]) for k in previous.state_dict())
x = torch.randn(16, 5)
y = (torch.arange(16) % 2).float()
for model in [previous, current]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    torch.manual_seed(77)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
    optimizer.step()
assert all(torch.equal(previous.state_dict()[k], current.state_dict()[k]) for k in previous.state_dict())
result = dict(status='PASS', synthetic_only=True, real_training_rows=0,
              linear_cross_known_case=True, single_and_mixture_input_gradcheck=True,
              zero_head_and_finite_optimizer=True, deterministic_RFF_basis=True,
              RFF_kernel_max_error=kernel_error, previous_LocalGLM_one_step_exact=True,
              source_sha256=hashlib.sha256((R / 'source/cross_models.py').read_bytes()).hexdigest())
(R / 'SYNTHETIC_CHECKS.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
