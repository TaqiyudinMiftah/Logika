from pathlib import Path
import sys,json,hashlib
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
import torch,numpy as np
from cpu_models import Network
torch.set_num_threads(2);torch.manual_seed(20261030)
x=torch.randn(171,31);net=Network(31,64);net.eval();assert torch.equal(net(x),torch.zeros(171))
with torch.no_grad():net.head.bias.add_(.01);a=net(x);b=torch.cat([net(z)for z in x.split(13)]);err=float((a-b).abs().max());assert err<5e-6
a=net(x)
loss=torch.nn.functional.binary_cross_entropy_with_logits(a,torch.arange(171).remainder(2).float())+.1*a.square().mean();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all()for p in net.parameters())
s=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent=R.parent/'woe-feature-cann';assert s(R/'source/woe_features.py')==s(parent/'source/woe_features.py')
v=dict(status='PASS',zero_initialization=True,nonzero_input_attention_finite_gradient=True,batch_replay_max_error=err,encoder_unchanged_sha256=s(R/'source/woe_features.py'),inherited_own_label_exclusion_gate_sha256=s(parent/'SYNTHETIC_WOE_GATE.json'))
(R/'SYNTHETIC_STRUCTURE_GATE.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
