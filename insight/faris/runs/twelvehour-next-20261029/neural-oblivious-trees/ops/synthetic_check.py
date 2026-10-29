import sys,torch,numpy as np,json
sys.path.insert(0,'/mnt/home/faris/logika-obltree-synthetic-20261029')
import cpu_models as m
from torch.autograd import gradcheck
torch.set_num_threads(2);torch.manual_seed(7731)
z=torch.randn(7,11,dtype=torch.float64,requires_grad=True);assert gradcheck(m.sparsemax,(z,),eps=1e-6,atol=1e-5)
out=m.sparsemax(z);assert torch.allclose(out.sum(-1),torch.ones(7,dtype=torch.float64))and torch.all(out>=0)
x=torch.randn(4096,13);x[:,0]=0;records=[]
for route in ['sigmoid','sparsemoid']:
 for learn in [False,True]:
  s=dict(trees=16,depth=3,learn_selection=learn,route=route);init=m.initialization(x,s,2026);net=m.model(13,s,2026,init)
  assert net(x[:100]).abs().max()==0
  path=net.routes(x[:100]);assert torch.allclose(path.sum(-1),torch.ones(100,16),atol=1e-6)
  with torch.no_grad():net.leaf.copy_(torch.randn_like(net.leaf))
  net(x[:100]).square().mean().backward();assert all(torch.isfinite(p.grad).all()for p in net.parameters()if p.requires_grad)
  assert (net.selection_logits.grad is not None)==learn
  a=net(x[:128]).detach();b=torch.cat([net(x[:64]),net(x[64:128])]).detach();e=float((a-b).abs().max());assert e<=5e-6
  records.append(dict(route=route,learn_selection=learn,batch_replay=e,path_sum_error=float((path.sum(-1)-1).abs().max())))
print(json.dumps(dict(status='PASS',sparsemax_double_gradient_check=True,zero_response=True,constant_feature_initialization=True,records=records)))
