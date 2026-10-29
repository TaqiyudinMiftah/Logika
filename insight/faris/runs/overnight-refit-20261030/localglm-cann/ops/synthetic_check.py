import sys,json,torch,numpy as np,importlib.util
sys.path.insert(0,'/mnt/home/faris/logika-softks-synthetic-20261029');import cpu_models as n
from torch.autograd import gradcheck
torch.set_num_threads(2)
z=torch.tensor([.8,.1,-.3,-.9],dtype=torch.float64,requires_grad=True);t=torch.tensor(-.1,dtype=torch.float64,requires_grad=True);y=torch.tensor([1.,1.,0.,0.],dtype=torch.float64)
assert gradcheck(lambda a,b:n.ks_reward(a,y,b,.25,.5),(z,t),eps=1e-6,atol=1e-5)
r=n.ks_reward(z,y,t,.25,.5);g=torch.autograd.grad(r,z)[0];assert torch.all(g[:2]>0)and torch.all(g[2:]<0)
p=torch.sigmoid((z-t)/.25);assert abs(float(r.detach())-float((p[:2].mean()-p[2:].mean()).detach()))<1e-12
assert torch.isfinite(n.ks_reward(z[:2],y[:2],t,.25,.5))
constant=np.ones(100)*-2;labels=np.r_[np.zeros(80),np.ones(20)];assert np.isfinite(n.initial_threshold(labels,constant))
sp=importlib.util.spec_from_file_location('old','/mnt/home/faris/logika-twelvehour-next-20261029/neural-cann/source/cpu_models.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
torch.manual_seed(2026);a=old.Network(7,64);torch.manual_seed(2026);b=n.Network(7,64,.3)
for k,v in a.named_parameters():assert torch.equal(v,dict(b.named_parameters())[k])
a.eval();b.eval();x=torch.randn(128,7);labels=torch.randint(0,2,(128,)).float();oa=torch.optim.AdamW(a.parameters(),lr=.0003,weight_decay=.01);ob=torch.optim.AdamW([{'params':list(b.body.parameters())+list(b.head.parameters()),'weight_decay':.01},{'params':[b.threshold],'weight_decay':0.}],lr=.0003)
for net,opt in [(a,oa),(b,ob)]:
 opt.zero_grad();v=net(x);loss=torch.nn.functional.binary_cross_entropy_with_logits(v,labels)+.1*v.square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
for k,v in a.named_parameters():assert torch.equal(v,dict(b.named_parameters())[k])
assert torch.equal(a(x),b(x));assert float(b(x).abs().max())>0
print(json.dumps(dict(status='PASS',reward_double_gradient_check=True,reward_class_mean_identity=True,reward_gradient_direction=True,singleclass_minibatch_finite=True,constant_offset_threshold_finite=True,lambda0_BCE_one_step_bitwise_network_agreement=True,scope='Synthetic math/initialization/singleoptimizerstep only, not fullfit reproduction')))
