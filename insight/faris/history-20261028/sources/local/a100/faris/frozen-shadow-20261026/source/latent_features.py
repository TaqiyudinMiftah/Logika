"""One fixed multi-view variational model, trained only on the context partition."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import argparse,json,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from common import R,root_path,dump,sha

class Joint(nn.Module):
    def __init__(self,views):
        super().__init__();self.views=views
        self.enc=nn.ModuleList([nn.Sequential(nn.Linear(len(v)*2,48),nn.SiLU(),nn.Linear(48,16)) for v in views])
        self.dec=nn.ModuleList([nn.Sequential(nn.Linear(8,48),nn.SiLU(),nn.Linear(48,len(v))) for v in views])
        self.outcome=nn.Linear(8,1);self.historical=nn.Linear(8,1)
    def encode(self,x,mask):
        means=[];precisions=[]
        for ids,encoder in zip(self.views,self.enc):
            part=x[:,ids];observed=mask[:,ids];encoded=encoder(torch.cat([part,observed],1))
            mu,logvar=encoded[:,:8],encoded[:,8:].clamp(-5,5)
            availability=(observed.sum(1)>0).float()[:,None]
            means.append(mu);precisions.append(torch.exp(-logvar)*availability)
        precision=1+sum(precisions);var=1/precision;mean=var*sum(p*m for p,m in zip(precisions,means))
        return mean,var
    def forward(self,x,mask):
        mean,var=self.encode(x,mask);z=mean+torch.randn_like(mean)*var.sqrt()
        recon=0
        for ids,decoder in zip(self.views,self.dec):
            loss=((decoder(z)-x[:,ids])**2)*mask[:,ids]
            recon=recon+loss.sum()/mask[:,ids].sum().clamp(min=1)
        kl=.5*(mean.pow(2)+var-var.log()-1).mean()
        return mean,var,recon+.005*kl,self.outcome(mean).squeeze(1),self.historical(mean).squeeze(1)
def main(fold):
    start=time.time();torch.manual_seed(262026+fold);torch.set_num_threads(2)
    root=root_path('packet_root');current=pd.read_parquet(root/f'packets/fold{fold}/current.parquet');roles=np.load(root/f'packets/fold{fold}/roles.npz')
    ctx=roles['context'];columns=current.select_dtypes(include='number').columns.tolist()
    raw=current[columns].to_numpy(float);raw=np.sign(raw)*np.log1p(abs(raw));mask=np.isfinite(raw)
    mean=np.nanmean(raw[ctx],0);sd=np.nanstd(raw[ctx],0);mean=np.nan_to_num(mean);sd=np.maximum(np.nan_to_num(sd,nan=1),.05)
    x=np.clip((np.where(mask,raw,mean)-mean)/sd,-6,6).astype('float32');mask=mask.astype('float32')
    def view(c):
        if 'bureau' in c:return 0
        if any(s in c for s in ['repay','late_','allocation','loan_count','dpd_']):return 1
        if any(s in c for s in ['ecom','price','orders']):return 2
        return 3
    views=[[j for j,c in enumerate(columns) if view(c)==k] for k in range(4)];assert all(views)
    labels=pd.read_parquet(R/'prepared/current.parquet',columns=['customer_id'])
    historical=pd.read_parquet(R/'prepared/historical.parquet',columns=['customer_id','target12'])
    known=historical[historical.target12>=0].groupby('customer_id').target12.max()
    aux=labels.customer_id.iloc[:70000].map(known).to_numpy(float)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':torch.cuda.set_per_process_memory_fraction(.6)
    model=Joint(views).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    tx=torch.tensor(x[ctx],device=device);tm=torch.tensor(mask[ctx],device=device)
    ty=torch.tensor(roles['y_context'],dtype=torch.float32,device=device)
    th=torch.tensor(np.nan_to_num(aux[ctx]),dtype=torch.float32,device=device);available=torch.tensor(np.isfinite(aux[ctx]),device=device)
    rng=np.random.default_rng(262026+fold);trace=[]
    for epoch in range(40):
        model.train();total=0;order=rng.permutation(len(ctx))
        for startidx in range(0,len(ctx),1024):
            ids=order[startidx:startidx+1024];mu,var,rec,p,h=model(tx[ids],tm[ids])
            target_loss=nn.functional.binary_cross_entropy_with_logits(p,ty[ids])
            a=available[ids];history_loss=nn.functional.binary_cross_entropy_with_logits(h[a],th[ids][a]) if a.any() else h.sum()*0
            loss=rec+.25*target_loss+.25*history_loss;assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();total+=float(loss.detach())
        trace.append(total/int(np.ceil(len(ctx)/1024)))
    model.eval();representations=[]
    with torch.inference_mode():
        for first in range(0,70000,2048):
            mu,var=model.encode(torch.tensor(x[first:first+2048],device=device),torch.tensor(mask[first:first+2048],device=device))
            representations.append(np.column_stack([mu.cpu().numpy(),var.mean(1).cpu().numpy()]))
    features=np.concatenate(representations);assert features.shape==(70000,9) and np.isfinite(features).all()
    dest=R/f'learned/fold{fold}';dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(features,columns=[f'joint_z{j}' for j in range(8)]+['joint_posterior_variance']).to_parquet(dest/'latent.parquet',index=False)
    dump({'status':'complete','fold':fold,'model':'multi-view product-of-Gaussian-experts variational autoencoder + historical/current outcome tasks',
          'epochs':40,'seed':262026+fold,'device':device,'training_rows':50000,'outer_labels_loaded':False,'correction_labels_used':False,
          'columns':columns,'views':[[columns[j] for j in ids] for ids in views],'loss_first_last':[trace[0],trace[-1]],
          'loss_weights':{'reconstruction_per_view':1,'KL':.005,'current_default_BCE':.25,'historical12_BCE':.25},
          'latent_dimension_interpretation':'Not semantically identified; coordinates are learned, no claim of psychological traits.',
          'prediction_sha256':sha(dest/'latent.parquet'),'source_sha256':sha(__file__),'seconds':time.time()-start,'fitted_weights_saved':False},dest/'LATENT.json')
    print(json.dumps({'status':'complete','fold':fold,'seconds':time.time()-start}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);main(p.parse_args().fold)
