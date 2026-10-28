"""Architectures consume the existing frozen columns; no source features are built."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import QuantileTransformer


class Preprocessor:
    def fit(self, frame, rows):
        self.fit_rows=np.asarray(rows).copy()
        self.num=[c for c in frame if pd.api.types.is_numeric_dtype(frame[c])]
        self.cat=[c for c in frame if c not in self.num]
        x=frame[self.num].to_numpy(float);x[~np.isfinite(x)]=np.nan
        self.median=np.nanmedian(x[rows],axis=0);self.median=np.nan_to_num(self.median)
        filled=np.where(np.isfinite(x[rows]),x[rows],self.median)
        self.quantile=QuantileTransformer(n_quantiles=min(256,len(rows)),output_distribution='normal',subsample=None,random_state=261027).fit(filled)
        self.vocabs={c:{v:i+1 for i,v in enumerate(sorted(frame[c].iloc[rows].astype('string').fillna('__MISSING__').unique()))}for c in self.cat}
        self.cards=[len(self.vocabs[c])+1 for c in self.cat]
        return self

    def transform(self,frame):
        x=frame[self.num].to_numpy(float);missing=~np.isfinite(x)
        z=self.quantile.transform(np.where(missing,self.median,x)).astype('float32')
        z=np.concatenate([np.clip(z,-5,5),missing.astype('float32')],axis=1)
        cats=np.column_stack([frame[c].astype('string').fillna('__MISSING__').map(self.vocabs[c]).fillna(0).to_numpy('int64')for c in self.cat]) if self.cat else np.empty((len(frame),0),dtype='int64')
        assert np.isfinite(z).all()
        return z,cats

    def recipe(self):
        return {'numeric_columns':self.num,'categorical_columns':self.cat,'categorical_cardinalities':self.cards,
                'numeric_transform':'context-only median,256quantile Gaussian mapping clipped[-5,5], missing mask',
                'categorical_transform':'context-only sorted vocabulary; unknownindex0',
                'learned_source_feature_values_changed':False,'internal_missing_masks':'model preprocessing, not new frozen source columns',
                'fit_rows_count':len(self.fit_rows)}


class FlatInput(nn.Module):
    def __init__(self,n_num,cards,embedding=8):
        super().__init__();self.embeddings=nn.ModuleList([nn.Embedding(c,embedding)for c in cards]);self.width=n_num+len(cards)*embedding

    def forward(self,x,cat):
        return torch.cat([x]+[e(cat[:,i])for i,e in enumerate(self.embeddings)],dim=1)


class ResidualBlock(nn.Module):
    def __init__(self,width,dropout):
        super().__init__();self.net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),nn.ReLU(),nn.Dropout(dropout),nn.Linear(width*2,width),nn.Dropout(dropout))

    def forward(self,x):return x+self.net(x)


class DenseArchitecture(nn.Module):
    def __init__(self,n_num,cards,spec):
        super().__init__();self.input=FlatInput(n_num,cards,spec['cat_embedding_dim']);self.kind=spec['kind'];width=spec['width'];d=self.input.width
        if self.kind=='resnet':
            self.backbone=nn.Sequential(nn.Linear(d,width),*[ResidualBlock(width,spec['dropout'])for _ in range(spec['blocks'])],nn.LayerNorm(width),nn.ReLU())
            self.head=nn.Linear(width,1)
        elif self.kind=='mlp_control':
            modules=[]
            for _ in range(spec['blocks']):modules.extend([nn.Linear(d,width),nn.ReLU(),nn.Dropout(spec['dropout'])]);d=width
            self.backbone=nn.Sequential(*modules);self.head=nn.Linear(width,1)
        else:
            self.cross=nn.ModuleList([nn.Sequential(nn.Linear(d,32,bias=False),nn.Linear(32,d))for _ in range(3)])
            self.deep=nn.Sequential(nn.Linear(d,width),nn.ReLU(),nn.Dropout(spec['dropout']),nn.Linear(width,width),nn.ReLU(),nn.Dropout(spec['dropout']))
            self.head=nn.Linear(width+d if self.kind=='dcn_parallel'else width,1)

    def forward(self,x,cat):
        z=self.input(x,cat)
        if self.kind in ['resnet','mlp_control']:return self.head(self.backbone(z))
        crossed=z
        for layer in self.cross:crossed=crossed+z*layer(crossed)
        deep=self.deep(z if self.kind=='dcn_parallel'else crossed)
        return self.head(torch.cat([crossed,deep],dim=1)if self.kind=='dcn_parallel'else deep)


class FeatureTransformer(nn.Module):
    """FT-style tokenizer and pre-normalized Transformer; explicitly an adaptation."""
    def __init__(self,n_num,cards,spec):
        super().__init__();d=spec['width'];self.numeric_weight=nn.Parameter(torch.empty(n_num,d));self.numeric_bias=nn.Parameter(torch.zeros(n_num,d));nn.init.normal_(self.numeric_weight,std=.02)
        self.embeddings=nn.ModuleList([nn.Embedding(c,d)for c in cards]);self.cat_bias=nn.Parameter(torch.zeros(len(cards),d));self.cls=nn.Parameter(torch.zeros(1,1,d))
        layer=nn.TransformerEncoderLayer(d_model=d,nhead=4,dim_feedforward=d*2,dropout=spec['dropout'],activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=spec['blocks'],enable_nested_tensor=False);self.head=nn.Sequential(nn.LayerNorm(d),nn.ReLU(),nn.Linear(d,1))

    def forward(self,x,cat):
        num=x.unsqueeze(-1)*self.numeric_weight+self.numeric_bias
        parts=[self.cls.expand(len(x),-1,-1),num]
        if self.embeddings:parts.append(torch.stack([e(cat[:,i])for i,e in enumerate(self.embeddings)],dim=1)+self.cat_bias)
        return self.head(self.encoder(torch.cat(parts,dim=1))[:,0])


class OfficialTabM(nn.Module):
    def __init__(self,n_num,cards,spec):
        super().__init__();import tabm
        self.model=tabm.TabM.make(n_num_features=n_num,cat_cardinalities=cards,d_out=1,n_blocks=spec['blocks'],d_block=spec['width'],dropout=spec['dropout'],k=spec['ensemble_members'],arch_type='tabm')

    def forward(self,x,cat):return self.model(x,cat if cat.shape[1]else None).squeeze(-1)


def make_model(n_num,cards,spec):
    if spec['kind']=='tabm':return OfficialTabM(n_num,cards,spec)
    if spec['kind']=='ft_transformer':return FeatureTransformer(n_num,cards,spec)
    return DenseArchitecture(n_num,cards,spec)


def training_loss(logits,targets,pair_weight):
    if logits.ndim==1:logits=logits[:,None]
    loss=nn.functional.binary_cross_entropy_with_logits(logits,targets[:,None].expand_as(logits))
    if pair_weight:
        positive=torch.nonzero(targets>0,as_tuple=True)[0];negative=torch.nonzero(targets==0,as_tuple=True)[0]
        if len(positive)and len(negative):
            a=positive[torch.randint(len(positive),(1024,),device=logits.device)];b=negative[torch.randint(len(negative),(1024,),device=logits.device)]
            loss=loss+pair_weight*nn.functional.softplus(-(logits[a]-logits[b])).mean()
    return loss
