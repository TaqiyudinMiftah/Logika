"""Fold-local preprocessing and frozen, small probe learners."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from .common import identity

class Preprocessor:
    def __init__(self,bank,representation,model,seed=2026):
        self.bank=bank;self.rep=representation;self.c=representation['config'];self.model=model;self.seed=seed
    def fit_transform(self,rows,y):
        from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,PowerTransformer,QuantileTransformer,StandardScaler,RobustScaler,SplineTransformer
        self.fit_rows=np.asarray(rows);self.names,self.cats=self.bank.columns(self.rep)
        x,self.numeric_names=self.bank.cohort(self.fit_rows,self.fit_rows,self.names,self.c)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);self.fill=np.nanmedian(x,axis=0)
        self.fill=np.nan_to_num(self.fill);self.group_fill={}
        if self.c['missing']=='zero':self.fill=np.zeros(x.shape[1],dtype=np.float32)
        if self.c['missing']=='group':
            groups=self.bank.categorical.employment_type.iloc[rows].to_numpy()
            for g in np.unique(groups):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning);v=np.nanmedian(x[groups==g],axis=0)
                self.group_fill[str(g)]=np.where(np.isfinite(v),v,self.fill)
        self.indicators=np.flatnonzero(np.isnan(x).any(axis=0)) if self.c['indicator'] else np.array([],dtype=int)
        x=self._fill(x,rows)
        self.low=self.high=None
        if self.c['clip']:
            self.low,self.high=np.quantile(x,[self.c['clip'],1-self.c['clip']],axis=0);x=np.clip(x,self.low,self.high)
        self.active=np.flatnonzero(np.ptp(x,axis=0)>1e-10)
        if not len(self.active):self.active=np.array([0])
        x=x[:,self.active];self.feature_names=[self.numeric_names[i] for i in self.active]
        self.transformer=None
        method=self.c['transform']
        if method=='signed_log':x=np.sign(x)*np.log1p(np.abs(x))
        elif method=='yeo':self.transformer=PowerTransformer(standardize=False);x=self.transformer.fit_transform(x)
        elif method in ['quantile_normal','rank']:
            self.transformer=QuantileTransformer(n_quantiles=min(256,len(x)),output_distribution='normal' if method=='quantile_normal' else 'uniform',subsample=min(10000,len(x)),random_state=self.seed)
            x=self.transformer.fit_transform(x)
        self.scaler=None
        if self.c['scale']!='none' or self.model=='linear' or self.c['reducer']!='none':
            self.scaler=RobustScaler() if self.c['scale']=='robust' else StandardScaler()
            x=self.scaler.fit_transform(x)
        self.category_model=None;self.category_maps=[];self.category_prior=float(np.mean(y))
        c=self.bank.categorical.iloc[rows][self.cats].astype(str)
        encoding=self.c['encoding']
        if len(self.cats):
            if encoding=='onehot':
                self.category_model=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
                xc=self.category_model.fit_transform(c);catnames=self.category_model.get_feature_names_out(self.cats).tolist()
            elif encoding=='ordinal':
                self.category_model=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1,dtype=np.float32)
                xc=self.category_model.fit_transform(c);catnames=self.cats
            else:
                from sklearn.model_selection import StratifiedKFold
                xc=np.zeros((len(rows),len(self.cats)),dtype=np.float32);catnames=self.cats
                splits=list(StratifiedKFold(3,shuffle=True,random_state=self.seed).split(np.zeros(len(y)),y)) if encoding=='target' else []
                for j,name in enumerate(self.cats):
                    values=c[name].to_numpy()
                    if encoding=='frequency':
                        mapping=pd.Series(values).value_counts(normalize=True).to_dict();xc[:,j]=pd.Series(values).map(mapping).to_numpy()
                    else:
                        for ti,vi in splits:
                            tab=pd.DataFrame({'v':values[ti],'y':np.asarray(y)[ti]}).groupby('v').y.agg(['sum','count'])
                            m=((tab['sum']+20*np.mean(np.asarray(y)[ti]))/(tab['count']+20)).to_dict()
                            xc[vi,j]=pd.Series(values[vi]).map(m).fillna(np.mean(np.asarray(y)[ti])).to_numpy()
                        tab=pd.DataFrame({'v':values,'y':y}).groupby('v').y.agg(['sum','count'])
                        mapping=((tab['sum']+20*self.category_prior)/(tab['count']+20)).to_dict()
                    self.category_maps.append(mapping)
            x=np.column_stack([x,xc]);self.feature_names+=list(catnames)
        if len(self.indicators):
            original,_=self.bank.cohort(self.fit_rows,rows,self.names,self.c)
            x=np.column_stack([x,np.isnan(original[:,self.indicators]).astype(np.float32)])
            self.feature_names += [self.numeric_names[i]+'_missing' for i in self.indicators]
        self.selected=np.arange(x.shape[1]);selection=self.c['select'];count=self.c['feature_count'];k=x.shape[1] if count=='all' else min(int(count),x.shape[1])
        if selection in ['gain','true_gain']:
            from lightgbm import LGBMClassifier
            m=LGBMClassifier(n_estimators=60,num_leaves=15,n_jobs=1,verbosity=-1,random_state=self.seed,importance_type='gain' if selection=='true_gain' else 'split');m.fit(x,y)
            self.selected=np.argsort(-m.feature_importances_,kind='stable')[:k]
        elif selection=='mi':
            from sklearn.feature_selection import mutual_info_classif
            # Fixed training-only sample bounds the estimator cost, not CV coverage.
            ii=np.random.default_rng(self.seed).choice(len(y),min(6000,len(y)),replace=False)
            score=mutual_info_classif(x[ii],np.asarray(y)[ii],random_state=self.seed,n_jobs=1)
            self.selected=np.argsort(-score,kind='stable')[:k]
        elif selection=='variance' or (count!='all' and selection=='none'):self.selected=np.argsort(-np.var(x,axis=0),kind='stable')[:k]
        elif selection=='correlation':
            corr=np.nan_to_num(np.corrcoef(x,rowvar=False));keep=[]
            for j in range(x.shape[1]):
                if not keep or np.max(np.abs(corr[j,keep]))<self.c['correlation']:keep.append(j)
                if len(keep)>=k:break
            self.selected=np.array(keep,dtype=int)
        x=x[:,self.selected];self.feature_names=[self.feature_names[i] for i in self.selected]
        self.reducer=None;self.group_reducers=[];r=self.c['reducer'];d=min(self.c['dimensions'],x.shape[1],len(x)-1)
        if r=='pca':
            from sklearn.decomposition import PCA
            self.reducer=PCA(d,svd_solver='randomized',random_state=self.seed)
        elif r=='svd':
            from sklearn.decomposition import TruncatedSVD
            self.reducer=TruncatedSVD(d,random_state=self.seed)
        elif r=='ica':
            from sklearn.decomposition import FastICA
            self.reducer=FastICA(d,max_iter=200,tol=.001,random_state=self.seed)
        elif r=='projection':
            from sklearn.random_projection import GaussianRandomProjection
            self.reducer=GaussianRandomProjection(d,random_state=self.seed)
        elif r=='nmf':
            from sklearn.decomposition import NMF
            self.nmf_min=x.min(axis=0);self.reducer=NMF(d,init='nndsvda',max_iter=120,random_state=self.seed)
        elif r=='pls':
            from sklearn.cross_decomposition import PLSRegression
            self.reducer=PLSRegression(min(d,8),max_iter=200)
        elif r=='cluster':
            from sklearn.cluster import MiniBatchKMeans
            self.reducer=MiniBatchKMeans(n_clusters=d,batch_size=1024,max_iter=100,n_init=3,random_state=self.seed)
        elif r=='group_pca':
            from sklearn.decomposition import PCA
            from .features import lineage
            parts=[]
            for source in ['application','profile','previous','repayment','bureau','account','shopping']:
                ix=[i for i,n in enumerate(self.feature_names) if source in self.bank.manifest['info'].get(n,{'sources':lineage(n)})['sources']]
                if not ix:continue
                p=PCA(min(d,len(ix)),svd_solver='randomized',random_state=self.seed);parts.append(p.fit_transform(x[:,ix]));self.group_reducers.append((ix,p))
            reduced=np.column_stack(parts)
        if self.reducer:
            input_x=np.maximum(x-self.nmf_min,0) if r=='nmf' else x
            if r=='pls':self.reducer.fit(input_x,y);reduced=self.reducer.transform(input_x)
            else:reduced=self.reducer.fit_transform(input_x)
            if r=='ica' and self.reducer.n_iter_>=200:raise RuntimeError('ICA did not converge')
        if self.reducer or self.group_reducers:x=np.column_stack([x,reduced]) if self.c['append'] else reduced
        self.spline=None
        if self.c['spline']:
            self.spline=SplineTransformer(n_knots=5,degree=3,include_bias=False)
            self.spline_columns=min(64,x.shape[1]);z=self.spline.fit_transform(x[:,:self.spline_columns]);x=np.column_stack([x,z])
        self.output_features=x.shape[1]
        self.audit={'fit_rows_hash':identity(self.fit_rows.tolist()),'fit_rows':len(rows),'selected_features':self.feature_names,
                    'output_features':self.output_features,'target_encoding_inner_folds':3 if encoding=='target' else 0}
        return np.nan_to_num(x,nan=0,posinf=1e10,neginf=-1e10).astype(np.float32)
    def _fill(self,x,rows):
        if self.group_fill:
            groups=self.bank.categorical.employment_type.iloc[rows].to_numpy()
            for g in np.unique(groups):
                ix=groups==g;x[ix]=np.where(np.isnan(x[ix]),self.group_fill.get(str(g),self.fill),x[ix])
        return np.where(np.isnan(x),self.fill,x)
    def transform(self,rows):
        x,_=self.bank.cohort(self.fit_rows,rows,self.names,self.c);ind=np.isnan(x[:,self.indicators]).astype(np.float32)
        x=self._fill(x,rows)
        if self.low is not None:x=np.clip(x,self.low,self.high)
        x=x[:,self.active]
        if self.c['transform']=='signed_log':x=np.sign(x)*np.log1p(np.abs(x))
        if self.transformer:x=self.transformer.transform(x)
        if self.scaler:x=self.scaler.transform(x)
        if self.cats:
            cat=self.bank.categorical.iloc[rows][self.cats].astype(str)
            if self.category_model:xc=self.category_model.transform(cat)
            else:
                fallback=self.category_prior if self.c['encoding']=='target' else 0
                xc=np.column_stack([cat[n].map(m).fillna(fallback).to_numpy() for n,m in zip(self.cats,self.category_maps)])
            x=np.column_stack([x,xc])
        if len(self.indicators):x=np.column_stack([x,ind])
        x=x[:,self.selected]
        if self.reducer:
            reduced=self.reducer.transform(np.maximum(x-self.nmf_min,0) if self.c['reducer']=='nmf' else x)
            x=np.column_stack([x,reduced]) if self.c['append'] else reduced
        elif self.group_reducers:
            reduced=np.column_stack([p.transform(x[:,ix]) for ix,p in self.group_reducers]);x=np.column_stack([x,reduced]) if self.c['append'] else reduced
        if self.spline:x=np.column_stack([x,self.spline.transform(x[:,:self.spline_columns])])
        return np.nan_to_num(x,nan=0,posinf=1e10,neginf=-1e10).astype(np.float32)

def fit_probe(model,x,y,c,seed,device='cpu'):
    weight=float(c['weight']);sample_weight=None
    if c.get('cohort_weight'):
        # Downweight dense covariate cohorts using training-only numeric quantiles.
        bins=np.digitize(x[:,0],np.unique(np.quantile(x[:,0],[.2,.4,.6,.8])))
        counts=np.bincount(bins);sample_weight=np.clip(len(y)/(len(counts)*counts[bins]),.25,4)
    if model=='lgb':
        from lightgbm import LGBMClassifier
        kwargs=dict(n_estimators=250,num_leaves=31,learning_rate=.035,min_child_samples=200,reg_lambda=40,
                    scale_pos_weight=weight,n_jobs=1,verbosity=-1,random_state=seed,max_bin=63)
        if device=='opencl':kwargs.update(device_type='gpu',gpu_platform_id=0,gpu_device_id=0,gpu_use_dp=False)
        m=LGBMClassifier(**kwargs)
    elif model=='cat':
        from catboost import CatBoostClassifier
        kwargs=dict(iterations=300,depth=5,learning_rate=.035,l2_leaf_reg=25,class_weights=[1,weight],thread_count=1,random_seed=seed,verbose=False,allow_writing_files=False)
        if device=='cuda':kwargs.update(task_type='GPU',devices='0',gpu_ram_part=.60,pinned_memory_size='128mb')
        m=CatBoostClassifier(**kwargs)
    elif model=='xgb':
        from xgboost import XGBClassifier
        m=XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.035,min_child_weight=80,reg_lambda=50,
                        scale_pos_weight=weight,n_jobs=1,random_state=seed,tree_method='hist',device='cuda:0' if device=='cuda' else 'cpu',max_bin=128,verbosity=0)
    else:
        from sklearn.linear_model import LogisticRegression
        m=LogisticRegression(C=.05,max_iter=1000,tol=.001,class_weight={0:1,1:weight},solver='lbfgs',random_state=seed)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always');m.fit(x,y,sample_weight=sample_weight)
    messages=[str(w.message) for w in captured]
    if model=='linear' and any('converge' in w.lower() for w in messages):raise RuntimeError('Logistic probe did not converge: '+'; '.join(messages))
    return m,weight,messages

def predict(m,x,weight):
    p=np.asarray(m.predict_proba(x)[:,1],dtype=float)
    return np.clip(p/(weight*(1-p)+p),1e-8,1-1e-8)

def fit_predict(bank,recipe,fitrows,y,queryrows,device='cpu',seed=None):
    seed=recipe['seed'] if seed is None else seed
    pre=Preprocessor(bank,recipe['representation'],recipe['model'],seed)
    x=pre.fit_transform(fitrows,y);m,w,warnings_=fit_probe(recipe['model'],x,y,pre.c,seed,device)
    trainp=predict(m,x,w);p=predict(m,pre.transform(queryrows),w)
    return p,pre,m,w,np.quantile(trainp,np.linspace(0,1,257)),warnings_
