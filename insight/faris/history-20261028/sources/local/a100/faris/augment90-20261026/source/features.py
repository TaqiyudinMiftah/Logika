"""Lazy, typed feature formulas. Population quantities use context only."""
from pathlib import Path
import hashlib, json, os
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
BANK = Path(os.environ.get('LOGIKA_FEATURE_BANK','/mnt/home/faris/faris/logika-meta-experiments/meta-20261016-v1/bank'))

def ratio(a, b):
    a,b=np.broadcast_arrays(np.asarray(a,dtype=float),np.asarray(b,dtype=float))
    return np.divide(a,b,out=np.full(a.shape,np.nan),where=np.isfinite(b)&(b>0))

class Features:
    def __init__(self, fold):
        self.fold=fold
        self.manifest=json.loads((R/'BANK_MANIFEST.json').read_text())
        self.catalog=json.loads((R/'CATALOG.json').read_text())
        self.specs={s['id']:s for s in self.catalog['feature_definitions']}
        self.lookup={n:j for j,n in enumerate(self.manifest['columns'])}
        self.numeric=np.load(BANK/'numeric.npy',mmap_mode='r')
        self.cats=pd.read_parquet(BANK/'categorical.parquet').iloc[:70000]
        self.current=pd.read_parquet(R/f'packets/fold{fold}/current.parquet')
        self.extra=pd.read_parquet(R/'extra.parquet')
        self.ix=np.load(R/f'packets/fold{fold}/roles.npz')
        self.ctx=self.ix['context'];self.cache={};self.prior_cache={}
        self.legacy={p['name']:p for p in self.manifest['priors'] if p.get('legacy')}
    def base(self,name):
        if name in self.cache:return self.cache[name]
        if name in self.current:
            v=self.current[name].to_numpy()
        elif name in self.extra:v=self.extra[name].to_numpy()
        elif name in self.legacy:
            p=self.legacy[name];k,n=self.base(p['k']),self.base(p['n']);mu=self.cohort(p['k'],p['n'])
            v=ratio(k+p['strength']*mu,n+p['strength'])
        elif name in self.lookup:v=np.asarray(self.numeric[:70000,self.lookup[name]],dtype=float)
        elif name in self.cats:v=self.cats[name].to_numpy()
        else:raise KeyError(name)
        self.cache[name]=v
        return v
    def cohort(self,k,n):
        key=(k,n)
        if key not in self.prior_cache:
            self.prior_cache[key]=float(np.nansum(self.base(k)[self.ctx])/max(np.nansum(self.base(n)[self.ctx]),1))
            assert 0<=self.prior_cache[key]<=1
        return self.prior_cache[key]
    def compute(self,s):
        x=[self.base(n) for n in s['inputs']];op=s['op']
        with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
            if op=='identity':return x[0]
            if op=='ratio':v=ratio(x[0],x[1])
            elif op=='net_ratio':v=ratio(x[0]-np.maximum(x[1],0),x[2])
            elif op=='stress':
                income,cash,expense,payment=x
                deficit=np.maximum((1+s['expense_increase'])*expense+payment-(1-s['income_loss'])*income,0)
                v=ratio(np.maximum(s['months']*deficit-np.maximum(cash,0),0),income)
            elif op=='amort_balance':
                principal,annual,payment,income=x;r=annual/12;n=s['months']
                accum=np.expm1(n*np.log1p(r))
                annuity=np.divide(accum,r,out=np.full_like(r,float(n)),where=r!=0)
                v=ratio(principal*(1+r)**n-payment*annuity,income)
            elif op=='interest_income':v=ratio(12*x[1]-x[0],x[2])
            elif op=='payment_shock':
                principal,annual,payment,income=x;r=(annual+s['rate_shift'])/12
                proposed=ratio(principal*r,-np.expm1(-12*np.log1p(r)))
                v=ratio(proposed-payment,income)
            elif op=='beta':
                k,n=x;strength=s['strength'];mu=self.cohort(*s['inputs'])
                k=np.nan_to_num(k,nan=0);n=np.nan_to_num(n,nan=0)
                posterior=(k+strength*mu)/(n+strength)
                if s['quantity']=='posterior':v=posterior
                elif s['quantity']=='sd':v=np.sqrt(np.maximum(posterior*(1-posterior)/(n+strength+1),0))
                else:v=n/(n+strength)
            elif op=='relative_change':v=ratio(x[0]-x[1],np.abs(x[1]))
            elif op=='dispersion':v=ratio(x[0],np.abs(x[1]))
            else:raise ValueError(op)
        v=np.asarray(v,dtype=np.float32);v[~np.isfinite(v)]=np.nan
        return v
    def pack(self,pack):
        self.cache={}
        values={};dropped={};seen={}
        for fid in pack['features']:
            v=self.compute(self.specs[fid]);context=pd.Series(v[self.ctx])
            if context.nunique(dropna=False)<=1:
                dropped[fid]='constant_or_all_missing_in_context';continue
            digest=hashlib.sha256(pd.util.hash_pandas_object(context,index=False).values.tobytes()).hexdigest()
            if digest in seen:
                dropped[fid]='exact_context_duplicate_of_'+seen[digest];continue
            seen[digest]=fid;values[fid]=v
        frame=pd.DataFrame(values)
        assert 0<len(frame.columns)<=100 and len(frame)==70000
        return frame,dropped

def formula_gate():
    assert np.allclose(ratio([6,1],[3,0]),[2,np.nan],equal_nan=True)
    f=Features(0)
    f.cache={'p':np.array([120.,120.]),'r':np.array([0.,.12]),'pay':np.array([10.,10.]),'i':np.array([20.,20.])}
    z=f.compute({'op':'amort_balance','inputs':['p','r','pay','i'],'months':6})
    expected=(120*1.01**6-10*((1.01**6-1)/.01))/20
    assert abs(z[0]-3)<1e-6 and abs(z[1]-expected)<1e-6
    f.cache={'income':np.array([100.,100.]),'cash':np.array([0.,100.]),'expense':np.array([70.,70.]),'payment':np.array([20.,20.])}
    z=f.compute({'op':'stress','inputs':['income','cash','expense','payment'],'income_loss':.2,'expense_increase':0.,'months':3})
    assert np.allclose(z,[.3,0])
    return {'ratio_zero_handling':True,'zero_rate_amortization':True,'positive_rate_amortization':True,'cash_stress':True}
