"""Training-only robust feature maps; missing observations never become measured zero."""
import numpy as np
from scipy import sparse
from scipy.special import ndtri
from cpu_offset import FrozenInput,_hash
class BookInput(FrozenInput):
 def __init__(self,mode='asinh'):self.mode=mode;assert mode in {'asinh','rank_normal','both'}
 def fit(self,frame,rows):
  super().fit(frame,rows);a=self._numeric(frame,rows);self.scales=[];self.knots=[];self.quantiles=[]
  for j in range(a.shape[1]):
   x=a[:,j];x=x[np.isfinite(x)];nz=np.abs(x[x!=0]);self.scales.append(float(np.median(nz))if len(nz)else 1.)
   q=np.quantile(x,np.linspace(0,1,257))if len(x)else np.zeros(257);knots,counts=np.unique(q,return_counts=True);self.knots.append(knots);self.quantiles.append(.001+.998*(np.cumsum(counts)-counts/2)/257)
  self.scales=np.array(self.scales);self.scales[self.scales<1e-12]=1.
  z=self._maps(a);self.map_mean=np.nanmean(z,axis=0);self.map_scale=np.nanstd(z,axis=0);self.map_mean[~np.isfinite(self.map_mean)]=0.;self.map_scale[~np.isfinite(self.map_scale)|(self.map_scale<1e-8)]=1.
  return self
 def _maps(self,a):
  values=[]
  if self.mode in {'asinh','both'}:values.append(np.arcsinh(a/self.scales))
  if self.mode in {'rank_normal','both'}:
   mapped=np.full(a.shape,np.nan)
   for j,knots in enumerate(self.knots):
    good=np.isfinite(a[:,j]);q=np.interp(a[good,j],knots,self.quantiles[j])if len(knots)>1 else np.full(good.sum(),.5);mapped[good,j]=ndtri(np.clip(q,.001,.999))
   values.append(mapped)
  return np.column_stack(values)
 def glm(self,frame,rows):
  raw=super().glm(frame,rows);z=(self._maps(self._numeric(frame,rows))-self.map_mean)/self.map_scale;z=np.clip(z,-10,10);z[~np.isfinite(z)]=0.;return sparse.hstack([raw,sparse.csr_matrix(z)],format='csr')
 def recipe(self):
  value=super().recipe();value.update(feature_engineering_mode=self.mode,derived_numeric_features=len(self.numeric)*(2 if self.mode=='both'else 1),robust_scale_sha256=_hash(self.scales),quantile_knots_sha256=[_hash(x)for x in self.knots],derived_mean_sha256=_hash(self.map_mean),derived_scale_sha256=_hash(self.map_scale),fit_on_inner_covariates_only=True,missing_indicators_retained=True);return value
