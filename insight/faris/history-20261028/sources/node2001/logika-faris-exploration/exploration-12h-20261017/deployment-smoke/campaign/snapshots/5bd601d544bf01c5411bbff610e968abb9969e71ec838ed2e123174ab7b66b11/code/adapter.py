import numpy as np,joblib
from sklearn.linear_model import LogisticRegression
from scipy.stats import ks_2samp
def run_trial(ctx):
 x=np.random.default_rng(ctx.seed).normal(size=(256,8));y=(x[:,0]+x[:,1]>0).astype(int);m=LogisticRegression(C=ctx.params['C']).fit(x[:192],y[:192]);joblib.dump(m,ctx.path('model.joblib'));ctx.model('model.joblib');np.savez_compressed(ctx.path('validation.npz'),y=y[192:],prediction=m.predict_proba(x[192:])[:,1]);return {'validation':['validation.npz']}
def evaluate(ctx,result):
 z=np.load(ctx.path('validation.npz'));p=z['prediction'];y=z['y'];return {'score':float(ks_2samp(p[y==0],p[y==1]).statistic)}
