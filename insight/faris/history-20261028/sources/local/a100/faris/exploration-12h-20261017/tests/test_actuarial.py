import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from actuarial import BinomialGLM,CredibilityScorecard,reliability


def fixture():
 rng=np.random.default_rng(88);x=rng.normal(size=(600,8));p=1/(1+np.exp(-(-1+x[:,0]-.7*x[:,1])));y=rng.binomial(1,p);return x,y


def test_binomial_links_finite_and_directional():
 x,y=fixture()
 for link in ['logit','cloglog']:
  m=BinomialGLM(link=link,strength=.5).fit(x[:450],y[:450]);p=m.predict_proba(x[450:]);assert np.isfinite(p).all() and ((p>0)&(p<1)).all();assert m.optimization_.success
  from sklearn.metrics import roc_auc_score
  assert roc_auc_score(y[450:],p[:,1])>.7
  # Query covariates cannot modify the retained training parameters.
  before=m.coef_.copy();m.predict_proba(x[450:]*1000);np.testing.assert_equal(before,m.coef_)


def test_scorecard_crossfits_supervised_maps_and_missing_bucket(monkeypatch):
 x,y=fixture();x[::7,0]=np.nan;calls=[];method=CredibilityScorecard._maps
 def checked(self,a,b):calls.append((len(a),len(b)));return method(self,a,b)
 monkeypatch.setattr(CredibilityScorecard,'_maps',checked)
 m=CredibilityScorecard(credibility=True,strength=48).fit(x[:450],y[:450]);assert calls==[(300,300)]*3+[(450,450)];assert set(m.crossfit_)=={0,1,2}
 p=m.predict_proba(x[450:]);assert np.isfinite(p).all();assert np.isfinite(m.predict_proba(np.full((2,8),np.nan))).all()
 # Empty-bin credibility shrinks to the training prior, with finite uncertainty.
 maps=method(m,np.full((30,1),np.nan),np.array([0,1]*15));assert np.isfinite(m._apply(np.array([[999.],[np.nan]]),maps)).all()


def test_probability_diagnostics_are_separate_from_discrimination():
 y=np.array([0,0,1,1]);p=np.array([.1,.2,.7,.8]);a=reliability(y,p);assert abs(a['brier']-.045)<1e-12;assert a['observed_default_rate']==.5
