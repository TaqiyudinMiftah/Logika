# V190: Regularized worst-group regret V25 correction

g=employment x allocation-present; merge correction groups n<200 to OTHER; Rg(beta)=mean_g[BCE(zK+Xbeta)-BCE(zK)]; minimize .05*log(mean_g exp(Rg/.05))+.003/2||beta[1:]||^2; pK=sigmoid(zK+.5*Xbeta); final=.75*pK+.25*spline77 ; final=.75*sigmoid(eta)+.25*original_spline77; mean5fold test probabilities

KS=0.5290371425; AUC=0.8470924637; deltaKS=-0.0012475175; paired95% CI=[-0.0020406661282034223, 0.0004421305020880253].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.
