# V186: Inner-crossfitted restrained affine calibration before fixed spline fusion

3-fold inner OOF corrected Kumo on6k; fit a,b by meanBCE(a+b*logit(p_inner))+.02/2*(a^2+(b-1)^2), a in[-.5,.5], b in[.5,1.5]; pK=sigmoid(z25+.5*(a+(b-1)*z25)); final=.75*pK+.25*spline77 ; final=.75*sigmoid(eta)+.25*original_spline77; mean5fold test probabilities

KS=0.5306076887; AUC=0.8470329634; deltaKS=+0.0003230286; paired95% CI=[-0.00021540810763042982, 0.0006579051502370005].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.
