# V189: Inner-OOF hierarchical employment and history calibration

On inner OOF z fit additive b0+b_employment+b_history with meanBCE(z+delta)+.02/2*b0^2+.04/2*sum(b_group^2); history=zero/1_to3/over3/missing allocation. eta=z25+.5*clip(delta,-.25,.25); final=.75*sigmoid(eta)+.25*spline77; unseen levels use global fallback ; final=.75*sigmoid(eta)+.25*original_spline77; mean5fold test probabilities

KS=0.5306378083; AUC=0.8470252084; deltaKS=+0.0003531482; paired95% CI=[-0.0002700043004696867, 0.0010102756956592712].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.
