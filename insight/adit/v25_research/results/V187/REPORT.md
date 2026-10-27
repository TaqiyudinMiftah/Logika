# V187: Delete-one-third bagging of V25 correction heads

Split6k correction into3 disjoint query blocks; train3 independent V25 heads on complementary4k rows; pK=mean_j sigmoid(logit(Kumo)+.5*X_j*beta_j); final=.75*pK+.25*spline77 ; final=.75*sigmoid(eta)+.25*original_spline77; mean5fold test probabilities

KS=0.5300516408; AUC=0.8470116318; deltaKS=-0.0002330193; paired95% CI=[-0.0007266945726761564, 0.000630720337094295].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.
