# V179: Group-ridge spectral correction: preserve V25 penalty, shrink new family10x

V25 offset design plus9 spectral inputs; meanBCE+.003/2||base_beta||Â²+.03/2||new_beta||Â²; inference strength.5 ; final=.75*sigmoid(eta)+.25*original_spline77; mean5fold test probabilities

KS=0.5297030656; AUC=0.8470628048; deltaKS=-0.0005815944; paired95% CI=[-0.001949261912155209, 0.0019387949894558523].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.
