# V188: Baseline-orthogonalized spectral innovation residual

B=(X25^T*X25/n+.01I)^(-1)*X25^T*S9/n; R=S9-X25*B; fit gamma by BCE(z25+R*gamma)+.03/2||gamma||^2+.2/2 mean((R*gamma)^2); eta=z25+.25*R*gamma; final=.75*sigmoid(eta)+.25*spline77 ; final=.75*sigmoid(eta)+.25*original_spline77; mean5fold test probabilities

KS=0.5300801729; AUC=0.8470658060; deltaKS=-0.0002044872; paired95% CI=[-0.00057572904383133, 0.0009781365866330047].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.
