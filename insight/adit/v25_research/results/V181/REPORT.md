# V181: Fixed equal average of V167 spectral and V146 tree-complemented V25

p_final=.5*p_V167+.5*p_V146; fixed matched-fold score average; mean5fold test probabilities

KS=0.5300242737; AUC=0.8471525273; deltaKS=-0.0002603864; paired95% CI=[-0.0016284660590027006, 0.0018709925573478351].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.
