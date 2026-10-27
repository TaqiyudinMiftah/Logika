# V183: Strongly regularized shallow LightGBM residual on fixed V25 logits

eta=eta_V25+.15*clip(GBDT80_depth2_residual,-.5,.5); tree training init_score=eta_V25; fixed trees, no outer early stopping ; final=.75*sigmoid(eta)+.25*original_spline77; mean5fold test probabilities

KS=0.5299917510; AUC=0.8470201310; deltaKS=-0.0002929090; paired95% CI=[-0.0008159494493746977, 0.0004980755744682621].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.
