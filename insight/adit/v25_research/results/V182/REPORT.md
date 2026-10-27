# V182: 64-landmark Nystrom RBF residual on existing V25 domain features

eta=eta_V25+.25*Nystrom64_RBF(robust_scaled8inputs+missing)@gamma; fitBCE+.05/2||gamma||Â²+.2/2 mean(deltaÂ²) ; final=.75*sigmoid(eta)+.25*original_spline77; mean5fold test probabilities

KS=0.5303342526; AUC=0.8470281278; deltaKS=+0.0000495925; paired95% CI=[-0.00015708847692229023, 0.00023739523537769363].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.
