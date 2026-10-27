# V180: Reliability-weighted spectral residual with frozen V25 correction

eta=eta_V25+.25*w*phi_spectral@gamma; w=I(spectral_supported)*n_completed90/(n_completed90+10); fitBCE+.03/2||gamma||Â²+.2/2 mean((w*phi@gamma)^2) ; final=.75*sigmoid(eta)+.25*original_spline77; mean5fold test probabilities

KS=0.5302846600; AUC=0.8470484353; deltaKS=+0.0000000000; paired95% CI=[-0.0005244142841181298, 0.0006621718348871458].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.
