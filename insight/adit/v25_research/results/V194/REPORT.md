# V194: Inner-crossfitted spectral/local evidence combination

For each outer fold,3 inner splits of6k train base18/global, spectral29/global, local18/V178 heads on4k and predict held2k. Fit w_s,w_l in[0,1.5] by BCE(z_base+w_s*(z_s-z_base)+w_l*(z_l-z_base))+.02/2*((w_s-1)^2+(w_l-1)^2). Apply weights to full6k corrected-Kumo heads; final=.75*sigmoid(eta)+.25*original_spline77; mean5fold test probabilities

KS=0.5309350573; AUC=0.8471088815; deltaKS=+0.0006503973; paired95% CI=[-0.001302092156420609, 0.002804097867702555].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0002166049, deltaAUC=+0.0000001061, wins=1/5, deltaKS95% CI=[-0.0009156679015691366, 0.00038567648089507777].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, LARGER_TRAIN_VALIDATION_HEAD_GAP, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.
