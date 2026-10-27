# V193: V185 plus bounded inner-OOF partial-pooling calibration residual

z=logit(pV185); d=.5*clip(logit(pV189)-logit(pV25),-.1,.1); p=sigmoid(z+d); per-fold test fusion then mean5fold; mean5fold test probabilities

KS=0.5314595146; AUC=0.8471075443; deltaKS=+0.0011748545; paired95% CI=[-0.00112906081881107, 0.0033472886158873905].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=+0.0003078523, deltaAUC=-0.0000012311, wins=4/5, deltaKS95% CI=[-0.0003895990991073767, 0.0007626677076973815].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.
