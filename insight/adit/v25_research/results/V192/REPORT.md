# V192: Joint context/spectral correction with selective group shrinkage

Same raw V178 Kumo and29 inputs asV191; meanBCE+.003/2*sum(base_beta^2)+.01/2*sum(spectral_beta^2), intercept unpenalized; inference shrink.5; original75:25 spline fusion; mean5fold test probabilities

KS=0.5306509745; AUC=0.8471070053; deltaKS=+0.0003663145; paired95% CI=[-0.0016661947805252542, 0.002735304837238835].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0005006877, deltaAUC=-0.0000017701, wins=3/5, deltaKS95% CI=[-0.0011305862307339626, 0.00034192014138960717].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, LARGER_TRAIN_VALIDATION_HEAD_GAP, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.
