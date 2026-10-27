# V196: Stable9 spectral correction on local-context Kumo

Drop only2 dominant-cycle argmax inputs fromV191; fit27-input head with L2.003, shrink.5 on cached rawV178 Kumo; original75:25 spline fusion; mean5fold test probabilities

KS=0.5304604256; AUC=0.8471253428; deltaKS=+0.0001757656; paired95% CI=[-0.0016020396700091421, 0.0025223418835012964].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0006912366, deltaAUC=+0.0000165674, wins=2/5, deltaKS95% CI=[-0.0018003098783620692, 0.0007113851159139199].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, LARGER_TRAIN_VALIDATION_HEAD_GAP, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0009990890, deltaAUC=+0.0000177984, wins=1/5, paired deltaKS CI=[-0.0021057761480612, 0.000576188567054966].

Findings: BROAD_RANKING_GAIN_WITHOUT_KS_GAIN, CORRECTION_HEAD_GAP_REMAINS_HIGH, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
