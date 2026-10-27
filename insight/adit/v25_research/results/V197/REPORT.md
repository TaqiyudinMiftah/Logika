# V197: Stable9 spectral correction with original global Kumo

Drop only2 dominant-cycle argmax inputs fromV167; fit27-input head on original global Kumo with L2.003, shrink.5; original75:25 spline fusion; mean5fold test probabilities

KS=0.5303465303; AUC=0.8470262986; deltaKS=+0.0000618702; paired95% CI=[-0.0017702064341767964, 0.002580705679907496].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0008051320, deltaAUC=-0.0000824768, wins=4/5, deltaKS95% CI=[-0.002090092152268155, 0.0008933270135349092].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, LARGER_TRAIN_VALIDATION_HEAD_GAP, AUC_DECREASE_VS_V25, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0011129843, deltaAUC=-0.0000812458, wins=2/5, paired deltaKS CI=[-0.0021541768411002388, 0.0007591078915685017].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CORRECTION_HEAD_GAP_REMAINS_HIGH, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
