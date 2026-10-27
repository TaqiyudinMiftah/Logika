# V216: Smooth credit-level responses only

V25 original18/globalKumo; add.03/2 D2 penalty only to bureau-score and late1/late30 posterior spline coefficients; other responses originalL2.003; shrink.5 and75:25spline77fusion; mean5fold test probabilities

KS=0.5304712619; AUC=0.8470885687; deltaKS=+0.0001866018; paired95% CI=[-0.0009745427768269199, 0.001543397360189467].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0006804004, deltaAUC=-0.0000202067, wins=2/5, deltaKS95% CI=[-0.003175329644033492, 0.0012439122388558925].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0009882527, deltaAUC=-0.0000189756, wins=2/5, paired deltaKS CI=[-0.003355316644399245, 0.0009159010755896533].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
