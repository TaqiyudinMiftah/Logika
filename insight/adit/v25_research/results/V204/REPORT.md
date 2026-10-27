# V204: Gaussian-rank stable9 local-context correction

V19627 inputs andrawV178; training-only100-quantile Gaussian ranks before3-knot correction splines; L2.003, shrink.5; original75:25 spline fusion; mean5fold test probabilities

KS=0.5310554193; AUC=0.8470483310; deltaKS=+0.0007707592; paired95% CI=[-0.0013801372920634464, 0.0035223112352887912].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0000962430, deltaAUC=-0.0000604444, wins=3/5, deltaKS95% CI=[-0.00213614076105999, 0.0019130997688115079].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0004040953, deltaAUC=-0.0000592133, wins=3/5, paired deltaKS CI=[-0.0023489035781103064, 0.0019050596990610586].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
