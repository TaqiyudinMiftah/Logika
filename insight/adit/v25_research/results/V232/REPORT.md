# V232: Frozen Small/Large raw fusion before correction

raw=.50globalLargeV168+.50globalSmallV207; refit original18-input6k correction, L2.003/shrink.5; final75:25spline77; mean5foldtest; mean5fold test probabilities

KS=0.5301855915; AUC=0.8467773258; deltaKS=-0.0000990685; paired95% CI=[-0.001625352507693853, 0.0018504648972674466].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0009660707, deltaAUC=-0.0003314497, wins=4/5, deltaKS95% CI=[-0.0030497000548337814, 0.0011774119123801078].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V25, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0012739230, deltaAUC=-0.0003302186, wins=2/5, paired deltaKS CI=[-0.0032999584628564337, 0.0010291578791924035].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
