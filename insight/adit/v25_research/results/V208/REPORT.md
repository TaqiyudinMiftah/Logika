# V208: Frozen Small plus stable9 correction information

ReuseV207 frozenSmall context predictions; fit27-input correction adding stable9 spectral inputs, L2.003/shrink.5; retain original75:25spline77fusion; mean5fold test probabilities

KS=0.5291365605; AUC=0.8462325436; deltaKS=-0.0011480995; paired95% CI=[-0.0036901516253799753, 0.0017176163464593561].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0020151017, deltaAUC=-0.0008762318, wins=1/5, deltaKS95% CI=[-0.004283288518277725, 0.000567938014715402].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V25, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0023229540, deltaAUC=-0.0008750007, wins=1/5, paired deltaKS CI=[-0.004339635515899073, 0.00033165805015634953].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
