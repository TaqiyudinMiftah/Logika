# V211: Frozen Small original18 correction with reduced knots

Same cachedSmall90/50kcontext asV207,18-input head with quadratic2knots ratherthan3, L2.003/shrink.5; same75:25spline77fusion; mean5fold test probabilities

KS=0.5280516858; AUC=0.8463724706; deltaKS=-0.0022329742; paired95% CI=[-0.004447726671625643, 0.0007101457377447345].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0030999764, deltaAUC=-0.0007363048, wins=0/5, deltaKS95% CI=[-0.005503443089144669, 8.696599808512184e-06].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V25, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0034078287, deltaAUC=-0.0007350737, wins=0/5, paired deltaKS CI=[-0.005827850922636468, -0.00017899909716104085].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
