# V231: Global anchor vote with all three population views

raw=mean(globalV168,rawV178,rawV222,rawV223)=.8875global+.0375each local view; refit original18-input6k correction, L2.003/shrink.5; final75:25spline77; mean5foldtest; mean5fold test probabilities

KS=0.5301237648; AUC=0.8470541786; deltaKS=-0.0001608953; paired95% CI=[-0.0006023972139401285, 0.000636488220779643].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0010278975, deltaAUC=-0.0000545968, wins=2/5, deltaKS95% CI=[-0.003197609456471359, 0.001256929382861861].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0013357498, deltaAUC=-0.0000533657, wins=2/5, paired deltaKS CI=[-0.003382036413606071, 0.0010537739962040269].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
