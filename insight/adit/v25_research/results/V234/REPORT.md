# V234: Half-strength employment-population context

raw=mean(globalV168,rawV222)=.925global+.075employment-local; refit18-input6k correction L2.003/shrink.5; same75:25spline77; mean5foldtest; mean5fold test probabilities

KS=0.5302485398; AUC=0.8470116126; deltaKS=-0.0000361202; paired95% CI=[-0.0005647083545683107, 0.0006577583079847512].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0009031225, deltaAUC=-0.0000971628, wins=2/5, deltaKS95% CI=[-0.003293658026052962, 0.0013596738809021306].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V25, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0012109748, deltaAUC=-0.0000959317, wins=2/5, paired deltaKS CI=[-0.0033514142832886144, 0.0009625858269274375].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
