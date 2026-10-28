# V235: Half-strength underwriting-population context

raw=mean(globalV168,rawV223)=.925global+.075APR-local; refit18-input6k correction L2.003/shrink.5; same75:25spline77; mean5foldtest; mean5fold test probabilities

KS=0.5303567840; AUC=0.8470355664; deltaKS=+0.0000721240; paired95% CI=[-0.0005539760464484539, 0.0007726091194470818].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0007948782, deltaAUC=-0.0000732090, wins=2/5, deltaKS95% CI=[-0.0031635828289614865, 0.0015325246868734062].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0011027305, deltaAUC=-0.0000719779, wins=2/5, paired deltaKS CI=[-0.0034392615522394862, 0.001288386531777501].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
