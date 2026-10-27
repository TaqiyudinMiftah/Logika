# V230: Conservative global/employment/underwriting raw fusion

raw=mean(globalV168,rawV222,rawV223)=.90global+.05employment+.05APR; refit original18-input6k correction, L2.003/shrink.5; final75:25spline77; mean5foldtest; mean5fold test probabilities

KS=0.5300727012; AUC=0.8470227584; deltaKS=-0.0002119588; paired95% CI=[-0.0006856759413433083, 0.0006063562166892073].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0010789611, deltaAUC=-0.0000860170, wins=2/5, deltaKS95% CI=[-0.003291519597717324, 0.0012096031421466608].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V25, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0013868134, deltaAUC=-0.0000847860, wins=2/5, paired deltaKS CI=[-0.0034564950859027137, 0.0010174867891659216].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
