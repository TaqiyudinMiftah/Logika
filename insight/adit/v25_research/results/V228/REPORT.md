# V228: Conservative global/coverage/employment raw fusion

raw=mean(globalV168,rawV178,rawV222)=.90global+.05coverage+.05employment; refit original18-input6k correction, L2.003/shrink.5; final75:25spline77; mean5foldtest; mean5fold test probabilities

KS=0.5301357660; AUC=0.8470552063; deltaKS=-0.0001488940; paired95% CI=[-0.0006384380754945879, 0.0005555384514905309].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0010158963, deltaAUC=-0.0000535692, wins=2/5, deltaKS95% CI=[-0.0032763333354321215, 0.0012439970926382147].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0013237486, deltaAUC=-0.0000523381, wins=2/5, paired deltaKS CI=[-0.0034853225500312197, 0.0008805888611143101].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
