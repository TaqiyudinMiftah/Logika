# V229: Conservative global/coverage/underwriting raw fusion

raw=mean(globalV168,rawV178,rawV223)=.90global+.05coverage+.05APR; refit original18-input6k correction, L2.003/shrink.5; final75:25spline77; mean5foldtest; mean5fold test probabilities

KS=0.5301045249; AUC=0.8470707616; deltaKS=-0.0001801352; paired95% CI=[-0.00043134637751922057, 0.0007143018777388568].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0010471374, deltaAUC=-0.0000380138, wins=2/5, deltaKS95% CI=[-0.003228404467676077, 0.0013710591979960657].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0013549897, deltaAUC=-0.0000367827, wins=2/5, paired deltaKS CI=[-0.003309041860285278, 0.0011376307465287502].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
