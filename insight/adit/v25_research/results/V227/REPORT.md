# V227: Three population views with a fixed global anchor

rawKumo=mean(rawV178,rawV222,rawV223)=.85global+.05each coverage/employment/APR local; refit18-input correction and75:25spline77fusion; mean5fold test probabilities

KS=0.5300681716; AUC=0.8470602453; deltaKS=-0.0002164884; paired95% CI=[-0.0007359496338272848, 0.0006788803357726275].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0010834906, deltaAUC=-0.0000485301, wins=2/5, deltaKS95% CI=[-0.0033611675126492793, 0.0011989942893248287].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0013913429, deltaAUC=-0.0000472990, wins=2/5, paired deltaKS CI=[-0.003475643915491403, 0.0010374452073002086].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
