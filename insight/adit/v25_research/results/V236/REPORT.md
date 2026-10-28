# V236: Global, history-population and Small raw capacity fusion

raw=mean(globalLargeV168,rawV178,globalSmallV207)=.6166667globalLarge+.05coverage-local+.3333333Small; refit18-input6k correction L2.003/shrink.5; same75:25spline77; mean5foldtest; mean5fold test probabilities

KS=0.5302212892; AUC=0.8469280960; deltaKS=-0.0000633708; paired95% CI=[-0.00116132203556, 0.0016906304963091297].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0009303730, deltaAUC=-0.0001806794, wins=2/5, deltaKS95% CI=[-0.002723406137374207, 0.0012821471271001468].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V25, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0012382253, deltaAUC=-0.0001794484, wins=2/5, paired deltaKS CI=[-0.0028840337820921326, 0.0010767367250965965].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
