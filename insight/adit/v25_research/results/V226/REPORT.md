# V226: Occupation and underwriting population contexts

rawKumo=.5rawV222+.5rawV223=.85global+.075employment-local+.075APR-local; refit18-input correction and75:25spline77fusion; mean5fold test probabilities

KS=0.5299228021; AUC=0.8470163891; deltaKS=-0.0003618579; paired95% CI=[-0.0008201181522495243, 0.0008237221775467402].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0012288601, deltaAUC=-0.0000923863, wins=2/5, deltaKS95% CI=[-0.0032913200261424024, 0.0013777631249669326].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V25, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0015367124, deltaAUC=-0.0000911552, wins=2/5, paired deltaKS CI=[-0.0036481839531791794, 0.0011025975671088593].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
