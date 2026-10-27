# V209: Conservative Small/Large capacity-diversity blend

Perfold final=.90*pV193+.10*pV207, fixed weights before seeingSmall results; test probabilities averaged after per-fold blending; mean5fold test probabilities

KS=0.5315002809; AUC=0.8471027000; deltaKS=+0.0012156208; paired95% CI=[-0.0011157525985110133, 0.0031707158500331426].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=+0.0003486186, deltaAUC=-0.0000060754, wins=2/5, deltaKS95% CI=[-0.0006987261661526356, 0.0010102810424907876].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=+0.0000407663, deltaAUC=-0.0000048443, wins=2/5, paired deltaKS CI=[-0.0009711864445887618, 0.0008398786941456672].

Findings: KS_GAIN_WITH_AUC_TRADEOFF, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
