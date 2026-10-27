# V213: V193 anchor plus10% low-gap P-spline expert

Perfold p=.90*pV193+.10*pV200; fixed weights before own evaluation; mean5foldtest; mean5fold test probabilities

KS=0.5310916560; AUC=0.8471977728; deltaKS=+0.0008069960; paired95% CI=[-0.0012555206883598236, 0.003258413407700175].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0000600062, deltaAUC=+0.0000889974, wins=3/5, deltaKS95% CI=[-0.0008746124509978497, 0.0008070246311454738].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0003678585, deltaAUC=+0.0000902284, wins=2/5, paired deltaKS CI=[-0.001093994906349116, 0.0006571077333404775].

Findings: BROAD_RANKING_GAIN_WITHOUT_KS_GAIN, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
