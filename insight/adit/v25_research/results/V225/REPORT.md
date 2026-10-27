# V225: Coverage and underwriting population contexts

rawKumo=.5rawV178+.5rawV223=.85global+.075coverage-local+.075APR-local; refit18-input correction and75:25spline77fusion; mean5fold test probabilities

KS=0.5304617367; AUC=0.8470881601; deltaKS=+0.0001770766; paired95% CI=[-0.0003906885921119263, 0.0009695278400486977].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0006899256, deltaAUC=-0.0000206153, wins=2/5, deltaKS95% CI=[-0.0030528855492827993, 0.0015516403921885776].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0009977779, deltaAUC=-0.0000193842, wins=2/5, paired deltaKS CI=[-0.0032634499491557166, 0.0012993400596196631].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
