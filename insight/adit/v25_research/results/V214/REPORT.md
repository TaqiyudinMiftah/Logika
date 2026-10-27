# V214: Smooth capacity effects while retaining sharp credit-risk responses

V25 original18/globalKumo; BCE+.003/2||beta_nonintercept||^2+.03/2||D2*(beta_spline/scale)||^2 only on capacity inputs; other bases unchanged; shrink.5 and original75:25spline77fusion; mean5fold test probabilities

KS=0.5282367731; AUC=0.8470604557; deltaKS=-0.0020478869; paired95% CI=[-0.0030235060139275185, 0.0003586031687522379].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0029148891, deltaAUC=-0.0000483197, wins=2/5, deltaKS95% CI=[-0.005001703239716174, -0.00015726918244114897].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0032227414, deltaAUC=-0.0000470886, wins=1/5, paired deltaKS CI=[-0.005156845426477863, -0.0003144019120019661].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
