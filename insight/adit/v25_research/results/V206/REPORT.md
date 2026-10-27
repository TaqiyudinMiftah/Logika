# V206: Roughness-regularized stable9 local-context correction

V19627-input rawV178 head; meanBCE+.003/2||beta_nonintercept||^2+.03/2||D2*(beta_spline/scale)||^2; shrink.5; original75:25 spline fusion; mean5fold test probabilities

KS=0.5288514727; AUC=0.8474009176; deltaKS=-0.0014331873; paired95% CI=[-0.003148985761432735, 0.0017016249295863694].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0023001896, deltaAUC=+0.0002921422, wins=1/5, deltaKS95% CI=[-0.004017488218465268, 0.0006520301560396362].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0026080419, deltaAUC=+0.0002933733, wins=1/5, paired deltaKS CI=[-0.004217142734552032, 0.00045749399227835096].

Findings: BROAD_RANKING_GAIN_WITHOUT_KS_GAIN, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
