# V200: P-spline-inspired coefficient roughness correction

V25 head objective meanBCE+.003/2||beta_nonintercept||^2+.03/2||D2*(beta_spline/scale)||^2; difference penalty within each numeric spline block; shrink.5; original75:25 spline fusion; mean5fold test probabilities

KS=0.5295657798; AUC=0.8473194858; deltaKS=-0.0007188802; paired95% CI=[-0.0027445167325058804, 0.0018535647120008212].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0015858824, deltaAUC=+0.0002107104, wins=2/5, deltaKS95% CI=[-0.0041523645660126954, 0.001037174909750017].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0018937347, deltaAUC=+0.0002119414, wins=1/5, paired deltaKS CI=[-0.004212917868210564, 0.0008201758371722072].

Findings: BROAD_RANKING_GAIN_WITHOUT_KS_GAIN, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
