# V215: Smooth evidence reliability while retaining risk-level thresholds

V25 original18/globalKumo; BCE+.003/2||beta_nonintercept||^2+.03/2||D2*(beta_spline/scale)||^2 only on evidence inputs; other bases unchanged; shrink.5 and original75:25spline77fusion; mean5fold test probabilities

KS=0.5295894328; AUC=0.8471970164; deltaKS=-0.0006952272; paired95% CI=[-0.0024568305035144707, 0.0014619671585224914].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0015622294, deltaAUC=+0.0000882410, wins=2/5, deltaKS95% CI=[-0.004113942103961835, 0.0011523850891518906].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0018700817, deltaAUC=+0.0000894721, wins=2/5, paired deltaKS CI=[-0.004225944054013072, 0.0008846048967332968].

Findings: BROAD_RANKING_GAIN_WITHOUT_KS_GAIN, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
