# V201: Stabilized Jeffreys-information correction

Minimize meanBCE+.003/2||beta_nonintercept||^2-.5/n*logdet(X^T*W(beta)*X/n+.0001I); original V25 design and shrink.5; original75:25 spline fusion; mean5fold test probabilities

KS=0.5307039316; AUC=0.8470460045; deltaKS=+0.0004192716; paired95% CI=[-0.00028357342876210445, 0.0008566673540344627].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0004477306, deltaAUC=-0.0000627709, wins=2/5, deltaKS95% CI=[-0.0029467968906535026, 0.0014987680960927448].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0007555829, deltaAUC=-0.0000615399, wins=2/5, paired deltaKS CI=[-0.0030974449133375852, 0.001439291897899586].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
