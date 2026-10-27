# V205: Reduced-knot original18-input V25 correction

Same18 original V25 correction inputs andglobal frozenKumo; quadratic spline2knots rather than3; L2.003, shrink.5; original75:25 spline fusion; mean5fold test probabilities

KS=0.5299121554; AUC=0.8471723010; deltaKS=-0.0003725046; paired95% CI=[-0.001952731918901801, 0.0012779638182463691].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0012395068, deltaAUC=+0.0000635256, wins=2/5, deltaKS95% CI=[-0.0038875138287104483, 0.0008511924593308655].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0015473591, deltaAUC=+0.0000647567, wins=2/5, paired deltaKS CI=[-0.00400442177928054, 0.0004866661748973088].

Findings: BROAD_RANKING_GAIN_WITHOUT_KS_GAIN, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
