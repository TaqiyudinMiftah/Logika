# V198: Reduced-knot stable9 local-context correction

Same27 inputs/rawV178 asV196; quadratic spline2knots instead3, training-only knots/scales; L2.003 and shrink.5; original75:25 spline fusion; mean5fold test probabilities

KS=0.5293739929; AUC=0.8472601438; deltaKS=-0.0009106671; paired95% CI=[-0.002469983187527114, 0.001716625313401948].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0017776694, deltaAUC=+0.0001513684, wins=1/5, deltaKS95% CI=[-0.003709327102186205, 0.000505557617577747].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0020855217, deltaAUC=+0.0001525994, wins=1/5, paired deltaKS CI=[-0.0038164588276729887, 0.00029769562809539845].

Findings: BROAD_RANKING_GAIN_WITHOUT_KS_GAIN, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
