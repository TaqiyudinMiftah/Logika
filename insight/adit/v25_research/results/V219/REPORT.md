# V219: Bootstrap stability ensemble of reduced-knot local correction

3 class-stratified bootstrap draws of6k correction; each fitsV19827-input2-knot head on fixedrawV178; pK=mean3 corrected probabilities; final=.75*pK+.25*spline77; save variance of member logits; mean5fold test probabilities

KS=0.5301791684; AUC=0.8471111593; deltaKS=-0.0001054917; paired95% CI=[-0.0023874632320346396, 0.0024065116889883573].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0009724939, deltaAUC=+0.0000023839, wins=1/5, deltaKS95% CI=[-0.0033585533467148988, 0.001180454737713651].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0012803462, deltaAUC=+0.0000036150, wins=1/5, paired deltaKS CI=[-0.0035557141961930967, 0.001015001860062763].

Findings: BROAD_RANKING_GAIN_WITHOUT_KS_GAIN, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
