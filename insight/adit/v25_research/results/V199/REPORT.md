# V199: Gaussian-rank preprocessing for V25 correction

On6k correction only: median impute, QuantileTransformer100quantiles normal output, quadratic3-knot spline, scale; missing indicators+employment; original offset BCE+L2.003 and shrink.5; original75:25 spline fusion; mean5fold test probabilities

KS=0.5299487417; AUC=0.8471200081; deltaKS=-0.0003359184; paired95% CI=[-0.0017760274384163916, 0.001407460557714637].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0012029206, deltaAUC=+0.0000112327, wins=2/5, deltaKS95% CI=[-0.0035501126728687826, 0.0010998797248061077].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0015107729, deltaAUC=+0.0000124638, wins=2/5, paired deltaKS CI=[-0.0036192712875069864, 0.000983248505372356].

Findings: BROAD_RANKING_GAIN_WITHOUT_KS_GAIN, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
