# V239: Group-sparse selection of complete risk-feature effects

Same original18 V25 spline design. Minimize meanBCE(offset+Xbeta)+.003/2||beta_nonintercept||^2+.001*sum_g sqrt(dim_g)||beta_g||_2. Each numeric spline block and its missing indicator is one group; employment dummies one group; intercept/logit not group-penalized. Convex proximal solver, KKT mapping checked. eta=offset+.5Xbeta; final75:25 original spline77; mean5foldtest.; mean5fold test probabilities

KS=0.5297967456; AUC=0.8473498662; deltaKS=-0.0004879145; paired95% CI=[-0.002319406890632358, 0.001563439154119995].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0013549167, deltaAUC=+0.0002410908, wins=2/5, deltaKS95% CI=[-0.004277687733946342, 0.0015515697531535897].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0016627690, deltaAUC=+0.0002423219, wins=2/5, paired deltaKS CI=[-0.0043952232199676675, 0.0013238492399999524].

Findings: BROAD_RANKING_GAIN_WITHOUT_KS_GAIN, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
