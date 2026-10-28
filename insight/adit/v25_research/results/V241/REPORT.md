# V241: Feature-leverage weighted offset correction

Original18 spline design. On6k only, b=encoded covariates excluding intercept/Kumo; S=cov(b)+.1I; d2=(b-mean)'inv(S)(b-mean); t=training95th percentile(d2); w=clip(sqrt(max(t,1e-12)/max(d2,1e-12)),.25,1). Fit sum w*BCE/sum w+.003/2||beta_nonintercept||^2. No query-time weight/gate. eta=Kumo logit+.5Xbeta; final75:25spline77, mean5foldtest.; mean5fold test probabilities

KS=0.5302003018; AUC=0.8469968971; deltaKS=-0.0000843582; paired95% CI=[-0.000703452721787834, 0.0007804419668717114].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0009513604, deltaAUC=-0.0001118783, wins=2/5, deltaKS95% CI=[-0.003378316366410969, 0.0014164019380805662].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V25, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0012592127, deltaAUC=-0.0001106472, wins=2/5, paired deltaKS CI=[-0.003521745036255028, 0.0011928729496436831].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
