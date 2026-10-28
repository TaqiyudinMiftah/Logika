# V238: Natural cubic offset correction with linear tails

Original18 V25 inputs, signedlog and training-median imputation; per numeric input restricted cubic basis with correction-training quantile knots(.05,.35,.65,.95), linear tails, standardized basis plus missing and employment indicators. Repeated knots use linear-only fallback. eta=logit(globalKumo)+.5Xbeta; fit meanBCE+.003/2||beta_nonintercept||^2 on6k; final=.75sigmoid(eta)+.25spline77, mean5foldtest.; mean5fold test probabilities

KS=0.5287981226; AUC=0.8471927546; deltaKS=-0.0014865374; paired95% CI=[-0.0028105301495477742, 0.0006301145802009334].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0023535396, deltaAUC=+0.0000839792, wins=1/5, deltaKS95% CI=[-0.004650500520554488, 0.00031790591242565294].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0026613919, deltaAUC=+0.0000852102, wins=1/5, paired deltaKS CI=[-0.004922584086451021, 0.000164353777737693].

Findings: BROAD_RANKING_GAIN_WITHOUT_KS_GAIN, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
