# V242: Brier proper-score correction with unchanged representation

Same18-input V25 design and frozen globalKumo. Initialize beta with same-foldBCE head then minimize mean((sigmoid(KumoLogit+Xbeta)-y)^2)+.003/2||beta_nonintercept||^2 on6k. One fixed local optimization, no outer tuning. Inference eta=KumoLogit+.5Xbeta; final75:25spline77, mean5foldtest. Ridge scale held numerically fixed, not equal effective regularization across scoring losses.; mean5fold test probabilities

KS=0.5296911374; AUC=0.8471003683; deltaKS=-0.0005935226; paired95% CI=[-0.0021182686070507577, 0.0007866639400086692].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0014605249, deltaAUC=-0.0000084071, wins=2/5, deltaKS95% CI=[-0.0035432195636946827, 0.0008500093750388839].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0017683772, deltaAUC=-0.0000071761, wins=2/5, paired deltaKS CI=[-0.0038189137188849947, 0.0007713775659094275].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
