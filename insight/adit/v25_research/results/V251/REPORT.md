# V251: Fixed V25 with orthogonalized item-history support

Keep the original V25 head beta0 fixed. On identical6k correction rows fit signedlog standardization for U; missing values are0 in centered basis with explicit shared support flags. Fit A=lstsq(Xbase,U,rcond=1e-6); R=U-Xbase*A. Do not rescale residuals. eta0=logit(frozenKumo)+.5*Xbase*beta0. Fit gamma minimizing meanBCE(eta0+R*gamma)+.01/2||gamma||^2. Inference eta=eta0+.25*R*gamma; final=.75sigmoid(eta)+.25original_spline77; mean5fold test. Projection is training-only; query targets are unused.; mean5fold test probabilities

KS=0.5301166426; AUC=0.8469813748; deltaKS=-0.0001680174; paired95% CI=[-0.001104132977172001, 0.0007584250185840384].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0010350197, deltaAUC=-0.0001274007, wins=2/5, deltaKS95% CI=[-0.003407915208800849, 0.0009684283923739308].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V25, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0013428720, deltaAUC=-0.0001261696, wins=2/5, paired deltaKS CI=[-0.003623795880707395, 0.000749870001136515].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
