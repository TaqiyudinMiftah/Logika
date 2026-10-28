# V253: Fixed V25 with unprojected item information control

Keep the original V25 head beta0 fixed. On identical6k correction rows fit signedlog standardization for U; missing values are0 in centered basis with explicit shared support flags. R=U, without projecting out base features. eta0=logit(frozenKumo)+.5*Xbase*beta0. Fit gamma minimizing meanBCE(eta0+R*gamma)+.01/2||gamma||^2. Inference eta=eta0+.25*R*gamma; final=.75sigmoid(eta)+.25original_spline77; mean5fold test. Projection is training-only; query targets are unused.; mean5fold test probabilities

KS=0.5299600439; AUC=0.8469666453; deltaKS=-0.0003246161; paired95% CI=[-0.0009738523835021661, 0.0007945285206611583].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0011916184, deltaAUC=-0.0001421301, wins=2/5, deltaKS95% CI=[-0.003361580870546621, 0.0013631040254980422].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V25, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0014994707, deltaAUC=-0.0001408990, wins=2/5, paired deltaKS CI=[-0.0035198708196451296, 0.0011293062079508083].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
