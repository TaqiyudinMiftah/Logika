# V240: Training-only principal-component compression of correction basis

Fit original18 V25 spline encoder on6k; center its basis/missing/employment columns and retain PCA components explaining95%training variance, whiten=False. Preserve separate intercept and standardizedKumo logit. Fit meanBCE(offset+XPCA*beta)+.003/2||beta_nonintercept||^2. Inference eta=offset+.5XPCA*beta; final75:25spline77, mean5foldtest. No target used in PCA or component-count selection.; mean5fold test probabilities

KS=0.5293251724; AUC=0.8470954144; deltaKS=-0.0009594876; paired95% CI=[-0.0027588851034423406, 0.001052995271482969].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0018264898, deltaAUC=-0.0000133610, wins=2/5, deltaKS95% CI=[-0.004466862373221464, 0.000789239982599371].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0021343421, deltaAUC=-0.0000121299, wins=2/5, paired deltaKS CI=[-0.004650543903789539, 0.0004922038840041878].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
