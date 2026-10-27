# V202: Laplace predictive uncertainty for original V25 head

Fit original V25 beta; Sigma=(X^T*W*X+n*.003*diag(0,1,...)+1e-6I)^-1; eta=zK+.5Xbeta; v=.25*diag(X*Sigma*X^T); pK~=sigmoid(eta/sqrt(1+pi*v/8)); original75:25 spline fusion; mean5fold test probabilities

KS=0.5305400943; AUC=0.8470400856; deltaKS=+0.0002554342; paired95% CI=[-0.00022797446591806548, 0.0007388064937507292].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0006115680, deltaAUC=-0.0000686898, wins=2/5, deltaKS95% CI=[-0.0032301354500582275, 0.0015483165321713778].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0009194203, deltaAUC=-0.0000674588, wins=2/5, paired deltaKS CI=[-0.003345798632157401, 0.0013417080486651475].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
