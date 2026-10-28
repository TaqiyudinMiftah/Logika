# V248: Joint data representation without price trajectory

V25 original18-input quadratic offset design plus standardized signedlog linear inputs ip_price_comparable_categories,ip_multicat_price_orders180,ip_valid_price_items180,ip_price_change_observed,ip_order_coherence_observed,ip_affordability_cross_observed,ip_debt_cross_observed,ip_liquidity_cross_observed,ip_within_order_price_covariance_shrunk,ip_within_order_price_dispersion,ip_price_upgrade_x_installment_burden,ip_order_covariance_x_total_burden,ip_price_dispersion_x_liquidity_shortfall; all new value-availability masks are explicit shared support features. Fit meanBCE(KumoLogit+Xbeta)+.003/2||base_beta_nonintercept||^2+.01/2||new_beta||^2 on identical6k; new means/scales fit only6k; undefined additional values contribute0 in centered basis and retainNaN in raw cache; inference eta=KumoLogit+.5Xbeta; final75:25original_spline77; mean5foldtest.; mean5fold test probabilities

KS=0.5294077101; AUC=0.8469072303; deltaKS=-0.0008769499; paired95% CI=[-0.0021775534060619853, 0.0007685933446491071].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0017439521, deltaAUC=-0.0002015451, wins=2/5, deltaKS95% CI=[-0.004153198628266919, 0.0009428291973470707].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V25, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0020518044, deltaAUC=-0.0002003140, wins=2/5, paired deltaKS CI=[-0.004278128434584002, 0.0006294824369111301].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.

Unique information vs shared support controlV243: deltaKS=+0.0000606618, deltaAUC=-0.0000128480, paired deltaKS CI=[-0.001063986618377688, 0.001186174282564095]. No independent holdout is created by this comparison.
