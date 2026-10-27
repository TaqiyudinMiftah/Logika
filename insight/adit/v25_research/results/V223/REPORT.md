# V223: Underwriting-regime frozen Large population context

Frozen Kumo Large;50k labeled context,8 estimators; disjoint6k offset-head training;14k outer validation; .75*sigmoid(logit(p_Kumo)+.5*Xbeta)+.25*p_original_spline77; mean5fold test probabilities; p_Kumo=.85*global_V168_90inputs+.15*local_apr_regime_expert; reference and local contexts use original50k pool only

OOF KS=0.53021277; AUC=0.84703242; delta KS against cached V25=-0.00007189.95% paired delta CI=[-0.0008135958002010585, 0.0010307531911573238].

All5folds evaluated. Frozen weights have no gradient updates. In-context fitting was rerun. Torch/CUDA runtime differs from archived control; V168 is a fresh runtime control. Development search bias remains, and a public score is unknown. V25 remains the public incumbent.

Error analysis is held-out only. Legacy snapshot provenance remains unverified.


V185 comparison: deltaKS=-0.0009388932, deltaAUC=-0.0000763597, wins=2/5, deltaKS95% CI=[-0.0031711858495691273, 0.001697078253536826].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0012467455, deltaAUC=-0.0000751286, wins=2/5, paired deltaKS CI=[-0.0034914672331964602, 0.001509313047756952].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
