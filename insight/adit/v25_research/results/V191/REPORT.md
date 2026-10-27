# V191: Joint local-context Kumo and29-input spectral correction

Use exact cached V178 raw Kumo=.85*global+.15*local; refit29-input V167 correction with L2.003 and inference shrink.5; final=.75*sigmoid(logit(pK178)+.5*X29*beta)+.25*original_spline77; mean5fold test probabilities

KS=0.5311832529; AUC=0.8470755068; deltaKS=+0.0008985929; paired95% CI=[-0.0013096221906036877, 0.0032342631605577748].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=+0.0000315906, deltaAUC=-0.0000332686, wins=1/5, deltaKS95% CI=[-0.00034709226632174684, 0.00022095577140747097].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, LARGER_TRAIN_VALIDATION_HEAD_GAP, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.
