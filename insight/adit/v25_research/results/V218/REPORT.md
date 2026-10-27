# V218: Public-incumbent anchor with10% frozen Small

Perfold p=.90*pV25+.10*pV207; fixed weights; mean5foldtest; mean5fold test probabilities

KS=0.5302501273; AUC=0.8470037167; deltaKS=-0.0000345327; paired95% CI=[-0.0007125359453358765, 0.0008773166871357067].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0009015349, deltaAUC=-0.0001050587, wins=2/5, deltaKS95% CI=[-0.0031942945513103865, 0.0014303840765089286].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V25, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0012093872, deltaAUC=-0.0001038276, wins=2/5, paired deltaKS CI=[-0.0034299999705620643, 0.0010765368492303255].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
