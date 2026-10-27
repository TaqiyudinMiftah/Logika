# V220: V193 anchor with10% resampled reduced-knot expert

Perfold p=.9*pV193+.1*pV219; fixed weights; mean5foldtest; mean5fold test probabilities

KS=0.5314909887; AUC=0.8471555928; deltaKS=+0.0012063287; paired95% CI=[-0.0009794833796053892, 0.0033741403170764843].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=+0.0003393264, deltaAUC=+0.0000468174, wins=2/5, deltaKS95% CI=[-0.0004367029029353264, 0.0010024598885434192].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=+0.0000314741, deltaAUC=+0.0000480485, wins=1/5, paired deltaKS CI=[-0.0006477966391333189, 0.0008679233473567132].

Findings: POOLED_KS_AND_AUC_IMPROVE_ON_REUSED_DEVELOPMENT_FOLDS, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
