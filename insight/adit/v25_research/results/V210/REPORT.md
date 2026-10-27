# V210: V193 anchor with10% high-AUC reduced-knot representation

Perfold p=.90*pV193+.10*pV198; fixed weights, then mean5foldtest; mean5fold test probabilities

KS=0.5314971058; AUC=0.8471504442; deltaKS=+0.0012124458; paired95% CI=[-0.0012589113185661888, 0.0032165172004676645].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=+0.0003454436, deltaAUC=+0.0000416688, wins=2/5, deltaKS95% CI=[-0.00038150699555706543, 0.000775335221083062].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=+0.0000375913, deltaAUC=+0.0000428998, wins=1/5, paired deltaKS CI=[-0.0006081101947939954, 0.0006178375159301391].

Findings: POOLED_KS_AND_AUC_IMPROVE_ON_REUSED_DEVELOPMENT_FOLDS, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
