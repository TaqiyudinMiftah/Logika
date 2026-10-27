# V195: Coverage safeguard for V185 spectral correction

p=pV178 when aux_allocation_count<=0 or missing; otherwise pV185; per-fold test fusion then mean5fold; mean5fold test probabilities

KS=0.5301174581; AUC=0.8471314825; deltaKS=-0.0001672019; paired95% CI=[-0.0016870799827978565, 0.0019181036206950847].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0010342042, deltaAUC=+0.0000227071, wins=3/5, deltaKS95% CI=[-0.002426299182393274, 0.0005361233670778504].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.
