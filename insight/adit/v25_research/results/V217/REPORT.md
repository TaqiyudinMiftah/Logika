# V217: V193 anchor plus10% highest-AUC stable spectral smoother

Perfold p=.90*pV193+.10*pV206; fixed weights; mean5foldtest; mean5fold test probabilities

KS=0.5309201140; AUC=0.8471930519; deltaKS=+0.0006354540; paired95% CI=[-0.0013537449836874048, 0.003191952117441852].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0002315482, deltaAUC=+0.0000842765, wins=3/5, deltaKS95% CI=[-0.0009212774668758639, 0.000789925376854053].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0005394005, deltaAUC=+0.0000855076, wins=1/5, paired deltaKS CI=[-0.0009532766888779631, 0.0005388229953770136].

Findings: BROAD_RANKING_GAIN_WITHOUT_KS_GAIN, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
