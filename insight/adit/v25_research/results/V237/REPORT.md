# V237: Five-view conservative raw context/capacity fusion

raw=mean(globalLarge,coverageRaw,employmentRaw,APRRaw,globalSmall)=.71globalLarge+.03each local view+.20Small; refit18-input6k correction L2.003/shrink.5; same75:25spline77; mean5foldtest; mean5fold test probabilities

KS=0.5304349086; AUC=0.8469820790; deltaKS=+0.0001502486; paired95% CI=[-0.0007543627306984573, 0.0012621359796432606].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0007167536, deltaAUC=-0.0001266964, wins=2/5, deltaKS95% CI=[-0.003009894545684208, 0.0014916778496735438].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, AUC_DECREASE_VS_V25, AUC_DECREASE_VS_V185, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0010246059, deltaAUC=-0.0001254654, wins=2/5, paired deltaKS CI=[-0.0032055806694640254, 0.0011916110973940709].

Findings: NO_GAIN_AGAINST_DEVELOPMENT_CONTROL, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
