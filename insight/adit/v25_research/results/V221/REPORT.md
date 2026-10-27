# V221: Uncertainty-gated V193 and bootstrap correction blend

m=median of3-member logit variance on correction-training only; w=.1*m/(m+query_variance); p=(1-w)*pV193+w*pV219; ifm undefined/nearzero retainV193 explicitly; mean5foldtest; mean5fold test probabilities

KS=0.5316293525; AUC=0.8471248959; deltaKS=+0.0013446925; paired95% CI=[-0.0010941447045733298, 0.0033411652331080477].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=+0.0004776903, deltaAUC=+0.0000161205, wins=3/5, deltaKS95% CI=[-0.00029151151748111893, 0.0008013130848074171].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=+0.0001698380, deltaAUC=+0.0000173516, wins=2/5, paired deltaKS CI=[-0.000570580088502573, 0.0005896054627933019].

Findings: POOLED_KS_AND_AUC_IMPROVE_ON_REUSED_DEVELOPMENT_FOLDS, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
