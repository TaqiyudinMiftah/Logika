# V212: Coverage-aware conservative Small/Large blend

wSmall=.15*10/(max(finite_allocation_count,0)+10); p=(1-wSmall)*pV193+wSmall*pV207; missing count getsw.15; mean5foldtest; mean5fold test probabilities

KS=0.5311396610; AUC=0.8471133433; deltaKS=+0.0008550010; paired95% CI=[-0.0011351800284535335, 0.003182729832802792].

Same5outer folds. Kumo frozen and its context scores unchanged. Teacher was fitted on the same6k correction rows; judge utility on held-out OOF, not training loss. Repeated development search remains. V25 public incumbent unchanged; no new public score claimed.


V185 comparison: deltaKS=-0.0000120012, deltaAUC=+0.0000045678, wins=2/5, deltaKS95% CI=[-0.0007631766999804723, 0.0009040096787144865].

Generalization decision: EXPERIMENT_ONLY_KEEP_V25_PUBLIC_INCUMBENT. Flags: REUSED_DEVELOPMENT_FOLDS, MANY_PRIOR_EXPERIMENTS, NO_INDEPENDENT_GENERALIZATION_TEST, DELTA_KS_V25_CI_INCLUDES_ZERO_OR_NEGATIVE, DELTA_KS_V185_CI_INCLUDES_ZERO_OR_NEGATIVE, FEWER_THAN_FOUR_FOLD_KS_WINS_VS_V185. Public V25 remains0.56822; V185 verified0.56699. No new public score is inferred.


V193: deltaKS=-0.0003198536, deltaAUC=+0.0000057989, wins=1/5, paired deltaKS CI=[-0.0010782610937352278, 0.0007510896438461442].

Findings: BROAD_RANKING_GAIN_WITHOUT_KS_GAIN, CONDITIONAL_KS_DELTA_CI_DOES_NOT_ESTABLISH_IMPROVEMENT.
