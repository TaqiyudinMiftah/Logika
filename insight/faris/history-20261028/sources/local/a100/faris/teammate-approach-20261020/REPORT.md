# Conditional correction research

Conclusion: no_joint_KS_AUC_improvement.

The primary procedure chooses its correction using only inner training validation.
The three fixed variants below are secondary diagnostics, not post-hoc winners.

| Pipeline | KS | AUC | Log loss |
|---|---:|---:|---:|
| incumbent | 0.52929947 | 0.84720269 | 0.29249967 |
| control | 0.52929947 | 0.84720269 | 0.29249967 |
| primary | 0.52822426 | 0.84704922 | 0.29263763 |
| credit_agreement | 0.52862782 | 0.84718556 | 0.29252586 |
| consumption_obligations | 0.52852182 | 0.84710559 | 0.29259829 |
| measurement_reliability | 0.52750194 | 0.84706256 | 0.29260233 |

Primary choices by outer fold: consumption_obligations, control, consumption_obligations, consumption_obligations, measurement_reliability.

Primary KS change: -0.00107520; AUC change: -0.00015347.
KS interval: [-0.0022283549134150594, 0.0011520571501491988].
AUC interval: [-0.0003014550157991791, -1.4804724556411513e-05].

80 correction fits implement three hypotheses and one nested selector across five folds.
All models, inner predictions and final OOF predictions are retained.
No new Kumo inference or competition submission was needed for this matched correction comparison.
All labels have historical exposure; these are not private leaderboard scores or untouched confirmation.
Snapshot timing uncertainty and globally estimated label-free reference features are inherited.
Confidence intervals omit historical selection and retraining uncertainty.
