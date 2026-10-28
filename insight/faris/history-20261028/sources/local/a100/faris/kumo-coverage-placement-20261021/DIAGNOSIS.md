# Completed placement comparison

All three recipes completed five folds with 70,000 aligned OOF predictions. All 15 fitted-context caches passed independent full-file SHA256 reads. All five shared spline pipelines were independently replayed.

| Recipe | Pooled KS | AUC | Mean fold KS |
|---|---:|---:|---:|
| Repaired control90 | 0.530219 | 0.847026 | 0.533361 |
| True coverage96 | 0.529192 | 0.847173 | 0.533956 |
| Shuffled null96 | 0.529408 | 0.847015 | 0.532961 |

Coverage improved four individual fold KS values against the control and all five against the shuffled null, but lowered pooled KS against both. Fold maxima occur at different probability thresholds; their average is not the frozen pooled objective. AUC improvements appear in both within-fold and cross-fold pairs, so these results do not establish cross-fold calibration drift as the cause of the KS loss. No label-derived threshold alignment or routing has been deployed.

The descriptive mean ROC separation over false-positive rates 0.05–0.30 increases slightly (+0.0002465) while peak pooled KS decreases. That is evidence of differing effects on the ROC curve, not proof of a deployable gain. Selecting new thresholds or mixtures on these labels would add further adaptive exposure.

The coverage placement is not promoted. Target 0.54 remains unmet. The preregistered A100 temperature-adaptation study remains separate; no partial-fold performance selection is permitted. Historical label reuse and unresolved snapshot availability remain disclosed.
