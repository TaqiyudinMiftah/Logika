# V139: V25 factorization interaction embedding rank=4, L2base=0.003, L2embedding=0.001

corrected=sigmoid(logit(p_Kumo)+0.5*(X_V25@beta + interaction_representation)); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.52581204; delta vsV25=-0.00447262; AUC=0.84593424; deltaAUC=-0.00109504. KS wins 0/5.95% deltaKS CI=[-0.006334522028755662, -0.0010813702980265521].

Correction train-validation KS gap=0.03300226 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V139_V25_modeling.csv` SHA256 `d7776bc562445beae998a9b3a1b3a0d456f72116a6adb2f8b7ca060dbcd5153b`.
