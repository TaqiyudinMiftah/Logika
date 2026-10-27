# V122: Fixed V25 modeling: L2=0.003, strength=1.0, Kumo weight=0.75, blend=probability

corrected=sigmoid(logit(p_Kumo)+1.0*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.52727040; delta vsV25=-0.00301426; AUC=0.84536771; deltaAUC=-0.00166157. KS wins 2/5.95% deltaKS CI=[-0.005304808644192508, -0.0001356206873597586].

Correction train-validation KS gap=0.02121052 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V122_V25_modeling.csv` SHA256 `e0d7f650324d71e368e1e0bfe0c446450ad871491568a87e0a56ff845b7f8b06`.
