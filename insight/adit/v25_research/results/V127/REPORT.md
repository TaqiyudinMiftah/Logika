# V127: Fixed V25 modeling: L2=0.001, strength=0.5, Kumo weight=0.75, blend=probability

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.53067122; delta vsV25=+0.00038656; AUC=0.84694533; deltaAUC=-0.00008395. KS wins 2/5.95% deltaKS CI=[-0.0007457498544474361, 0.001345725301449125].

Correction train-validation KS gap=0.01209394 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V127_V25_modeling.csv` SHA256 `052c480af818133f21f5a909467c4ae83f1591f40e20efcff8fc107866595160`.
