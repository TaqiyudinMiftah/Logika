# V141: V25 factorization interaction embedding rank=8, L2base=0.003, L2embedding=0.001

corrected=sigmoid(logit(p_Kumo)+0.5*(X_V25@beta + interaction_representation)); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.52553849; delta vsV25=-0.00474617; AUC=0.84541714; deltaAUC=-0.00161214. KS wins 1/5.95% deltaKS CI=[-0.007322543994082248, -0.0010436071046548596].

Correction train-validation KS gap=0.03595793 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V141_V25_modeling.csv` SHA256 `ecf6315baebc507bba930a73408bcb7ee87f13ba5ffbfb44d216c565702746c8`.
