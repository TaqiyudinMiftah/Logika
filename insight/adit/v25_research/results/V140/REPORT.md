# V140: V25 factorization interaction embedding rank=4, L2base=0.003, L2embedding=0.003

corrected=sigmoid(logit(p_Kumo)+0.5*(X_V25@beta + interaction_representation)); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.52749307; delta vsV25=-0.00279159; AUC=0.84634666; deltaAUC=-0.00068262. KS wins 1/5.95% deltaKS CI=[-0.004952882233551753, -0.000267939249790059].

Correction train-validation KS gap=0.02772425 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V140_V25_modeling.csv` SHA256 `3ad0ac5a8c7d0387a67bd54da797088f376d7f5f4abc3a9a3b4185f204b3d8f7`.
