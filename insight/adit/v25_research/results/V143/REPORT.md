# V143: V25 factorization interaction embedding rank=4, L2base=0.001, L2embedding=0.001

corrected=sigmoid(logit(p_Kumo)+0.5*(X_V25@beta + interaction_representation)); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.52607367; delta vsV25=-0.00421099; AUC=0.84579418; deltaAUC=-0.00123510. KS wins 1/5.95% deltaKS CI=[-0.006296293878388546, -0.0010480316729817535].

Correction train-validation KS gap=0.03372526 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V143_V25_modeling.csv` SHA256 `99bc8e8aaee0a2ef019888f431ece6ed93c667473e22ce66a11da2e2338831fa`.
