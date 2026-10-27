# V121: Fixed V25 modeling: L2=0.003, strength=0.75, Kumo weight=0.75, blend=probability

corrected=sigmoid(logit(p_Kumo)+0.75*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.52913497; delta vsV25=-0.00114969; AUC=0.84636164; deltaAUC=-0.00066764. KS wins 2/5.95% deltaKS CI=[-0.0024081754230092763, 0.000980171990652861].

Correction train-validation KS gap=0.01699399 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V121_V25_modeling.csv` SHA256 `e13c3d6ec5b4f49e013cbdd202174218f865e7e7ae028522f3166060802830b7`.
