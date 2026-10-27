# V123: Fixed V25 modeling: L2=0.003, strength=0.5, Kumo weight=0.5, blend=probability

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=0.5*corrected+0.5*p_spline77; mean5fold test probabilities

OOF KS=0.52721670; delta vsV25=-0.00306796; AUC=0.84643710; deltaAUC=-0.00059218. KS wins 1/5.95% deltaKS CI=[-0.004639809943802894, 3.393668771707009e-05].

Correction train-validation KS gap=0.01110767 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V123_V25_modeling.csv` SHA256 `692779616613830dc6673305449e8c01d61a16b7d5d9b4dafd9acd99c2ccd335`.
