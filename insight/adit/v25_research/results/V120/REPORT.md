# V120: Fixed V25 modeling: L2=0.003, strength=0.25, Kumo weight=0.75, blend=probability

corrected=sigmoid(logit(p_Kumo)+0.25*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.52897926; delta vsV25=-0.00130540; AUC=0.84735856; deltaAUC=+0.00032928. KS wins 1/5.95% deltaKS CI=[-0.0026908558004891834, 0.0006932419688315731].

Correction train-validation KS gap=0.00774424 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V120_V25_modeling.csv` SHA256 `ed4133ab131efa5ebcedcf9587f53fb2fb95f56546f3b1643e1fbcd983bb7e67`.
