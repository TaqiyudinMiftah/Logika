# V129: Fixed V25 modeling: L2=0.03, strength=0.5, Kumo weight=0.75, blend=probability

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.52953272; delta vsV25=-0.00075194; AUC=0.84731552; deltaAUC=+0.00028624. KS wins 1/5.95% deltaKS CI=[-0.002039129370564624, 0.000936470727859491].

Correction train-validation KS gap=0.00715583 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V129_V25_modeling.csv` SHA256 `563a2a35cf7d48bbb2e5dbec3b5c15a4693e9a384f60540730fdcd834a23a669`.
