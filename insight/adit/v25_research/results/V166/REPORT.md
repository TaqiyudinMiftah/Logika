# V166: V25 plus streamed30d item-value category entropy

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities; extra_inputs=['light_basket_value_entropy_30d']; undefined exposure stays missing; event cache or training-only reference

OOF KS=0.52976926; delta vsV25=-0.00051540; AUC=0.84701513; deltaAUC=-0.00001415. KS wins 2/5.95% deltaKS CI=[-0.0016029484574948932, 0.0007176648174437023].

Correction train-validation KS gap=0.01300667 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V166_V25_modeling.csv` SHA256 `b6d03967204c4a0e1cc7979ba17c9b878c0d5c4329d2b8b9a8e61e93490a79e9`.
