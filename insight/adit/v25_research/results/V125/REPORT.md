# V125: Fixed V25 modeling: L2=0.003, strength=0.5, Kumo weight=0.9, blend=probability

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=0.9*corrected+0.09999999999999998*p_spline77; mean5fold test probabilities

OOF KS=0.53027518; delta vsV25=-0.00000948; AUC=0.84687636; deltaAUC=-0.00015292. KS wins 3/5.95% deltaKS CI=[-0.002081950318837264, 0.0018663097761793842].

Correction train-validation KS gap=0.01110767 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V125_V25_modeling.csv` SHA256 `36d9aa7da790977396d99129fe7f8d8096592f41ef87f75cb5ad50e298ded251`.
