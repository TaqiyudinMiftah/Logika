# V124: Fixed V25 modeling: L2=0.003, strength=0.5, Kumo weight=1.0, blend=probability

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=1.0*corrected+0.0*p_spline77; mean5fold test probabilities

OOF KS=0.52944333; delta vsV25=-0.00084133; AUC=0.84654993; deltaAUC=-0.00047935. KS wins 2/5.95% deltaKS CI=[-0.0034886764914097146, 0.0016191771780549183].

Correction train-validation KS gap=0.01110767 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V124_V25_modeling.csv` SHA256 `85a42d0ea01ddf34740537d154577f5445141327e97c71e33df4dacc30bd3a16`.
