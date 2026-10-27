# V131: Fixed V25 modeling: L2=0.001, strength=0.25, Kumo weight=0.75, blend=probability

corrected=sigmoid(logit(p_Kumo)+0.25*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.52916416; delta vsV25=-0.00112050; AUC=0.84734485; deltaAUC=+0.00031557. KS wins 2/5.95% deltaKS CI=[-0.0023953474612063595, 0.0006349933249198523].

Correction train-validation KS gap=0.00756655 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V131_V25_modeling.csv` SHA256 `5e047bd4cd27f81aa8526fba6d0b53323591793fd44770865e1638b7b9520f5b`.
