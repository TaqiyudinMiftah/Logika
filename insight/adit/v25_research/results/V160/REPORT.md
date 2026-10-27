# V160: V25 local regularization follow-up: L2=0.0003, strength0.5, fixed75:25

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.53049697; delta vsV25=+0.00021231; AUC=0.84690263; deltaAUC=-0.00012665. KS wins 3/5.95% deltaKS CI=[-0.00074668850594366, 0.0013572954102584272].

Correction train-validation KS gap=0.01292640 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V160_V25_modeling.csv` SHA256 `f2243fa784bc2a69483ba0bc73e713f4d525e83680284b882fbe02723e698108`.
