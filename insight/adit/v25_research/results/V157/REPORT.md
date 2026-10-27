# V157: V25 teacher-tethered soft_cdf student, weight=0.03, tether=3.0

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities; beta=beta_V25+student_theta/strength; student trained with soft_cdf loss and teacher tether

OOF KS=0.53025601; delta vsV25=-0.00002865; AUC=0.84699583; deltaAUC=-0.00003345. KS wins 2/5.95% deltaKS CI=[-0.0004481590329447588, 0.0005300831427439367].

Correction train-validation KS gap=0.01116953 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V157_V25_modeling.csv` SHA256 `880c31a9b5866c9425b114b69e67895732d21dcf5bd5816d1ad18998c4acfece`.
