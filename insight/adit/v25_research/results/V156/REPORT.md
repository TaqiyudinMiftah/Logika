# V156: V25 teacher-tethered soft_cdf student, weight=0.01, tether=3.0

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities; beta=beta_V25+student_theta/strength; student trained with soft_cdf loss and teacher tether

OOF KS=0.53016883; delta vsV25=-0.00011583; AUC=0.84699685; deltaAUC=-0.00003243. KS wins 3/5.95% deltaKS CI=[-0.00041723652371573606, 0.0004967156535028548].

Correction train-validation KS gap=0.01119708 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V156_V25_modeling.csv` SHA256 `b1175df3923a593e77d9854f343385a2d43b0ef6c0cc2502c022cdab0f90e4ec`.
