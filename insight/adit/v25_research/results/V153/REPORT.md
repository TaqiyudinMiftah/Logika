# V153: V25 teacher-tethered pairwise student, weight=0.1, tether=3.0

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities; beta=beta_V25+student_theta/strength; student trained with pairwise loss and teacher tether

OOF KS=0.53003511; delta vsV25=-0.00024955; AUC=0.84698859; deltaAUC=-0.00004068. KS wins 4/5.95% deltaKS CI=[-0.0004717329090324823, 0.0005124188631133174].

Correction train-validation KS gap=0.01118944 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V153_V25_modeling.csv` SHA256 `138d98ee3d6ec4e500e65d35603e600ba54fd2cad8cc4890023b5e6a46651a62`.
