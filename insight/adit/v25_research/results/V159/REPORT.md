# V159: V25 teacher-tethered soft_cdf student, weight=0.03, tether=10.0

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities; beta=beta_V25+student_theta/strength; student trained with soft_cdf loss and teacher tether

OOF KS=0.53014483; delta vsV25=-0.00013983; AUC=0.84701888; deltaAUC=-0.00001040. KS wins 2/5.95% deltaKS CI=[-0.0003092923841221257, 0.0002728363823932545].

Correction train-validation KS gap=0.01144417 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V159_V25_modeling.csv` SHA256 `fa78c55b6ddaa0030d7cd87c923c02d4552e8aae1b00e82c558e3b25ef6edb59`.
