# V152: V25 teacher-tethered pairwise student, weight=0.03, tether=3.0

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities; beta=beta_V25+student_theta/strength; student trained with pairwise loss and teacher tether

OOF KS=0.53014030; delta vsV25=-0.00014436; AUC=0.84699458; deltaAUC=-0.00003470. KS wins 4/5.95% deltaKS CI=[-0.00044565210793290897, 0.000522679259009617].

Correction train-validation KS gap=0.01117790 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V152_V25_modeling.csv` SHA256 `be5650dfd43a5be00f78c29b6c7d69b37b00d072127f1211d37caaad04fb3407`.
