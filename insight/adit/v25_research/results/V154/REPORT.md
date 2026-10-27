# V154: V25 teacher-tethered pairwise student, weight=0.3, tether=3.0

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities; beta=beta_V25+student_theta/strength; student trained with pairwise loss and teacher tether

OOF KS=0.53013588; delta vsV25=-0.00014878; AUC=0.84697258; deltaAUC=-0.00005670. KS wins 4/5.95% deltaKS CI=[-0.000471039342384974, 0.0006343920019843817].

Correction train-validation KS gap=0.01149452 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V154_V25_modeling.csv` SHA256 `e40324191b3a33b5b0b65d327938b2c69b47a35f49ef41f5506db40bf2cce6e5`.
