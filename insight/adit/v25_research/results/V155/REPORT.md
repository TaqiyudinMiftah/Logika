# V155: V25 teacher-tethered pairwise student, weight=0.1, tether=10.0

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities; beta=beta_V25+student_theta/strength; student trained with pairwise loss and teacher tether

OOF KS=0.53020495; delta vsV25=-0.00007971; AUC=0.84701707; deltaAUC=-0.00001221. KS wins 3/5.95% deltaKS CI=[-0.0003198703296492272, 0.0002434934489254218].

Correction train-validation KS gap=0.01137805 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V155_V25_modeling.csv` SHA256 `571953778c82478b8bdfe121996b7dbabe48cefb1ce3a318b6ed517dd4db2517`.
