# V142: V25 factorization interaction embedding rank=8, L2base=0.003, L2embedding=0.003

corrected=sigmoid(logit(p_Kumo)+0.5*(X_V25@beta + interaction_representation)); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.52663436; delta vsV25=-0.00365030; AUC=0.84616353; deltaAUC=-0.00086575. KS wins 0/5.95% deltaKS CI=[-0.005976407290893662, -0.0005788757194137642].

Correction train-validation KS gap=0.02847674 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V142_V25_modeling.csv` SHA256 `5d1e2187bceae95ce02ee9380b6772195de4c280b4938ce982dfa31546e08812`.
