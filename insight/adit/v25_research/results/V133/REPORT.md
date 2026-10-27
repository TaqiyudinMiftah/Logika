# V133: V25 pairwise interaction representation L2base=0.003, L2interaction=0.01

corrected=sigmoid(logit(p_Kumo)+0.5*(X_V25@beta + interaction_representation)); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.52857022; delta vsV25=-0.00171444; AUC=0.84692679; deltaAUC=-0.00010249. KS wins 0/5.95% deltaKS CI=[-0.002706632039994275, -3.2168147103495896e-05].

Correction train-validation KS gap=0.01557188 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V133_V25_modeling.csv` SHA256 `58b3750efdf7dde8c58b4b5b8b4117667e419762088b0593cd2f24bc5d8be785`.
