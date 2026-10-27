# V132: V25 pairwise interaction representation L2base=0.003, L2interaction=0.003

corrected=sigmoid(logit(p_Kumo)+0.5*(X_V25@beta + interaction_representation)); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.52837343; delta vsV25=-0.00191123; AUC=0.84688898; deltaAUC=-0.00014030. KS wins 0/5.95% deltaKS CI=[-0.0029867743243464835, -1.7615940795645686e-05].

Correction train-validation KS gap=0.01527415 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V132_V25_modeling.csv` SHA256 `b0542cc71aaf21e63b707db1cd1b330541e231671e1f1d84deafd31cd599556a`.
