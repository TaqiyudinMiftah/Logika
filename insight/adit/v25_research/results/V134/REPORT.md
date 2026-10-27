# V134: V25 pairwise interaction representation L2base=0.003, L2interaction=0.03

corrected=sigmoid(logit(p_Kumo)+0.5*(X_V25@beta + interaction_representation)); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.52895801; delta vsV25=-0.00132665; AUC=0.84697601; deltaAUC=-0.00005327. KS wins 0/5.95% deltaKS CI=[-0.0021709495039644784, 0.00010783385575867636].

Correction train-validation KS gap=0.01450070 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V134_V25_modeling.csv` SHA256 `e4a7e61ba05dc90fee6789421068454535dabb3f5cc352041b25fc470430d634`.
