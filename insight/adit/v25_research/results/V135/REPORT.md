# V135: V25 pairwise interaction representation L2base=0.001, L2interaction=0.003

corrected=sigmoid(logit(p_Kumo)+0.5*(X_V25@beta + interaction_representation)); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.52827096; delta vsV25=-0.00201370; AUC=0.84679176; deltaAUC=-0.00023752. KS wins 1/5.95% deltaKS CI=[-0.002968741591509705, 7.110807812164523e-05].

Correction train-validation KS gap=0.01551224 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V135_V25_modeling.csv` SHA256 `9ab66aab09aa5ba91723dd491078eb303610fa93d4924071053c753b2ef2a049`.
