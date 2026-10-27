# V138: V25 factorization interaction embedding rank=4, L2base=0.003, L2embedding=0.0001

corrected=sigmoid(logit(p_Kumo)+0.5*(X_V25@beta + interaction_representation)); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.52576280; delta vsV25=-0.00452186; AUC=0.84568596; deltaAUC=-0.00134332. KS wins 1/5.95% deltaKS CI=[-0.007398083943520287, -0.0016551673968552313].

Correction train-validation KS gap=0.03711489 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V138_V25_modeling.csv` SHA256 `db816643c7801e8b76a58cb5eba96ae1d46764d4beb150e0386ce66ea3e0a5c4`.
