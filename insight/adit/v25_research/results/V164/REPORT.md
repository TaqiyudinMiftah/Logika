# V164: LightGBM116 expert:56k outer-training, inner early stopping; same V25 input information

p_final=(1-1.0)*p_V25+1.0*p_lgbm_a_116; expert trained56k outer-training with inner early stopping; mean5fold test

OOF KS=0.52299232; delta vsV25=-0.00729234; AUC=0.84376333; deltaAUC=-0.00326595. KS wins 0/5.95% deltaKS CI=[-0.010852520890501285, -0.003428099106734739].

Correction train-validation KS gap=0.01110767 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V164_V25_modeling.csv` SHA256 `7a531365c1ed320e34ba6dc36d33e146bc00b5a69ab05b32047c23cc36834a67`.
