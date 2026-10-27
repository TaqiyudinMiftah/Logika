# V161: V127 low-L2 correction plus20% cached HGB-a expert, V25 input information

p_final=(1-0.2)*p_V25+0.2*p_hgb_a_116; expert trained56k outer-training with inner early stopping; mean5fold test

OOF KS=0.53054021; delta vsV25=+0.00025555; AUC=0.84699493; deltaAUC=-0.00003435. KS wins 2/5.95% deltaKS CI=[-0.0013730962104125221, 0.0017775082499394213].

Correction train-validation KS gap=0.01209394 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V161_V25_modeling.csv` SHA256 `3ea7f748b32cbfaa445bc8dfe3ea0c2eb014e1563631e08fb09d0c933426cd26`.
