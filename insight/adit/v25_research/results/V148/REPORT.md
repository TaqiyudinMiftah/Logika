# V148: V25 plus hgb_b expert on existing116 inputs, expert_weight=1.0

p_final=(1-1.0)*p_V25+1.0*p_hgb_b_116; expert trained56k outer-training with inner early stopping; mean5fold test

OOF KS=0.52115046; delta vsV25=-0.00913420; AUC=0.84271820; deltaAUC=-0.00431108. KS wins 0/5.95% deltaKS CI=[-0.013527808760565632, -0.004397834256049596].

Correction train-validation KS gap=0.01110767 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V148_V25_modeling.csv` SHA256 `cc9e4ba8aa4f6c3b1039e1028e571e9dd60d0b367dd4640b567c01341bd2e637`.
