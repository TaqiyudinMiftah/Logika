# V149: V25 plus hgb_b expert on existing116 inputs, expert_weight=0.1

p_final=(1-0.1)*p_V25+0.1*p_hgb_b_116; expert trained56k outer-training with inner early stopping; mean5fold test

OOF KS=0.52988544; delta vsV25=-0.00039922; AUC=0.84708141; deltaAUC=+0.00005213. KS wins 3/5.95% deltaKS CI=[-0.0015652486871090166, 0.0008709606085078127].

Correction train-validation KS gap=0.01110767 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V149_V25_modeling.csv` SHA256 `a9c791af2cd4f6174f8eb654cf63ed22dd538d428bd54e19416bfbefda110100`.
