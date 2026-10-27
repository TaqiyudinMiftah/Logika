# V150: V25 plus hgb_b expert on existing116 inputs, expert_weight=0.2

p_final=(1-0.2)*p_V25+0.2*p_hgb_b_116; expert trained56k outer-training with inner early stopping; mean5fold test

OOF KS=0.52922498; delta vsV25=-0.00105968; AUC=0.84702555; deltaAUC=-0.00000373. KS wins 1/5.95% deltaKS CI=[-0.0027014700436127108, 0.0008448411189521415].

Correction train-validation KS gap=0.01110767 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V150_V25_modeling.csv` SHA256 `32ed49867a7c5e9e9e95b521e82df9e25ab9f767d4aeb9b71ef8e79b05fe7d10`.
