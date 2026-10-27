# V144: V25 plus hgb_a expert on existing116 inputs, expert_weight=1.0

p_final=(1-1.0)*p_V25+1.0*p_hgb_a_116; expert trained56k outer-training with inner early stopping; mean5fold test

OOF KS=0.52154337; delta vsV25=-0.00874129; AUC=0.84317425; deltaAUC=-0.00385503. KS wins 0/5.95% deltaKS CI=[-0.011808248384394879, -0.004064915614278546].

Correction train-validation KS gap=0.01110767 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V144_V25_modeling.csv` SHA256 `1b39af4be16274d5c1431625a04fc7366219ce6e497c28f9c55b41185c12ba37`.
