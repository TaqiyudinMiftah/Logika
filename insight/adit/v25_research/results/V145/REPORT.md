# V145: V25 plus hgb_a expert on existing116 inputs, expert_weight=0.1

p_final=(1-0.1)*p_V25+0.1*p_hgb_a_116; expert trained56k outer-training with inner early stopping; mean5fold test

OOF KS=0.53028478; delta vsV25=+0.00000012; AUC=0.84708251; deltaAUC=+0.00005323. KS wins 4/5.95% deltaKS CI=[-0.0011005784644045613, 0.001375331533768986].

Correction train-validation KS gap=0.01110767 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V145_V25_modeling.csv` SHA256 `85fd941738c5197a7f516ea7ca9ae594d8566562c5ee038293f7e87e1e1f9702`.
