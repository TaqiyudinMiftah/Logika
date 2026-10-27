# V151: V25 plus hgb_b expert on existing116 inputs, expert_weight=0.35

p_final=(1-0.35)*p_V25+0.35*p_hgb_b_116; expert trained56k outer-training with inner early stopping; mean5fold test

OOF KS=0.52741159; delta vsV25=-0.00287307; AUC=0.84674781; deltaAUC=-0.00028146. KS wins 1/5.95% deltaKS CI=[-0.004938312421822836, -0.00016139845786711086].

Correction train-validation KS gap=0.01110767 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V151_V25_modeling.csv` SHA256 `90199ea3d070acfb0bd97e2507cb7442ce61eedcbaba5a126dcdc2ccf98311dc`.
