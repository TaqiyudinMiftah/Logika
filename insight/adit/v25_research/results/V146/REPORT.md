# V146: V25 plus hgb_a expert on existing116 inputs, expert_weight=0.2

p_final=(1-0.2)*p_V25+0.2*p_hgb_a_116; expert trained56k outer-training with inner early stopping; mean5fold test

OOF KS=0.53052833; delta vsV25=+0.00024367; AUC=0.84704291; deltaAUC=+0.00001364. KS wins 3/5.95% deltaKS CI=[-0.0014507234672599095, 0.001874683001353624].

Correction train-validation KS gap=0.01110767 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V146_V25_modeling.csv` SHA256 `f7bcbba9d637c602a92b44873b90e5efbda870ede5010a0c95f1836383d8092d`.
