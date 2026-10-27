# V165: V25 plus credibility-smoothed financial stress times installment burden

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities; feature_transform=cf_bayes_burden from V25 counts; undefined exposure stays missing; training-only reference

OOF KS=0.53009423; delta vsV25=-0.00019043; AUC=0.84707758; deltaAUC=+0.00004830. KS wins 3/5.95% deltaKS CI=[-0.0008474618488830937, 0.0009324062229948621].

Correction train-validation KS gap=0.01205734 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V165_V25_modeling.csv` SHA256 `596828c0424c4e5e10a738284f7fef3e0caeacc2c901fb9afd7115e90560eda6`.
