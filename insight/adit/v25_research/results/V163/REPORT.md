# V163: V25 training-population class-prior alignment, no validation-label fit

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities; corrected Kumo logit adds alpha*(logit(pi_outertrain)-(1-strength)*logit(pi_context)-strength*logit(pi_correction)); priors from training only

OOF KS=0.53028466; delta vsV25=+0.00000000; AUC=0.84702936; deltaAUC=+0.00000008. KS wins 0/5.95% deltaKS CI=[-7.606367811099911e-06, 1.65308382788526e-05].

Correction train-validation KS gap=0.01110767 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V163_V25_modeling.csv` SHA256 `48224d689b8fa9be8382c6fbebe0d54c82715d79e9aaa614d4686a6e1eaa936c`.
