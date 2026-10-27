# V126: Fixed V25 modeling: L2=0.003, strength=0.5, Kumo weight=0.75, blend=logit

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=sigmoid(0.75*logit(corrected)+0.25*logit(p_spline77)); mean5fold test probabilities

OOF KS=0.53040649; delta vsV25=+0.00012183; AUC=0.84701560; deltaAUC=-0.00001368. KS wins 2/5.95% deltaKS CI=[-0.0008409316287327911, 0.0006440833298153203].

Correction train-validation KS gap=0.01110767 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V126_V25_modeling.csv` SHA256 `7c27a78baff1f538995b26decd9bdbc809f81459e5db291ae4d1d03f74a04fc5`.
