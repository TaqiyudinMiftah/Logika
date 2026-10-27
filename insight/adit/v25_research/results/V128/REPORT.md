# V128: Fixed V25 modeling: L2=0.01, strength=0.5, Kumo weight=0.75, blend=probability

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.52976767; delta vsV25=-0.00051699; AUC=0.84718277; deltaAUC=+0.00015349. KS wins 3/5.95% deltaKS CI=[-0.0013036488233994475, 0.0007517380449815228].

Correction train-validation KS gap=0.00918195 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V128_V25_modeling.csv` SHA256 `2966456ae91d3e213fb97c24abd1de4178db67d5a9bb7d56e6aa2d0985c0254f`.
