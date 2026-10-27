# V130: Fixed V25 modeling: L2=0.01, strength=1.0, Kumo weight=0.75, blend=probability

corrected=sigmoid(logit(p_Kumo)+1.0*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.52881918; delta vsV25=-0.00146548; AUC=0.84604208; deltaAUC=-0.00098720. KS wins 2/5.95% deltaKS CI=[-0.00337615233414211, 0.0008856536147134159].

Correction train-validation KS gap=0.01915399 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V130_V25_modeling.csv` SHA256 `b74548acb4e6b5eed5947d2382cdf03dd383d96060724ebd959c5bd1116d3662`.
