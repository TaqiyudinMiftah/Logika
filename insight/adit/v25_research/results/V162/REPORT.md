# V162: V25 plus normalized eligible-month cashflow stress representation

corrected=sigmoid(logit(p_Kumo)+0.5*X_V25@beta); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities; X includes negative_CF_months/(negative_CF_months+nonnegative_CF_months); undefined exposure stays missing

OOF KS=0.52940117; delta vsV25=-0.00088349; AUC=0.84699812; deltaAUC=-0.00003116. KS wins 2/5.95% deltaKS CI=[-0.001256041330649782, 0.00041797569057721917].

Correction train-validation KS gap=0.01187443 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V162_V25_modeling.csv` SHA256 `80cac642556f0127051d9ed4431ea9cb78f422e42632f7f1443f21f1cb0ade98`.
