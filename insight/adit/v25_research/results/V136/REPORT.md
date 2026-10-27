# V136: V25 pairwise interaction representation L2base=0.001, L2interaction=0.01

corrected=sigmoid(logit(p_Kumo)+0.5*(X_V25@beta + interaction_representation)); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.52858198; delta vsV25=-0.00170268; AUC=0.84683149; deltaAUC=-0.00019779. KS wins 1/5.95% deltaKS CI=[-0.002609034193363463, 0.00010326131495494403].

Correction train-validation KS gap=0.01548987 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V136_V25_modeling.csv` SHA256 `a19c675e071f38cd0eef2b56865a9325e9dae94d910fafe1fac8f141feda220c`.
