# V137: V25 pairwise interaction representation L2base=0.001, L2interaction=0.03

corrected=sigmoid(logit(p_Kumo)+0.5*(X_V25@beta + interaction_representation)); final=0.75*corrected+0.25*p_spline77; mean5fold test probabilities

OOF KS=0.52926416; delta vsV25=-0.00102050; AUC=0.84688390; deltaAUC=-0.00014538. KS wins 1/5.95% deltaKS CI=[-0.0018986701049280102, 0.0003719395142555973].

Correction train-validation KS gap=0.01584077 versusV25 0.01110767.

Paired fold/class-stratified bootstrap conditional on these fitted OOF predictions; excludes retraining, repeated search and public leaderboard uncertainty

Error analysis: ERROR_ANALYSIS.md, ERROR_SEGMENTS.csv and ERROR_CASES.csv. All outcomes kept. Public score unknown. V25 remains public incumbent.

Submission `v7/submissions/submission_V137_V25_modeling.csv` SHA256 `15b7188e653c590ad53bbe54ba9a78b4eb3e6257f5dd3becf4fa4747f4866be9`.
