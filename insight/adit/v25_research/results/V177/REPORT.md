# V177: Conservative fixed90:10 V25 and V169 score blend

p_final=.90*p_V25+.10*p_V169; fixed matched-fold score average

KS=0.5303371946; AUC=0.8470597359; deltaKS=+0.0000525346; paired95% CI=[-0.000406247132978918, 0.0004185717536216236].

Development-selected recipe; conditional bootstrap excludes search/retraining uncertainty. Source-head gap is diagnostic only; no new supervised gate or meta-weights fitted.

This CPU cache replay evaluates the exact queued prediction recipe without unnecessary Kumo/GPU refitting. All5folds, ID alignment and frozen-source checks verified. Public score unknown; V25 remains incumbent.
