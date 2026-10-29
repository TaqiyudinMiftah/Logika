# V262 held-out interpretation

Pooled KS improves by only 3.3061676557588626e-05; delta AUC -9.737319277558498e-07. Two folds improve, two tie and one declines in KS. All four interaction coefficient directions vary across folds. This does not establish a stable incremental credit-risk signal after the existing V25 correction.

At the top10% and top20% score cutoffs, one fewer actual default is captured; at the original KS-max population fraction the default capture count is unchanged. Mean absolute OOF probability change is 6.143809561405066e-05. No meaningful improvement in difficult-case separation is demonstrated.

Paired95% delta-KS interval [-0.000188366354834682, 0.00025193944317325657] includes zero; delta-AUC interval [-5.204771034036937e-06, 2.5667834433279456e-06] also includes zero. Small effects and extensive prior development reuse argue for keeping V25. This is evidence of no established gain, not proof of no possible useful credit post-hoc model.

Coefficient signs by fold: {"late30_x_allocation_reliability": [1, 1, -1, -1, 1], "installment_x_conditional_deficit": [-1, 1, 1, -1, 1], "installment_x_outflow_pressure": [1, -1, -1, 1, 1], "total_burden_x_sparse_allocation": [1, 1, -1, 1, 1]}.

Submission is supplied as requested for an optional exploratory check, not promoted as a better leaderboard candidate. Exact friend method remains unknown.
