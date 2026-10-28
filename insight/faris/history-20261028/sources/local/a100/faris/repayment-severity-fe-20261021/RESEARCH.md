# Joint repayment severity, fixed predictive model

The feature-only comparisons of cash-flow exposure, bureau trajectories and within-category shopping changes did not improve matched KS. Their retained evidence is in ../feature-only-20261021. A separate persistent-provider-error proposal was excluded without classifier fitting because its target-free diagnostic did not support the mechanism.

The existing feature builder fits independent empirical-Bayes beta-binomial rates for max DPD >=1, >=7, >=30, each with its own maturation eligibility and prior. These summarize the same nested events but do not pool information across severity categories. We will test one joint Dirichlet-multinomial severity posterior, not a new predictive architecture. Categories are maxDPD=0,1–6,7–29,30+, restricted to installments aged at least30 days at decision. The feature prior uses only the original50k context applicants in each outer fold, never correction/evaluation histories to fit shared parameters, and no default labels. Each applicant's own eligible history is transformed afterward.

Replace exactly two inputs in the18-field V25 correction: re_late_1_posterior and re_late_30_posterior. Keep all other features, raw Kumo90 scores, spline77, head architecture/budget/L2 .003/strength .5, original6k labels, fivefoldseed31027 and75:25 ensemble unchanged. No extra input dimensions are added.

Three fixed recipes: unchanged baseline; independent beta-binomial posteriors using the same >=30day events (maturity control); joint four-category Dirichlet-multinomial posterior on those events. Joint versus maturity control isolates the prior/encoding effect, while joint versus baseline assesses the complete feature replacement. No optimizer, regularization, model or ensemble selection. All15heads/all5folds finish before default metrics are opened. No partial-fold screening.

Count rebuilding and analytical posterior checks precede training. Prior optimization uses fixed positive alpha bounds and a fixed200iteration numerical budget. Retain every prior parameter vector, solver diagnostic, fit-ID hash, feature array, feature lineage and fitted head. Report pooled KS/AUC/log loss, per-fold effects, paired conditional bootstrap intervals for two primary contrasts x two metrics, corrected using Bonferroni. Only a joint KS/AUC gain with >=3originalfold KS wins and no fold KS loss >.005 is provisionally useful. Reused development labels and snapshot DPD timing remain limitations; no untouched confirmation, leaderboard claim or submission.

The Dirichlet-multinomial is an exchangeable severity model, not a claim that loan installments are independent in reality or that their generating process was recovered. A good prior fit is not evidence of default-prediction gain. The raw maxDPD snapshot qualification remains unresolved and inherited; no new future observation is introduced.

Original deadline remains1792585871.823748 (2026-10-21 19:31:11.823748 WIB). No fresh clock. The .54 pooled KS target is unchanged and unmet.

Reproduce preprocessing with `python source/build.py` in the dedicated remote directory; `source/run.py` will perform the fixed head comparison only after gates pass.
