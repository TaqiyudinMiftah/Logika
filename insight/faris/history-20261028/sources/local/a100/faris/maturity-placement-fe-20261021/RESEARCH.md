# Place the same two maturity-aligned features in the existing spline

The earlier predeclared maturity control scored KS .5306046301/AUC .8470538256 versus the identical-model repaired V25 baseline .5302185367/.8470258500. KS won3folds,tied1,lost1by.000032594. The joint Dirichlet feature did not improve KS. This is a small, adaptively discovered development lead, not confirmation or proof of generalization.

Test a single follow-up feature-placement hypothesis: use those same >=30day beta-binomial posterior means in both existing branches which consume them, rather than the correction branch alone. Replace exactly re_late_1_posterior and re_late_30_posterior. All other inputs and branches remain fixed: Kumo90/context50k/eightestimators/seeds2026+fold and raw probabilities; correction18/6000rows/L2.003/strength.5; spline77/56000rows/C.05/fiveknots/degree3/newton-cholesky/80iterations/tolerance1e-5; probabilityweights.75/.25; original5foldseed31027.

Only5new spline models are fitted. Retained correction models from the preceding study are reused with exact hashes. Four combinations are fixed before training: unchanged baseline; head-only maturity (already completed); spline-only maturity (attribution); both branches maturity (primary). These are placements of one feature definition, not weight search. No new feature columns, predictive architecture, seed or hyperparameter is admitted.

Population priors remain fitted on each50kcontext. Spline preprocessing/model fitting sees only56koutertrain, with labels assembled from the context packet and6kcorrection labels, never from an all-label frame. Query matrices contain no target. Every saved old spline must replay on the repaired features before fitting; new and old parameter configurations and transformed input widths must match. All5folds must finish before candidate scores are read.

Report pooled KS/AUC/logloss, folds, source/input/model hashes, retained models and exact checkpoint replay. Conditional paired bootstrap covers both versus baseline, both versus head-only and spline-only versus baseline (threecontrasts x twometrics), with Bonferroni95percent intervals. It does not remove adaptive feature selection or historical label reuse. Require positive KS and AUC and no major fold degradation to consider a nominee; .54 remains the target. Confirmation remains closed, and no submission is produced.

Original deadline1792585871.823748 remains unchanged. This run is CPU-only because it does not alter Kumo inputs. Use `python source/run.py` after the source/feature gate is sealed.
