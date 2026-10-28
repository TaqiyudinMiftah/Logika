# Structured correction research V238-V242

V25 remains the public incumbent (0.56822); V221 remains the numerical development incumbent before this batch (KS0.5316293525, AUC0.8471248959). These are new hypotheses, not expected improvements. No new competition submission is made automatically.

Evidence inspected before design: V224-V234 raw population/capacity combinations did not beat V193; V232 has pooledKS0.53018559/AUC0.84677733. V221's error analysis vsV25 has top10% defaults captured -6, zero-order segment(3428 borrowers,106 events) deltaKS-0.006918 and zero-allocation(23732 borrowers,2895 events) deltaKS-0.003086. These exploratory segment results are not independent discoveries and do not choose query gates. V200 roughness smoothing improved AUC to0.84731949 while KS fell to0.52956578, motivating alternatives to blanket smoothing.

Five different changes are isolated: V238 restricted cubic shape/boundaries; V239 group-sparse effect selection; V240 training-only PCA of the original correction basis; V241 bounded feature-leverage loss weights; V242 a Brier scoring objective. Exact fixed formulas, penalties, uncertainty and failure modes are in structured_head_plan.json and copied into each result. All use original18 V25 correction inputs and global Kumo predictions, unchanged original spline77 and75:25fusion. No new raw-table feature, DPD definition or target-based group is introduced.

The common design has50k frozenKumo context,6k correction training,14k validation per identical outer fold(seed31027). Every encoder, knot, median, PCA, leverage covariance/quantile and supervised coefficient is fit on the6k correction rows only. The context pool and outer validation labels are not used to fit these new heads. Existing global unsupervised references/static profiles/DPD snapshots retain inherited provenance uncertainty; this batch does not certify them point-in-time safe.

Preflight must pass all five branches before readiness: finite-difference gradients, convex group-proximal stationarity, natural cubic linear-tail identity, constant/missing predictors, unknown employment, finite query predictions and exact checkpoint serialization. Joblib custom transformers use an importable module name. Each experiment produces five checkpoints, OOF/CSV, paired bootstrap KS/AUC vsV25/V185/V193, fold comparisons, segment errors, optimizer diagnostics and master-ledger entry. Source hashes protect preflight and completed artifacts. Kumo stays100% frozen; CPU-only cached score reuse avoids further GPU/host allocations.

Research references establish methods, not LOGIKA performance:

- Restricted cubic splines have linear boundary tails: R splines documentation https://stat.ethz.ch/R-manual/R-devel/library/splines/html/ns.html . Our fixed four-knot truncated-power basis is implemented directly and tested; repeated knots use an explicitly recorded linear fallback.
- Group-level rather than individual-coefficient selection: Yuan & Lin(2006), https://www.columbia.edu/~my2550/papers/glasso.final.pdf . We adapt group sparsity to an offset logistic loss with ridge and standardized spline groups; this is not a reproduction of their empirical experiments.

Do not promote a candidate merely for a small pooledKS increase. Inspect deltaAUC, all folds, bootstrap uncertainty, head gap, low-coverage/high-risk errors and publicV185's decline vsV25. Reused outer folds and hundreds of prior trials create search overfitting; conditional bootstrap excludes that uncertainty. Independent generalization remains unestablished.
