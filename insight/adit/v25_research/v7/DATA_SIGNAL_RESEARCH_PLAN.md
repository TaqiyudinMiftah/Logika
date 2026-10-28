# Raw-data signal research V243-V250

This follows the V221 public regression audit. V25 remains the public incumbent0.56822; V221 public0.56614 is verified against exact CSV bytes. Highest pooled developmentKS is not a promotion rule.

The new cache reads all4,359,817 actual item rows and validates100,000 customer blocks against application cutoffs. Raw e-commerce SHA256495b3859f2146f87e7f8a383b308770af3d48342e3e1acdb45d7f1bed2bab742 matches node480. Processing stores one customer's events at a time; observed runtime46.5seconds locally. No target labels enter construction. Cancelled baskets do not enter these new price statistics. Every item's event must be strictly before its selected application. Item values are divided by positive quantity; order totals are checked once per order and never multiplied by item count.

Two data structures motivate the investigation:

1. Own-category price movement: compare recent0:30d unit prices with30:180d within the same customer and category, shrink low-support changes using a fixed effective-count factor, and measure breadth/dispersion across at least2 comparable categories. This differs from an aggregate price mean and from the older item-level z-score family: category composition is held fixed and equal category evidence/reliability is explicit. Without SKU/quality identifiers it remains a product-choice proxy.
2. Within-order price dependence: after centering each item on the customer's own180d category mean, estimate common price variation and dispersion only across different categories within the same basket. Equal weight per order prevents a three-item basket counting as three independent orders. Require3 multicategory orders and shrink covariance by n/(n+5). This differs from basket entropy, item-value concentration and a global latent price variance: it estimates within-customer dependence among items.

Cross-source hypotheses combine price movement/common variation with observed installment burden, existing burden and a bounded liquidity shortfall. No new DPD derivative or population price normalization is introduced. Account aggregates inherit the baseline's documented/unverified provenance; the whole pipeline is not newly certified point-in-time safe.

The cache contains8 support/availability variables and8 value/cross variables. Support is deliberately shared by ALL experiments so coverage cannot masquerade as new price information. New values retainNaN when undefined. Their standardized model contribution is0 when unavailable, with the same explicit support indicators present in the support-only control. Each additional input has one linear coefficient with L2.01; the original18-input V25 nonlinear design keeps L2.003. This avoids creating repeated missing-indicator columns for each new value and limits new head capacity to98-106 coefficients, versus145 for spectralV167.

| ID | Test |
|---|---|
| V243 | Baseline plus common support/availability only |
| V244 | Support plus within-category price trajectory |
| V245 | Support plus within-order common variation/dispersion |
| V246 | Support plus price-capacity cross-source information |
| V247 | All three value groups plus shared support |
| V248 | Full representation minus price trajectory |
| V249 | Full representation minus order coherence |
| V250 | Full representation minus capacity crosses |

The first scan does NOT establish new predictive utility. Price-trajectory univariate AUC is roughly.52-.53 on the two calendar halves, while covariance alone is around.50. Coverage has a stronger univariate relationship. Binned V25 residual intervals mostly include zero. The unchanged matched five-fold diagnostic models must determine whether any values add information beyond support and the existing baseline. H1/H2 are descriptive stability checks on already researched data, not independent holdouts. Negative results must be kept.

Preserve seed31027 and original50k context/6k correction/14k validation for each fold. Kumo/spline caches remain unchanged and Kumo100%frozen. No neural backbone training is needed. Because node480's16GiB RAM cgroup is full, CPU-only heads can execute locally from authenticated caches and then upload checkpoints/OOF/CSV with byte verification to the original node workspace. Native process locks are verified. Scikit-learn1.7.2 and PyArrow22 match the node; other package versions are recorded. Replaying all five original V25 validation partitions passes, and original test-mean prediction error is3.33e-16. This checks cached inference parity, not byte-equivalence of every future refit across OS/libraries.

Before every experiment, fetch/review shared GitHubHEAD and reject unexpected peer changes. Every result retains original source, cache hashes, all5 checkpoints, OOF, submission, paired KS/AUC bootstrap vsV25/V185/V193 and against support-onlyV243, source/coverage errors and overfitting caveats. The rolling queue retains3 pending ready jobs. No automatic Kaggle submission.
