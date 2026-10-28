# A matched actuarial measurement-error comparison

The manual feature transcript identifies category-relative prices, latent bureau
quality, income reliability and delinquency propensity as useful measurements.
It also shows that conditional signal can disappear after controlling for the
existing model. The V07/V101 attribution study found little stable new signal;
the joint-feature and learned-risk representation studies did not beat the
retained incumbent on both KS and AUC. More feature names are not the objective.

This study asks one different question: does integrating measurement uncertainty
in a compact probability model improve predictions beyond treating posterior
means as known? No matching quadrature experiment was found in the inspected
feature-session dialogue. That is a coverage statement, not proof of novelty.

Two models use identical covariates, priors, folds, ridge (.001) and monotonic
latent-risk signs. One plugs posterior means into a logistic GLM. The other
integrates logistic probabilities over four approximate independent posterior
normals using 128-point Gauss-Hermite quadrature. Its likelihood differentiates
through both mean and coefficient-dependent variance. Synthetic finite-difference
gradient checks, a zero-variance equivalence check, optimizer KKT checks and a
256-point prediction check are mandatory. There is no hyperparameter search.

Measurement models fit only the 56,000 outer-training customers: two-way
category/customer log prices with item/order noise; matched provider offsets and
score noise; employment-specific log-income noise; beta-binomial late-payment
propensity. Default labels are not arguments to the measurement estimator.
Missing measurements retain their training-prior variance and explicit flags.
The normal logit-beta approximation and independent-noise assumptions are
approximations. No exact data-generator recovery or causal interpretation is
asserted. A static bureau model is used because the preceding unlabelled
measurement experiment showed negligible dynamic prediction improvement.

The primary recipe keeps original corrected Kumo predictions and replaces the
25% spline stream with the integrated model. The plug-in replacement isolates
the uncertainty effect. A frozen 5% addition to the old incumbent is secondary;
no weights are optimized. Standalone scores are diagnostic. All original five
folds must finish, with each of 70,000 applicants predicted exactly once.
Models/transformations/posteriors/raw sufficient statistics remain retained.

The deadline is inherited: 1792585871.823748 (2026-10-21 19:31:11.823748 WIB).
It does not restart on process recovery. PC3 is reached only from PC2, with five
CPU workers and two numerical threads per worker, no GPU reservation. Execution
is an ordinary finite Python controller. An incomplete fit cannot enter a final
leaderboard. KS is independently checked with SciPy; AUC uses tied ranks.
Conditional paired bootstrap intervals account for the reported metric
contrasts, but cannot erase historical selection or repeated label exposure.

Item prices include cancelled baskets as preference observations, never as
actual spending. All measured events pass decision-time cutoffs; only complete
calendar months contribute to income. Repayment max-DPD snapshot timing remains
unresolved and is disclosed. Customer/application/order IDs are join keys only.
No competition submission is generated and no score increase is guaranteed.
