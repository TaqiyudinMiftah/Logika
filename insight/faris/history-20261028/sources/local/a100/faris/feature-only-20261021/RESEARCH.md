# Feature-only research, fixed V25 modeling procedure

The goal is 0.54 pooled tie-aware KS on the original 70,000-row, five-fold OOF protocol. It remains unmet. A feature gain must be attributed to a matched model, with KS and AUC both reported; no leaderboard feedback is used to select these proposals.

## Evidence reconstructed before proposing features

V25 retains the frozen Kumo Large90 classifier, 50,000 context rows, eight estimators, seeds 2026+fold, a 6,000-row offset corrector with L2 .003 and strength .5, spline77, and a 75:25 probability blend. It adds six columns to the original twelve correction fields: negative/nonnegative cash-flow months with an observed allocation, recent account months, completed orders90, shopping observation span, and allocated-installment count. The existing raw rebuild and all 65 retained attribution heads were independently checked. This study repeats the five V25 checkpoint replays before any new fit.

The correct teammate session and feature session were inspected through their extracted dialogue and feature-study evidence. Price/burden interactions, trajectory composites, entropy, lifecycle, specialists and actuarial bundles have related prior experiments. They are not new discoveries here. Some apparent improvements changed regularization as well as features: the matched V07/V101 attribution shows why we must freeze regularization.

Historical V25 OOF KS .5302846600/AUC .8470292789 used inherited shared covariate estimates. The repaired fold-local counterpart is .5302185367/.8470258500. Both are retained and distinguished. The matched scientific baseline uses repaired priors; historical V25 is a reproduction anchor, not an interchangeable comparator.

## Frozen first three hypotheses

1. **Repayment and cash flow:** V25 counts months and allocated installments but not their contractual exposure. Borrowers with the same count may face very different scheduled amounts and income coverage. Add six row-local fields: due/income, negative-cash-flow share of due amount, account overlap, allocated due-amount shares in negative/nonnegative months, and recent change in negative-month due share. Use due events aged at least30 days and completed account months. Allocation presence is observation, not settlement. Recorded net cash flow already includes outflows; do not claim to reconstruct debt-free surplus. No new DPD or amount-paid field is used.
2. **Bureau trajectories:** pooled score levels can conflate providers and reporting ages. Compare earliest/latest reports within the same provider, with at least90 days separation; calculate annualized score and log-debt changes, provider slope disagreement, latest report age and reporting lag. Never sum overlapping provider debts. This has related historical trend features: the contrast tests whether these specific six fields add information beyond the baseline.
3. **Shopping change:** changes in average purchase prices can arise from category mix. Compare recent90 days with days91–365 within each observed category, then give matched categories equal weight. Add price change and uncertainty, matched category count, category-composition turnover, cancellation-share change and purchase-gap change. Cancelled baskets enter cancellation behavior, not realized spending/price. No population price reference is estimated.

Actual unsupervised raw-data audit precedes fitting. Constant or unavailable features must be explicitly excluded before the plan is sealed; no performance-based feature selection is allowed. Each six-field block is tested separately. This is three financial hypotheses, not a combinatorial catalogue. The first placement is the existing correction stage; a negative result applies to that placement, not every model's ability to use the feature.

## Fixed modeling and validation

No new architecture, checkpoint, context size, learning schedule, seed, regularization, spline configuration, estimator count, correction strength or ensemble weight is selected. The saved repaired Kumo scores and spline predictions remain identical. Each correction fit sees its original6,000 labels only; median imputation, spline knots, scales, missing indicators and coefficients are fitted there. Adding inputs changes the design-matrix width but not the estimator or training budget. The baseline is refitted with the identical code/budget to separate refit rounding from feature effects. All five folds must finish before any candidate performance is opened.

Report pooled KS/AUC/log loss, original fold changes, aligned predictions, full retained models, source/input hashes, fitting time, parameter counts and conditional paired uncertainty. Three candidate comparisons and two metrics are predeclared; simultaneous intervals use Bonferroni across all six. The primary hypothesis is repayment/cash-flow exposure; other two are predeclared secondary comparisons. No post-hoc union, blend search, selective folding or public probing. Require positive KS/AUC, >=3/5 KS fold wins and no fold KS loss >.005 for provisional support; uncertainty can leave the result inconclusive. These are reused development labels, not fresh private validation.

The inherited deadline remains 2026-10-21 19:31:11.823748 WIB. No new12-hour window. Existing non-feature studies are legacy evidence and cannot change the fixed model or nominate a feature-only winner. Confirmation is not opened during this stage; any terminal confirmation needs a candidate frozen first and an explicit account of prior label reuse. No submission is created.

## Expected outputs

FEATURE_AUDIT.json and features.parquet (100k applicants, no labels); BASELINE_REPRODUCTION.json; MODEL_LOCK.json; SCIENTIFIC_GATE.json; retained per-fold correction models and predictions; POINT_RESULT.json/OOF.npz; conditional uncertainty, feature lineage, reproduction commands and a supported/negative/inconclusive report. New trials must be justified by the evidence, not by a desired experiment count.
