# One ablation of selected spectral extrema

The full 11-field temporal block reached KS 0.5309665315 and AUC 0.8469295983, with only two KS fold wins versus the maturity candidate. Its shuffled control reached 0.5267447011 / 0.8463162307. Independent verification replayed all 20 models and checked the complete 70,000-row OOF metrics. The real-time block is not promoted: pooled KS rose, AUC declined and fold effects were inconsistent.

## Hypothesis

Remove the four fields that select the largest periodogram peak and its period, for spending and order counts. Selecting an extreme across many frequencies may make these fields variable in sparse histories. Keep the six predefined power bands (periods 6–8, 12–16 and 25–35 days for both channels) and the coverage flag. This is one explicit ablation, not a search over periods or subsets.

Reuse exactly the verified 100,000-row true and shuffled features. Drop the same four columns from each condition. Verify source hashes and bit-identical subsets. Preserve the original 180 full-day window, completed unique-order semantics, eligibility masks and within-customer shuffle seed 20261127. The same permutation moves spending and counts together. No new transform is fitted.

## Frozen comparison

The maturity pipeline has 18 correction and 77 spline inputs. Adding seven fields produces 25 and 84 inputs; the earlier full block had 29 and 88. Both new conditions have identical width. Preserve Kumo90 predictions, context/head/validation roles, original five folds, seeds, training budgets, correction L2 .003 and strength .5, spline C .05 / five knots / degree 3 / newton-cholesky / 80 iterations / tolerance 1e-5, and the 75:25 probability blend.

Finish both full five-fold conditions before scoring: ten correction heads and ten spline fits, all retained. Record pooled KS/AUC/log loss, fold effects, feature lineage and runtime. Independently replay serialized models and verify empirical-CDF KS and rank AUC.

Predeclare four contrasts: real bands versus original baseline, maturity candidate, shuffled bands, and the previous full 11-field block. Use 1,000 paired fold/class bootstrap replications with Bonferroni adjustment for four contrasts and two metrics. These conditional intervals do not correct historical label reuse, adaptive hypothesis choice or training uncertainty. A KS-only improvement does not qualify for promotion.

No frequency, window or subset grid follows if this ablation fails joint improvement. No model or blend search, confirmation opening or competition CSV. The 0.54 target remains unmet. Preserve absolute deadline 1792585871.823748 and admission stop at 16:31 WIB.
