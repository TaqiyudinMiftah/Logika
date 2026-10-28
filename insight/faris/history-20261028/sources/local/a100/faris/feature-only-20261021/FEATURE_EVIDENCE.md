# Matched feature-only evidence

Frozen reference: KumoLarge90, original50k/6k/14kroles,5foldseed31027,headL2.003/strength.5,spline77,75:25probabilityblend. BaselineKS.5302185367/AUC.8470258500.

| Feature comparison | KS | AUC | Delta KS | Delta AUC | KS fold wins |
|---|---:|---:|---:|---:|---:|
| repayment_cashflow | 0.52944352 | 0.84711159 | -0.00077501 | +0.00008574 | 2/5 |
| bureau_trajectory | 0.52933141 | 0.84666848 | -0.00088713 | -0.00035737 | 1/5 |
| shopping_change | 0.52947112 | 0.84677274 | -0.00074741 | -0.00025311 | 1/5 |
| matched_maturity | 0.53060463 | 0.84705383 | +0.00038609 | +0.00002798 | 3/5 |
| joint | 0.53014483 | 0.84705758 | -0.00007371 | +0.00003173 | 2/5 |
| spline_only | 0.53012094 | 0.84702783 | -0.00009760 | +0.00000198 | 0/5 |
| both | 0.53063610 | 0.84705609 | +0.00041757 | +0.00003024 | 3/5 |
| cluster | 0.52947465 | 0.84701053 | -0.00074389 | -0.00001532 | 2/5 |
| eligible_denominators | 0.52896366 | 0.84694092 | -0.00125487 | -0.00008493 | 3/5 |
| mask_preserved | 0.53027831 | 0.84701819 | +0.00005977 | -0.00000766 | 4/5 |
| coverage_information | 0.52958530 | 0.84710390 | -0.00063324 | +0.00007805 | 1/5 |
| support | 0.52921368 | 0.84693975 | -0.00100486 | -0.00008610 | 3/5 |
| conditional_account | 0.53030642 | 0.84680314 | +0.00008788 | -0.00022271 | 3/5 |
| account_trait | 0.52969090 | 0.84700433 | -0.00052763 | -0.00002152 | 1/5 |
| true | 0.53096653 | 0.84692960 | +0.00074799 | -0.00009625 | 2/5 |
| shuffled | 0.52674470 | 0.84631623 | -0.00347384 | -0.00070962 | 1/5 |
| bands | 0.52952560 | 0.84711899 | -0.00069294 | +0.00009314 | 3/5 |
| shuffled_bands | 0.52716897 | 0.84645699 | -0.00304957 | -0.00056886 | 1/5 |

The maturity-only replacement in the fixedhead+spline is provisional; simultaneous conditional intervals include zero. Other point gains are not evidence of fresh private generalization. .54 remains the target. No confirmation partition is opened by this reporting script.

The field builder and parity/poisoning/invariance evidence are in handoff/. Every distinct fitted model, transform and failed result remains in its original study. Research is bounded by the original deadline; this snapshot neither resets it nor admits another recipe.

This table counts full five-fold recipe comparisons, including controls/placements as described; it does not equate these with independent discoveries or inflate counts with folds/retries. Null fit times are genuinely unavailable in this snapshot.
