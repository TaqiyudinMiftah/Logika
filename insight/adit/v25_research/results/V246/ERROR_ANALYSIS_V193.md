# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 9 | 257 | 0.3968871595330739 | 0.36186770428015563 |
| 0.200000 | 1 | 406 | 0.23399014778325122 | 0.2315270935960591 |
| 0.323686 | -2 | 476 | 0.11974789915966387 | 0.12394957983193278 |
| 0.400000 | -2 | 513 | 0.08576998050682261 | 0.08966861598440545 |
| 0.500000 | 2 | 492 | 0.07317073170731707 | 0.06910569105691057 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002792 | -0.000171 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002643 | -0.000209 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.002082 | +0.000470 |
| aux_account_months | zero | 21000 | 2461 | +0.000346 | -0.000347 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.000276 | -0.001680 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000367 | -0.000107 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000395 | -0.000169 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001233 | -0.000163 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001315 | -0.000199 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001618 | -0.000048 |
| employment_type | self_employed | 20987 | 2895 | -0.002088 | -0.000142 |
| employment_type | variable_income | 15540 | 2180 | -0.002246 | -0.000391 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002645 | -0.000103 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002673 | -0.000096 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.003116 | -0.000148 |
| employment_type | salaried | 33473 | 4432 | -0.003362 | -0.000096 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003735 | -0.000673 |
| baseline_risk_band | under005 | 29938 | 544 | -0.003800 | -0.000880 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.