# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 10 | 234 | 0.3803418803418803 | 0.33760683760683763 |
| 0.200000 | -1 | 358 | 0.22346368715083798 | 0.22625698324022347 |
| 0.323686 | 11 | 419 | 0.1360381861575179 | 0.10978520286396182 |
| 0.400000 | 2 | 444 | 0.08558558558558559 | 0.08108108108108109 |
| 0.500000 | 5 | 435 | 0.07586206896551724 | 0.06436781609195402 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.003269 | -0.000011 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003057 | +0.000017 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.002109 | -0.000130 |
| aux_account_months | zero | 21000 | 2461 | +0.001669 | -0.000187 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000985 | +0.000056 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.000760 | +0.000803 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.000267 | +0.000552 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000073 | +0.000018 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000259 | +0.000035 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000299 | -0.000004 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000563 | -0.000048 |
| employment_type | variable_income | 15540 | 2180 | -0.000670 | -0.000283 |
| employment_type | self_employed | 20987 | 2895 | -0.000734 | +0.000067 |
| baseline_risk_band | under005 | 29965 | 546 | -0.001539 | +0.000053 |
| employment_type | salaried | 33473 | 4432 | -0.002456 | +0.000030 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.002496 | -0.001061 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002549 | +0.000059 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.006597 | -0.000304 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.