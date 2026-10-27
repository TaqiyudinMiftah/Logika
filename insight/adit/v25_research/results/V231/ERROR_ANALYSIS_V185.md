# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 10 | 243 | 0.38271604938271603 | 0.34156378600823045 |
| 0.200000 | -6 | 364 | 0.21703296703296704 | 0.23351648351648352 |
| 0.323686 | 14 | 418 | 0.14354066985645933 | 0.11004784688995216 |
| 0.400000 | 5 | 455 | 0.08791208791208792 | 0.07692307692307693 |
| 0.500000 | 2 | 439 | 0.07517084282460136 | 0.07061503416856492 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.003634 | -0.000044 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003603 | -0.000014 |
| aux_account_months | zero | 21000 | 2461 | +0.001925 | -0.000210 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001546 | -0.000134 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.001024 | +0.000724 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000379 | -0.000032 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000341 | +0.000060 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000139 | -0.000068 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000111 | -0.000027 |
| employment_type | self_employed | 20987 | 2895 | -0.000251 | +0.000053 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000273 | +0.000006 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000314 | +0.000345 |
| employment_type | variable_income | 15540 | 2180 | -0.001236 | -0.000300 |
| baseline_risk_band | under005 | 29965 | 546 | -0.001708 | +0.000020 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002651 | +0.000040 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.002657 | -0.001252 |
| employment_type | salaried | 33473 | 4432 | -0.002870 | -0.000015 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.004379 | -0.000356 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.