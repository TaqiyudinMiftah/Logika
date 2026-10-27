# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -3 | 125 | 0.36 | 0.384 |
| 0.200000 | 10 | 226 | 0.24778761061946902 | 0.20353982300884957 |
| 0.323686 | -13 | 272 | 0.10661764705882353 | 0.15441176470588236 |
| 0.400000 | -4 | 290 | 0.0896551724137931 | 0.10344827586206896 |
| 0.500000 | -2 | 306 | 0.058823529411764705 | 0.06535947712418301 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.008609 | +0.001930 |
| baseline_risk_band | under005 | 29853 | 535 | +0.007661 | +0.002591 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.002145 | +0.001235 |
| employment_type | variable_income | 15540 | 2180 | +0.001597 | +0.000253 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001206 | +0.000430 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000208 | +0.000166 |
| employment_type | salaried | 33473 | 4432 | -0.000021 | +0.000364 |
| aux_account_months | zero | 21000 | 2461 | -0.000206 | +0.000238 |
| employment_type | self_employed | 20987 | 2895 | -0.000305 | +0.000216 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000365 | +0.000387 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000730 | +0.000284 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000795 | +0.000387 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000801 | +0.000227 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000950 | +0.000145 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001125 | +0.000330 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001510 | +0.000308 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001647 | +0.000310 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.002607 | +0.000866 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.