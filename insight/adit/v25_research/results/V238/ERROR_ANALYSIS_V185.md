# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 10 | 294 | 0.3877551020408163 | 0.35374149659863946 |
| 0.200000 | 15 | 486 | 0.24279835390946503 | 0.21193415637860083 |
| 0.323686 | -2 | 586 | 0.11945392491467577 | 0.12286689419795221 |
| 0.400000 | -4 | 593 | 0.08094435075885328 | 0.08768971332209106 |
| 0.500000 | -4 | 612 | 0.07026143790849673 | 0.07679738562091504 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.004537 | +0.001358 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | +0.003703 | -0.000914 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.003214 | +0.000399 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002879 | +0.000055 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002156 | -0.000045 |
| aux_account_months | zero | 21000 | 2461 | +0.001245 | -0.000260 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.001019 | +0.002646 |
| baseline_risk_band | under005 | 29965 | 546 | +0.000851 | +0.001136 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000330 | +0.000042 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000213 | +0.000048 |
| employment_type | variable_income | 15540 | 2180 | -0.000002 | -0.000362 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001365 | +0.000389 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001886 | +0.000090 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001966 | +0.000228 |
| employment_type | self_employed | 20987 | 2895 | -0.002402 | +0.000186 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002432 | +0.000215 |
| employment_type | salaried | 33473 | 4432 | -0.002441 | +0.000242 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002844 | -0.001266 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.