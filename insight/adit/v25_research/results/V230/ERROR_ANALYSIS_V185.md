# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 5 | 250 | 0.368 | 0.348 |
| 0.200000 | -7 | 367 | 0.2125340599455041 | 0.23160762942779292 |
| 0.323686 | 12 | 425 | 0.14588235294117646 | 0.11764705882352941 |
| 0.400000 | 4 | 461 | 0.0824295010845987 | 0.0737527114967462 |
| 0.500000 | 2 | 448 | 0.07366071428571429 | 0.06919642857142858 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.003346 | -0.000095 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003335 | -0.000060 |
| aux_account_months | zero | 21000 | 2461 | +0.001382 | -0.000255 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001342 | -0.000152 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000260 | +0.000052 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.000230 | +0.000561 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000027 | -0.000081 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000290 | -0.000047 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000466 | -0.000017 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000589 | -0.000091 |
| employment_type | self_employed | 20987 | 2895 | -0.000831 | +0.000034 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.001087 | +0.000145 |
| employment_type | variable_income | 15540 | 2180 | -0.001249 | -0.000326 |
| employment_type | salaried | 33473 | 4432 | -0.002531 | -0.000054 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002651 | +0.000042 |
| baseline_risk_band | under005 | 29965 | 546 | -0.002706 | -0.000103 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.003679 | -0.001423 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.004379 | -0.000363 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.