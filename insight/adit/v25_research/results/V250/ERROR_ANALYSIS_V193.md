# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 9 | 269 | 0.379182156133829 | 0.34572490706319703 |
| 0.200000 | -4 | 409 | 0.2200488997555012 | 0.22982885085574573 |
| 0.323686 | 2 | 477 | 0.1278825995807128 | 0.12368972746331237 |
| 0.400000 | 6 | 512 | 0.095703125 | 0.083984375 |
| 0.500000 | 3 | 513 | 0.07407407407407407 | 0.0682261208576998 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003503 | -0.000249 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003401 | -0.000239 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.002359 | -0.000017 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.002048 | -0.000184 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.000784 | -0.001514 |
| aux_account_months | zero | 21000 | 2461 | +0.000666 | -0.000360 |
| baseline_risk_band | under005 | 29938 | 544 | -0.000196 | -0.001044 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000772 | -0.000232 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001069 | -0.000256 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.001944 | -0.000767 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001991 | -0.000926 |
| employment_type | variable_income | 15540 | 2180 | -0.002057 | -0.000417 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002242 | -0.000182 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002366 | -0.000217 |
| employment_type | self_employed | 20987 | 2895 | -0.002406 | -0.000185 |
| employment_type | salaried | 33473 | 4432 | -0.002526 | -0.000183 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002611 | -0.000185 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.003182 | -0.000133 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.