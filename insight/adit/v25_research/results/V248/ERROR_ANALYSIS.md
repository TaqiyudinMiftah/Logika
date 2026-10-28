# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -2 | 109 | 0.3577981651376147 | 0.3761467889908257 |
| 0.200000 | 5 | 161 | 0.2360248447204969 | 0.20496894409937888 |
| 0.323686 | -16 | 196 | 0.09693877551020408 | 0.17857142857142858 |
| 0.400000 | -7 | 214 | 0.08411214953271028 | 0.11682242990654206 |
| 0.500000 | 2 | 208 | 0.057692307692307696 | 0.04807692307692308 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.001618 | -0.000121 |
| orders90_exposure_band | zero | 3428 | 106 | +0.001505 | +0.000148 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000891 | -0.000560 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000646 | -0.000277 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000365 | -0.000220 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000360 | -0.000195 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000262 | -0.000234 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000068 | -0.000913 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.000394 | +0.000082 |
| employment_type | salaried | 33473 | 4432 | -0.000412 | -0.000085 |
| aux_account_months | zero | 21000 | 2461 | -0.000439 | -0.000155 |
| employment_type | variable_income | 15540 | 2180 | -0.000573 | -0.000065 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000624 | -0.000207 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | -0.000887 | -0.000149 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001111 | -0.000057 |
| employment_type | self_employed | 20987 | 2895 | -0.001520 | -0.000237 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001578 | -0.001055 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001580 | -0.000065 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001758 | -0.000108 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.001810 | -0.000229 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001854 | -0.000125 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.003473 | +0.000011 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.