# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -2 | 23 | 0.4782608695652174 | 0.5652173913043478 |
| 0.200000 | 2 | 44 | 0.2727272727272727 | 0.22727272727272727 |
| 0.323686 | -7 | 60 | 0.1 | 0.21666666666666667 |
| 0.400000 | 4 | 52 | 0.09615384615384616 | 0.019230769230769232 |
| 0.500000 | 1 | 67 | 0.07462686567164178 | 0.05970149253731343 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.004657 | +0.000473 |
| baseline_risk_band | under005 | 29938 | 544 | +0.002186 | +0.000571 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000672 | +0.000160 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | +0.000559 | +0.000500 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000432 | +0.000098 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000398 | -0.000199 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000361 | +0.000094 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000360 | +0.000081 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000249 | +0.000087 |
| employment_type | self_employed | 20987 | 2895 | +0.000028 | +0.000071 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000227 | +0.000079 |
| aux_account_months | zero | 21000 | 2461 | -0.000327 | +0.000084 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.000377 | +0.000546 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000469 | +0.000095 |
| employment_type | variable_income | 15540 | 2180 | -0.000528 | +0.000027 |
| employment_type | salaried | 33473 | 4432 | -0.000555 | +0.000109 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000578 | +0.000087 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000875 | +0.000076 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.