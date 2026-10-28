# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 2 | 94 | 0.3723404255319149 | 0.35106382978723405 |
| 0.200000 | 2 | 134 | 0.26119402985074625 | 0.2462686567164179 |
| 0.323686 | -11 | 168 | 0.08928571428571429 | 0.15476190476190477 |
| 0.400000 | -2 | 179 | 0.11173184357541899 | 0.12290502793296089 |
| 0.500000 | -3 | 170 | 0.029411764705882353 | 0.047058823529411764 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6080 | 3493 | +0.004657 | -0.000677 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.003090 | -0.000294 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001038 | -0.000701 |
| orders90_exposure_band | zero | 3428 | 106 | +0.000301 | +0.000048 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000268 | -0.000178 |
| employment_type | salaried | 33473 | 4432 | +0.000016 | -0.000123 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000249 | -0.000042 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000315 | -0.000091 |
| employment_type | variable_income | 15540 | 2180 | -0.000361 | -0.000061 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | -0.000449 | -0.000150 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000619 | -0.000091 |
| aux_account_months | zero | 21000 | 2461 | -0.001057 | -0.000019 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.001076 | -0.000159 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001184 | -0.000105 |
| employment_type | self_employed | 20987 | 2895 | -0.001326 | -0.000192 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001425 | -0.000152 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.001665 | -0.000030 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001773 | -0.000168 |
| baseline_risk_band | under005 | 29853 | 535 | -0.001780 | -0.000892 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002045 | -0.000159 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.002048 | -0.000693 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.003163 | -0.000140 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.