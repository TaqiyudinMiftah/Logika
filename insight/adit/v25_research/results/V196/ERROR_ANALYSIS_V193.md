# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 7 | 76 | 0.4342105263157895 | 0.34210526315789475 |
| 0.200000 | -8 | 137 | 0.2116788321167883 | 0.27007299270072993 |
| 0.323686 | -3 | 155 | 0.13548387096774195 | 0.15483870967741936 |
| 0.400000 | 11 | 148 | 0.10810810810810811 | 0.033783783783783786 |
| 0.500000 | -2 | 141 | 0.05673758865248227 | 0.07092198581560284 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.002871 | +0.000026 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001991 | +0.000023 |
| aux_allocation_count | zero | 23732 | 2895 | +0.001650 | +0.000112 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.001558 | +0.000078 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001386 | -0.000115 |
| baseline_risk_band | under005 | 29938 | 544 | +0.001291 | +0.000539 |
| aux_account_months | zero | 21000 | 2461 | +0.001216 | +0.000058 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | +0.000969 | +0.000164 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000779 | +0.000004 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | +0.000297 | +0.000012 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.000001 | -0.000761 |
| employment_type | variable_income | 15540 | 2180 | -0.000198 | +0.000041 |
| employment_type | salaried | 33473 | 4432 | -0.000598 | +0.000030 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000802 | +0.000115 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001336 | -0.000012 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001378 | +0.000008 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001644 | -0.000007 |
| employment_type | self_employed | 20987 | 2895 | -0.001989 | -0.000020 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.