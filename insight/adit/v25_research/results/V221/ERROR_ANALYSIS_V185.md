# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 1 | 17 | 0.35294117647058826 | 0.29411764705882354 |
| 0.200000 | -2 | 23 | 0.21739130434782608 | 0.30434782608695654 |
| 0.323686 | 0 | 37 | 0.10810810810810811 | 0.10810810810810811 |
| 0.400000 | -2 | 34 | 0.029411764705882353 | 0.08823529411764706 |
| 0.500000 | -1 | 40 | 0.025 | 0.05 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| employment_type | self_employed | 20987 | 2895 | +0.000760 | +0.000018 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000726 | +0.000023 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000670 | +0.000023 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000638 | +0.000021 |
| aux_account_months | zero | 21000 | 2461 | +0.000586 | -0.000012 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000578 | +0.000023 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000318 | -0.000001 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000282 | -0.000002 |
| employment_type | salaried | 33473 | 4432 | +0.000199 | +0.000021 |
| employment_type | variable_income | 15540 | 2180 | +0.000157 | -0.000012 |
| baseline_risk_band | under005 | 29965 | 546 | +0.000072 | +0.000189 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.000020 | +0.000179 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000035 | +0.000010 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.000081 | -0.000033 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000088 | +0.000033 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000239 | +0.000013 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.000518 | -0.000029 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001991 | +0.000021 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.