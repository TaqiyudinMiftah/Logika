# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 6 | 262 | 0.38549618320610685 | 0.36259541984732824 |
| 0.200000 | 1 | 407 | 0.22604422604422605 | 0.22358722358722358 |
| 0.323686 | -4 | 486 | 0.12551440329218108 | 0.1337448559670782 |
| 0.400000 | 0 | 527 | 0.0872865275142315 | 0.0872865275142315 |
| 0.500000 | 4 | 510 | 0.07450980392156863 | 0.06666666666666667 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003414 | -0.000299 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003257 | -0.000330 |
| aux_account_months | zero | 21000 | 2461 | +0.000964 | -0.000434 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.000477 | +0.000347 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000308 | -0.000103 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000452 | -0.000262 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.000568 | -0.001359 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000624 | -0.000093 |
| baseline_risk_band | under005 | 29938 | 544 | -0.000862 | -0.000923 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000945 | -0.000149 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.001506 | -0.000641 |
| employment_type | variable_income | 15540 | 2180 | -0.002005 | -0.000405 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002204 | -0.000133 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002269 | -0.000094 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002576 | -0.000107 |
| employment_type | self_employed | 20987 | 2895 | -0.002765 | -0.000191 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003071 | -0.000887 |
| employment_type | salaried | 33473 | 4432 | -0.003100 | -0.000126 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.