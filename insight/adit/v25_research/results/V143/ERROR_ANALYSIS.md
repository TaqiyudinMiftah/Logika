# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 0 | 405 | 0.38271604938271603 | 0.38271604938271603 |
| 0.200000 | 15 | 625 | 0.2352 | 0.2112 |
| 0.323686 | -61 | 783 | 0.10472541507024266 | 0.1826309067688378 |
| 0.400000 | -13 | 885 | 0.09265536723163842 | 0.10734463276836158 |
| 0.500000 | -23 | 915 | 0.05683060109289618 | 0.08196721311475409 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29853 | 535 | +0.005935 | -0.003852 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.002863 | -0.004328 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000312 | -0.006250 |
| employment_type | salaried | 33473 | 4432 | -0.001370 | -0.000769 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.002118 | -0.000645 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.002434 | -0.005085 |
| aux_allocation_count | zero | 23732 | 2895 | -0.003152 | -0.001006 |
| employment_type | self_employed | 20987 | 2895 | -0.003290 | -0.001644 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.003703 | -0.001313 |
| aux_account_months | zero | 21000 | 2461 | -0.004233 | -0.001220 |
| employment_type | variable_income | 15540 | 2180 | -0.005075 | -0.001610 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.005820 | -0.001128 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.005894 | -0.001598 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.006785 | -0.001216 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.007002 | -0.001170 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.008674 | -0.005318 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.009350 | -0.001309 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.009911 | -0.001758 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.