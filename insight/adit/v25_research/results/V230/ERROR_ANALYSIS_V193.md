# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 6 | 251 | 0.3665338645418327 | 0.3426294820717131 |
| 0.200000 | -7 | 373 | 0.21715817694369974 | 0.2359249329758713 |
| 0.323686 | 7 | 423 | 0.13947990543735225 | 0.12293144208037825 |
| 0.400000 | 7 | 466 | 0.08798283261802575 | 0.07296137339055794 |
| 0.500000 | 2 | 440 | 0.07272727272727272 | 0.06818181818181818 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003104 | -0.000043 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002835 | -0.000073 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001566 | -0.000154 |
| aux_account_months | zero | 21000 | 2461 | +0.000939 | -0.000224 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000483 | +0.000037 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.000344 | +0.000564 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000304 | -0.000072 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000782 | -0.000048 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000990 | -0.000025 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001131 | -0.000091 |
| employment_type | self_employed | 20987 | 2895 | -0.001190 | +0.000033 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.001459 | +0.000125 |
| employment_type | variable_income | 15540 | 2180 | -0.001558 | -0.000332 |
| baseline_risk_band | under005 | 29938 | 544 | -0.001965 | -0.000119 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002218 | -0.000463 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.002504 | -0.001354 |
| employment_type | salaried | 33473 | 4432 | -0.002658 | -0.000050 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002705 | +0.000024 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.