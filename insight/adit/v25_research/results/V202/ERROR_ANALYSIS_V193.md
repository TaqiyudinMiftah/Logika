# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 8 | 245 | 0.37142857142857144 | 0.33877551020408164 |
| 0.200000 | -5 | 365 | 0.22465753424657534 | 0.23835616438356164 |
| 0.323686 | 12 | 425 | 0.13647058823529412 | 0.10823529411764705 |
| 0.400000 | 8 | 460 | 0.09565217391304348 | 0.0782608695652174 |
| 0.500000 | 0 | 440 | 0.07272727272727272 | 0.07272727272727272 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003335 | -0.000061 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.003209 | -0.000160 |
| aux_allocation_count | zero | 23732 | 2895 | +0.003199 | -0.000108 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.001567 | +0.000042 |
| aux_account_months | zero | 21000 | 2461 | +0.001241 | -0.000270 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000355 | -0.000049 |
| baseline_risk_band | over04 | 6096 | 3493 | +0.000200 | +0.000455 |
| baseline_risk_band | under005 | 29938 | 544 | -0.000149 | +0.000133 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.000507 | +0.000266 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000558 | -0.000023 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.000589 | -0.001291 |
| employment_type | self_employed | 20987 | 2895 | -0.000596 | +0.000073 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000698 | +0.000005 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000793 | -0.000076 |
| employment_type | variable_income | 15540 | 2180 | -0.000817 | -0.000315 |
| employment_type | salaried | 33473 | 4432 | -0.002756 | -0.000046 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002866 | +0.000071 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.003962 | -0.000360 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.