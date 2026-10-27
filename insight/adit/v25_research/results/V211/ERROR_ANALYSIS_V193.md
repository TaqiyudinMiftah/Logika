# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 8 | 408 | 0.39705882352941174 | 0.37745098039215685 |
| 0.200000 | 24 | 629 | 0.24324324324324326 | 0.20508744038155802 |
| 0.323686 | -10 | 775 | 0.12387096774193548 | 0.1367741935483871 |
| 0.400000 | -2 | 867 | 0.09342560553633218 | 0.09573241061130335 |
| 0.500000 | -19 | 903 | 0.05758582502768549 | 0.07862679955703211 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.004046 | -0.000738 |
| baseline_risk_band | under005 | 29938 | 544 | +0.003969 | -0.000339 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000572 | -0.000496 |
| aux_allocation_count | zero | 23732 | 2895 | -0.001597 | -0.000287 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001648 | -0.000379 |
| aux_account_months | zero | 21000 | 2461 | -0.002852 | -0.000384 |
| baseline_risk_band | over04 | 6096 | 3493 | -0.002940 | -0.000499 |
| employment_type | salaried | 33473 | 4432 | -0.002946 | -0.000671 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.003018 | -0.000793 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.003189 | -0.000722 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.003444 | -0.000855 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.003838 | -0.000878 |
| employment_type | variable_income | 15540 | 2180 | -0.003925 | -0.000946 |
| employment_type | self_employed | 20987 | 2895 | -0.004243 | -0.000666 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.007925 | -0.004708 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.008987 | -0.000957 |
| baseline_risk_band | 005_to015 | 19478 | 1835 | -0.009780 | -0.007869 |
| baseline_risk_band | 015_to04 | 14488 | 3635 | -0.009857 | -0.002024 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.