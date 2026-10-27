# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 2 | 101 | 0.37623762376237624 | 0.3564356435643564 |
| 0.200000 | 7 | 169 | 0.23076923076923078 | 0.1893491124260355 |
| 0.323686 | -19 | 219 | 0.0821917808219178 | 0.1689497716894977 |
| 0.400000 | -1 | 249 | 0.09236947791164658 | 0.0963855421686747 |
| 0.500000 | 2 | 249 | 0.06827309236947791 | 0.060240963855421686 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001973 | -0.000524 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.001480 | +0.001009 |
| aux_account_months | zero | 21000 | 2461 | +0.001115 | +0.000049 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000949 | -0.000034 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000916 | -0.000040 |
| employment_type | salaried | 33473 | 4432 | +0.000258 | -0.000122 |
| employment_type | variable_income | 15540 | 2180 | +0.000031 | -0.000085 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000237 | -0.000079 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000274 | -0.000116 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000582 | +0.000048 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000842 | -0.000034 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000881 | -0.000039 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000935 | +0.000103 |
| employment_type | self_employed | 20987 | 2895 | -0.001049 | +0.000104 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001110 | -0.000041 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001273 | -0.000351 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.002484 | -0.000187 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.003848 | -0.000008 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.