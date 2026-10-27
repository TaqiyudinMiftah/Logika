# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -1 | 88 | 0.38636363636363635 | 0.3977272727272727 |
| 0.200000 | -11 | 137 | 0.1678832116788321 | 0.24817518248175183 |
| 0.323686 | -7 | 162 | 0.09259259259259259 | 0.13580246913580246 |
| 0.400000 | 2 | 176 | 0.09659090909090909 | 0.08522727272727272 |
| 0.500000 | 1 | 164 | 0.07926829268292683 | 0.07317073170731707 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| employment_type | variable_income | 15540 | 2180 | +0.002603 | -0.000027 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.001754 | +0.000564 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001281 | +0.000278 |
| baseline_risk_band | under005 | 29853 | 535 | +0.001173 | -0.000014 |
| employment_type | salaried | 33473 | 4432 | +0.000374 | +0.000118 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000349 | +0.000103 |
| aux_allocation_count | zero | 23732 | 2895 | +0.000260 | +0.000070 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000201 | +0.000044 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.000247 | +0.000191 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000456 | +0.000041 |
| aux_account_months | zero | 21000 | 2461 | -0.000521 | +0.000075 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000536 | +0.000028 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000552 | +0.000029 |
| employment_type | self_employed | 20987 | 2895 | -0.000636 | +0.000013 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000692 | +0.000045 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000734 | +0.000088 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001242 | +0.000067 |
| baseline_risk_band | over04 | 6080 | 3493 | -0.001918 | +0.000121 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.