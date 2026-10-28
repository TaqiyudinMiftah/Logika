# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 7 | 99 | 0.3939393939393939 | 0.32323232323232326 |
| 0.200000 | 13 | 149 | 0.2550335570469799 | 0.16778523489932887 |
| 0.323686 | -2 | 198 | 0.09595959595959595 | 0.10606060606060606 |
| 0.400000 | -1 | 232 | 0.09051724137931035 | 0.09482758620689655 |
| 0.500000 | -3 | 243 | 0.06172839506172839 | 0.07407407407407407 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.003499 | -0.000281 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | +0.001576 | -0.000160 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | +0.001378 | -0.000157 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.001243 | -0.001214 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000821 | -0.000157 |
| employment_type | salaried | 33473 | 4432 | +0.000818 | -0.000138 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000785 | -0.000207 |
| employment_type | variable_income | 15540 | 2180 | +0.000729 | +0.000011 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000649 | +0.000178 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.000644 | -0.001096 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000616 | -0.000195 |
| employment_type | self_employed | 20987 | 2895 | +0.000140 | -0.000132 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000056 | -0.000165 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000108 | +0.000149 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000997 | -0.000005 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001155 | -0.000184 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.001835 | -0.000373 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001905 | -0.000109 |
| aux_allocation_count | zero | 23732 | 2895 | -0.002012 | +0.000120 |
| aux_account_months | zero | 21000 | 2461 | -0.002032 | +0.000145 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.002092 | +0.000044 |
| orders90_exposure_band | zero | 3428 | 106 | -0.005106 | +0.000159 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.