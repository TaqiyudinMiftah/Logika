# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 1 | 277 | 0.37184115523465705 | 0.36823104693140796 |
| 0.200000 | 23 | 451 | 0.24390243902439024 | 0.19290465631929046 |
| 0.323686 | 5 | 531 | 0.1224105461393597 | 0.11299435028248588 |
| 0.400000 | 0 | 563 | 0.08880994671403197 | 0.08880994671403197 |
| 0.500000 | -10 | 535 | 0.06542056074766354 | 0.08411214953271028 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29965 | 546 | +0.008293 | +0.002198 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002684 | -0.000025 |
| employment_type | variable_income | 15540 | 2180 | +0.002672 | -0.000382 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.002628 | +0.001054 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002480 | -0.000015 |
| aux_account_months | zero | 21000 | 2461 | +0.001651 | -0.000166 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001169 | +0.000082 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.000038 | +0.001706 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000097 | +0.000032 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.000425 | -0.001011 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000653 | +0.000187 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001034 | +0.000168 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001134 | +0.000072 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001258 | +0.000153 |
| employment_type | self_employed | 20987 | 2895 | -0.001688 | +0.000175 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.001706 | -0.000897 |
| employment_type | salaried | 33473 | 4432 | -0.002290 | +0.000213 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.005823 | +0.000312 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.