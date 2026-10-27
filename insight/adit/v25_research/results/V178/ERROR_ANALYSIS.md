# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 2 | 40 | 0.4 | 0.35 |
| 0.200000 | -4 | 74 | 0.20270270270270271 | 0.25675675675675674 |
| 0.323686 | -6 | 83 | 0.060240963855421686 | 0.13253012048192772 |
| 0.400000 | -2 | 109 | 0.08256880733944955 | 0.10091743119266056 |
| 0.500000 | 3 | 112 | 0.05357142857142857 | 0.026785714285714284 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6080 | 3493 | +0.001288 | +0.000746 |
| employment_type | variable_income | 15540 | 2180 | +0.000866 | +0.000087 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000498 | +0.000536 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.000429 | +0.000602 |
| employment_type | salaried | 33473 | 4432 | +0.000415 | +0.000144 |
| aux_account_months | zero | 21000 | 2461 | +0.000320 | +0.000224 |
| aux_allocation_count | over3 | 44414 | 6265 | +0.000249 | +0.000042 |
| aux_account_months | 1_to_3 | 49000 | 7046 | +0.000230 | +0.000048 |
| employment_type | self_employed | 20987 | 2895 | +0.000194 | +0.000049 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000177 | +0.000055 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000260 | +0.000219 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000498 | +0.000541 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000535 | +0.000054 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000619 | +0.000176 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.000651 | +0.000152 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.000702 | -0.000136 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.000828 | +0.000015 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001614 | +0.000026 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.