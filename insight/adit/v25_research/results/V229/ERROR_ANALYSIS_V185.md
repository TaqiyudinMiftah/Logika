# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 12 | 246 | 0.3861788617886179 | 0.33739837398373984 |
| 0.200000 | -3 | 358 | 0.22346368715083798 | 0.23184357541899442 |
| 0.323686 | 14 | 424 | 0.1391509433962264 | 0.10613207547169812 |
| 0.400000 | 2 | 447 | 0.087248322147651 | 0.08277404921700224 |
| 0.500000 | 4 | 434 | 0.07603686635944701 | 0.06682027649769585 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| aux_allocation_count | zero | 23732 | 2895 | +0.003171 | -0.000035 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.003013 | -0.000001 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.001916 | -0.000103 |
| aux_account_months | zero | 21000 | 2461 | +0.001371 | -0.000198 |
| baseline_risk_band | over04 | 6079 | 3488 | +0.000417 | +0.000815 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | +0.000349 | +0.000419 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.000218 | +0.000060 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000070 | -0.000020 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000356 | -0.000008 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.000425 | -0.000057 |
| employment_type | self_employed | 20987 | 2895 | -0.000430 | +0.000071 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000436 | +0.000025 |
| employment_type | variable_income | 15540 | 2180 | -0.000721 | -0.000296 |
| baseline_risk_band | under005 | 29965 | 546 | -0.000876 | -0.000054 |
| employment_type | salaried | 33473 | 4432 | -0.002135 | +0.000010 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.002183 | +0.000078 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.003720 | -0.001037 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.005934 | -0.000348 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.