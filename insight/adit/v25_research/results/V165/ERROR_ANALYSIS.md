# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 1 | 47 | 0.425531914893617 | 0.40425531914893614 |
| 0.200000 | 6 | 61 | 0.3114754098360656 | 0.21311475409836064 |
| 0.323686 | -4 | 62 | 0.08064516129032258 | 0.14516129032258066 |
| 0.400000 | -1 | 84 | 0.10714285714285714 | 0.11904761904761904 |
| 0.500000 | -3 | 83 | 0.024096385542168676 | 0.060240963855421686 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.002705 | +0.000137 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.002218 | -0.000293 |
| aux_allocation_count | zero | 23732 | 2895 | +0.001228 | +0.000059 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.000627 | +0.000046 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.000625 | +0.000628 |
| employment_type | salaried | 33473 | 4432 | +0.000554 | +0.000069 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.000221 | +0.000027 |
| aux_account_months | zero | 21000 | 2461 | +0.000191 | -0.000004 |
| employment_type | variable_income | 15540 | 2180 | +0.000064 | +0.000011 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000030 | +0.000010 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000026 | +0.000033 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000110 | +0.000053 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000111 | +0.000066 |
| employment_type | self_employed | 20987 | 2895 | -0.000316 | +0.000025 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.000665 | +0.000012 |
| baseline_risk_band | under005 | 29853 | 535 | -0.000709 | +0.000268 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000737 | +0.000134 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | -0.001517 | -0.000005 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.