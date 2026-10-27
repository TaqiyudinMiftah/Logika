# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 6 | 270 | 0.37407407407407406 | 0.35185185185185186 |
| 0.200000 | 0 | 417 | 0.2182254196642686 | 0.2182254196642686 |
| 0.323686 | -4 | 506 | 0.12055335968379446 | 0.12845849802371542 |
| 0.400000 | -1 | 544 | 0.08088235294117647 | 0.08272058823529412 |
| 0.500000 | -3 | 517 | 0.06576402321083172 | 0.07156673114119923 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | under005 | 29965 | 546 | +0.006546 | -0.000298 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | +0.002968 | -0.000042 |
| aux_allocation_count | zero | 23732 | 2895 | +0.002836 | -0.000123 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.002147 | +0.000080 |
| aux_account_months | zero | 21000 | 2461 | +0.001335 | -0.000307 |
| risk_conditional_negative_months | zero | 35443 | 4477 | +0.001031 | -0.000090 |
| employment_type | variable_income | 15540 | 2180 | +0.000113 | -0.000526 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | +0.000004 | +0.000003 |
| employment_type | self_employed | 20987 | 2895 | -0.000689 | +0.000056 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.000997 | +0.000673 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001452 | +0.000193 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.001804 | +0.001621 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001968 | +0.000042 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001989 | +0.000054 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.002170 | -0.001669 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.003198 | -0.000062 |
| employment_type | salaried | 33473 | 4432 | -0.003290 | +0.000113 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.005232 | -0.000799 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.