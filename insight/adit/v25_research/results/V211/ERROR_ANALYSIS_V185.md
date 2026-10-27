# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 7 | 407 | 0.3931203931203931 | 0.3759213759213759 |
| 0.200000 | 24 | 628 | 0.24044585987261147 | 0.20222929936305734 |
| 0.323686 | -5 | 780 | 0.1294871794871795 | 0.1358974358974359 |
| 0.400000 | -5 | 866 | 0.09006928406466513 | 0.09584295612009237 |
| 0.500000 | -19 | 907 | 0.05843439911797133 | 0.07938257993384785 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | +0.003823 | -0.000723 |
| baseline_risk_band | under005 | 29965 | 546 | +0.002989 | -0.000485 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000341 | -0.000513 |
| aux_allocation_count | zero | 23732 | 2895 | -0.001086 | -0.000309 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.001872 | -0.000377 |
| aux_account_months | zero | 21000 | 2461 | -0.002409 | -0.000415 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.002476 | -0.000793 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.002696 | -0.000721 |
| employment_type | salaried | 33473 | 4432 | -0.002819 | -0.000675 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.002921 | -0.000847 |
| baseline_risk_band | over04 | 6079 | 3488 | -0.003078 | -0.000299 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.003507 | -0.000886 |
| employment_type | variable_income | 15540 | 2180 | -0.003616 | -0.000940 |
| employment_type | self_employed | 20987 | 2895 | -0.003883 | -0.000665 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.008934 | -0.000939 |
| baseline_risk_band | 015_to04 | 14489 | 3637 | -0.009487 | -0.002034 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.010085 | -0.004609 |
| baseline_risk_band | 005_to015 | 19467 | 1836 | -0.011007 | -0.007954 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.