# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | 4 | 71 | 0.38028169014084506 | 0.323943661971831 |
| 0.200000 | 4 | 119 | 0.2689075630252101 | 0.23529411764705882 |
| 0.323686 | -11 | 142 | 0.07746478873239436 | 0.15492957746478872 |
| 0.400000 | -4 | 165 | 0.10909090909090909 | 0.13333333333333333 |
| 0.500000 | -5 | 148 | 0.02027027027027027 | 0.05405405405405406 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | over04 | 6080 | 3493 | +0.004280 | +0.000120 |
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.002156 | -0.000770 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | +0.001555 | -0.000438 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | +0.000463 | -0.000050 |
| employment_type | variable_income | 15540 | 2180 | +0.000332 | -0.000053 |
| orders90_exposure_band | zero | 3428 | 106 | +0.000301 | +0.000045 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.000286 | -0.000101 |
| aux_allocation_count | zero | 23732 | 2895 | -0.000441 | -0.000073 |
| orders90_exposure_band | 4_to10 | 36368 | 5141 | -0.000651 | -0.000146 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | -0.000743 | -0.000622 |
| orders90_exposure_band | over10 | 11489 | 3097 | -0.000780 | -0.000099 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001023 | -0.000090 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001122 | -0.000154 |
| employment_type | salaried | 33473 | 4432 | -0.001164 | -0.000076 |
| aux_account_months | zero | 21000 | 2461 | -0.001273 | -0.000071 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | -0.001314 | -0.000109 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.001321 | -0.000119 |
| employment_type | self_employed | 20987 | 2895 | -0.001437 | -0.000200 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.001656 | -0.000109 |
| orders90_exposure_band | 1_to3 | 18715 | 1163 | -0.001673 | -0.000007 |
| baseline_risk_band | under005 | 29853 | 535 | -0.001821 | -0.000658 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.003120 | -0.000101 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.