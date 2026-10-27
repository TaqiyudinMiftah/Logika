# Held-out error analysis

held-out OOF only; descriptive and post-selection, no in-sample predictions

| Top population fraction | Net additional defaults captured | Entrants | Entrant bad rate | Exit bad rate |
|---|---:|---:|---:|---:|
| 0.100000 | -4 | 177 | 0.3785310734463277 | 0.4011299435028249 |
| 0.200000 | -11 | 264 | 0.1893939393939394 | 0.23106060606060605 |
| 0.323686 | -2 | 304 | 0.10855263157894737 | 0.11513157894736842 |
| 0.400000 | 2 | 349 | 0.09169054441260745 | 0.08595988538681948 |
| 0.500000 | 0 | 325 | 0.07076923076923076 | 0.07076923076923076 |

## Segment changes

| Segment | Value | N | Events | Delta KS | Delta AUC |
|---|---|---:|---:|---:|---:|
| baseline_risk_band | 005_to015 | 19504 | 1855 | +0.003532 | -0.000514 |
| employment_type | salaried | 33473 | 4432 | +0.001619 | +0.000078 |
| baseline_risk_band | 015_to04 | 14563 | 3624 | +0.001307 | +0.000481 |
| baseline_risk_band | over04 | 6080 | 3493 | +0.000981 | -0.000095 |
| employment_type | variable_income | 15540 | 2180 | +0.000499 | -0.000198 |
| baseline_risk_band | under005 | 29853 | 535 | +0.000341 | -0.001045 |
| risk_conditional_nonnegative_months | over3 | 36801 | 5132 | +0.000153 | -0.000054 |
| aux_allocation_count | over3 | 44414 | 6265 | -0.000690 | -0.000056 |
| risk_conditional_negative_months | over3 | 16936 | 2238 | -0.000708 | -0.000071 |
| aux_account_months | 1_to_3 | 49000 | 7046 | -0.000830 | -0.000060 |
| aux_allocation_count | 1_to_3 | 1854 | 347 | -0.000872 | +0.000099 |
| employment_type | self_employed | 20987 | 2895 | -0.000882 | -0.000090 |
| risk_conditional_negative_months | 1_to_3 | 17621 | 2792 | -0.001115 | +0.000056 |
| risk_conditional_nonnegative_months | zero | 25586 | 3175 | -0.001223 | +0.000039 |
| aux_allocation_count | zero | 23732 | 2895 | -0.001728 | +0.000012 |
| risk_conditional_nonnegative_months | 1_to_3 | 7613 | 1200 | -0.001836 | -0.000038 |
| aux_account_months | zero | 21000 | 2461 | -0.001878 | +0.000049 |
| risk_conditional_negative_months | zero | 35443 | 4477 | -0.001930 | -0.000063 |

Segment effects are associations and exploratory; do not fit gates on outer labels.

ERROR_CASES.csv preserves four types of corrected and worsened mistakes. Sparse segment fluctuations are not stable discoveries.