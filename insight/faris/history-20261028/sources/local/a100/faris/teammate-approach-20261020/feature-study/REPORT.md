# Feature versus regularization attribution

The 11 fixed recipes completed five outer folds. No post-hoc winner was selected.

| Contrast | ΔKS | ΔAUC | Δlog loss | KS fold wins |
|---|---:|---:|---:|---:|
| V07_regularization | -0.00072577 | +0.00007738 | -0.00004668 | 3/5 |
| V07_coverage_published | +0.00122716 | -0.00005588 | +0.00002263 | 4/5 |
| V07_features_published | -0.00017131 | +0.00000398 | +0.00002242 | 1/5 |
| V07_coverage_original | +0.00044208 | -0.00007943 | +0.00004037 | 4/5 |
| V07_features_original | -0.00030696 | +0.00001876 | +0.00002325 | 0/5 |
| V07_total_published | +0.00033008 | +0.00002548 | -0.00000163 | 2/5 |
| V101_regularization | -0.00026697 | +0.00018295 | -0.00012030 | 2/5 |
| V101_coverage_published | -0.00021819 | +0.00000986 | -0.00000977 | 3/5 |
| V101_features_published | +0.00083943 | +0.00002063 | -0.00005605 | 2/5 |
| V101_coverage_original | +0.00035342 | -0.00008148 | +0.00004154 | 4/5 |
| V101_features_original | -0.00228857 | -0.00056744 | +0.00035240 | 0/5 |
| V101_total_published | +0.00035427 | +0.00021345 | -0.00018612 | 0/5 |

Read `features_published` to assess the feature block beyond matched regularization and coverage.
Read `regularization` and `coverage_published` to explain the other parts of its total change.
The original-setting feature contrast tests whether this attribution depends on regularization.

All 55 new heads are retained and replayed against the saved Kumo scores.
Published V07/V101 predictions were reconstructed before the experiment.
These families were chosen using historical results; all intervals are descriptive.
No independent fresh data, public-score guarantee, or new competition submission is claimed.
