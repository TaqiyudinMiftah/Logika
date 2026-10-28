# Explicit support counts did not improve the fixed pipeline

Allfive original folds completed. Independent rawcount reconstruction and ten serializedmodel replays passed. Full70000OOF KS/AUC were independently checked with tiedempiricalCDF/ranksum statistics.

| Pipeline | KS | AUC | Log loss |
|---|---:|---:|---:|
| Repaired V25 baseline | .5302185367 | .8470258500 | .2926003248 |
| Two maturity-aligned means | .5306361042 | .8470560913 | .2925702718 |
| Maturity means plus two source-support counts | .5292136801 | .8469397530 | .2926518895 |

Do not promote the support-count recipe. Both main metrics decreased. The existing18/77input recipe was replayed before adding the two fields; only feature widths changed to20/79, with the original architectures, hyperparameters, training budgets, folds and75:25weights held fixed. Kumo predictions were unchanged. Account/source counts were row-local and label-free; no evaluation records fitted shared feature parameters.

Allfive newheads, five new79inputsplines, their preprocessing, alignedpredictionarrays and sourcefeatures remain retained under models/fold*/ and features.parquet. INDEPENDENT_AUDIT.json reports fullreconstruction parity, headobjectivegradient checks, originalsolverparameters and exactmodelreplay. result.json reports conditional simultaneousintervals. Reuseddevelopmentlabels and adaptivehypothesisselection preclude claiming untouchedgeneralization. Confirmation was not opened, and no submission was written. Target.54 remains unmet.

This does not prove that support has no relationship to risk. It rejects this exact append-to-head-and-spline recipe under the frozen current protocol. A new name or a different model/weight sweep is not justified by this result.
