# Horizontal research results

Completed two training seeds on the same five outer folds. Historical labels and feature choices were already reused; no untouched validation remains.

Inherited Kumo and v3 DPD/paid snapshot availability remains uncertified. This is a conditional benchmark, not independent validation or automatic promotion.

Primary comparison: predeclared equal-probability combination of the two independently successful pipelines. Learned heads use only the disjoint 6,000-row correction set, upstream models only the 50,000-row context.

| Recipe | Seed | KS | AUC | KS delta vs friend | AUC delta vs friend | KS delta vs champion | AUC delta vs champion |
|---|---:|---:|---:|---:|---:|---:|---:|
|friend|2026|0.529299|0.847203|+0.000000|+0.000000|+0.007973|+0.004427|
|champion|2026|0.521327|0.842775|-0.007973|-0.004427|+0.000000|+0.000000|
|context_champion|2026|0.521242|0.842576|-0.008057|-0.004627|-0.000085|-0.000199|
|horizontal_probability|2026|0.526415|0.846544|-0.002884|-0.000659|+0.005088|+0.003768|
|horizontal_logit|2026|0.526109|0.846499|-0.003190|-0.000703|+0.004782|+0.003724|
|context_equal|2026|0.525409|0.846425|-0.003890|-0.000778|+0.004082|+0.003650|
|convex_specialists|2026|0.525952|0.846741|-0.003347|-0.000461|+0.004626|+0.003966|
|convex_enriched|2026|0.525928|0.846744|-0.003371|-0.000459|+0.004602|+0.003969|
|account_spline|2026|0.281736|0.695361|-0.247563|-0.151842|-0.239591|-0.147414|
|account_dynamic|2026|0.280852|0.695390|-0.248448|-0.151812|-0.240475|-0.147385|
|bureau_logit|2026|0.279685|0.692637|-0.249615|-0.154566|-0.241642|-0.150138|
|bureau_dynamic|2026|0.281878|0.694061|-0.247421|-0.153142|-0.239449|-0.148714|
|coverage_gate|2026|0.525399|0.846446|-0.003900|-0.000757|+0.004073|+0.003671|
|offset_calibration|2026|0.525462|0.846432|-0.003837|-0.000771|+0.004135|+0.003656|
|offset_pricing|2026|0.525685|0.846464|-0.003615|-0.000739|+0.004358|+0.003688|
|friend|2027|0.529299|0.847203|+0.000000|+0.000000|+0.008163|+0.004296|
|champion|2027|0.521136|0.842907|-0.008163|-0.004296|+0.000000|+0.000000|
|context_champion|2027|0.520735|0.842554|-0.008564|-0.004649|-0.000401|-0.000353|
|horizontal_probability|2027|0.526479|0.846600|-0.002821|-0.000602|+0.005343|+0.003694|
|horizontal_logit|2027|0.526336|0.846557|-0.002963|-0.000645|+0.005200|+0.003651|
|context_equal|2027|0.525167|0.846400|-0.004133|-0.000803|+0.004031|+0.003493|
|convex_specialists|2027|0.525819|0.846730|-0.003480|-0.000473|+0.004683|+0.003823|
|convex_enriched|2027|0.525619|0.846732|-0.003680|-0.000471|+0.004483|+0.003825|
|account_spline|2027|0.281736|0.695361|-0.247563|-0.151842|-0.239400|-0.147546|
|account_dynamic|2027|0.280852|0.695390|-0.248448|-0.151812|-0.240284|-0.147516|
|bureau_logit|2027|0.279685|0.692637|-0.249615|-0.154566|-0.241452|-0.150270|
|bureau_dynamic|2027|0.281878|0.694061|-0.247421|-0.153142|-0.239258|-0.148846|
|coverage_gate|2027|0.525301|0.846423|-0.003999|-0.000780|+0.004164|+0.003516|
|offset_calibration|2027|0.525406|0.846408|-0.003894|-0.000795|+0.004269|+0.003501|
|offset_pricing|2027|0.525257|0.846435|-0.004043|-0.000767|+0.004120|+0.003529|

Both-reference conditional stability criterion passed: none.
No positive conclusion follows from a higher point estimate alone. Confidence intervals do not undo historical adaptive feature selection.

Mechanism contrasts: horizontal_probability vs both anchors; context_equal vs the full champions measures fitting-partition cost; convex_specialists and coverage_gate vs context_equal test fusion; offset_pricing vs offset_calibration separates pricing features from calibration.

No competition test inference, submission refit, upload, or changes to the existing submission. Every fitted model and transformer remains in fits/ and heads/.
A KS-only gain with declining AUC is rejected. Conditional support requires nonnegative pooled AUC improvement in both seeds and positive mean improvement, at least 3/5 AUC fold wins per seed, adjusted AUC lower bound >-0.0005, and log-loss increase <=0.0005, in addition to the strict KS criterion.
The 2,000-resample intervals at the adjusted extreme quantiles have limited Monte Carlo resolution. They describe these fitted predictions only; they do not establish leaderboard generalization.
See scoreboard.json, uncertainty.json, diversity.json, cohort_diagnostics.json and source manifests for details.
