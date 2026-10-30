# Final three requested experiments —30October2026

The explicitly requested ecommerce CSV was submitted as **Kaggle #57812068**, at **16:59:21 UTC**, before the 17:00 UTC cutoff. Kaggle confirms **COMPLETE**, public KS **0.56531**; private score is unavailable. CSV SHA256: `da59d9fc65e61c063fbd1c6946dc13e52e832e86c179fb059509939e1bb99490`.

| Requested experiment | Pooled validation KS | AUC | Outcome |
|---|---:|---:|---|
| Ecommerce,seed 2026 |0.52964502597|0.84653901879|Complete validation and full-data production; explicitly submitted|
| Literal seed 3 |0.52885782283|0.84654125664|Complete validation; no production refit|
| Literal seed 4 |—|—|Incomplete Kumo fold 3; no valid pooled score|

The highest verified prior validation KS remains **0.53196797999**. Ecommerce's matched original seed 2026 comparator was **0.53066575785**, a feature delta of **−0.00102073188**. The new feature block therefore did not improve this matched point estimate. All 70k cases remain adaptively reused development; no independent gain or 0.54 target attainment is claimed.

Ecommerce appends eight deterministic label-free interactions to the LocalGLM 99-field view: spending and order acceleration, budget pressure, cancellation activity, essential/discretionary mix, COD reliance, spending burstiness and recency/frequency. Source is the retained named transaction aggregates; rows were matched by application ID. Corresponding original seed 2026 CANN and Kumo validation/production predictions remain fixed. The LocalGLM branch was freshly trained for all five folds and full 70,000-row production.

The two literal-seed experiments retain the same three model branches and validation roles, with per-fold initialization base_seed+fold. Validation was prioritized under the last-submission window. A queue-module naming collision and a missing canonical checkpoint caused pretraining failures. Seed4's remaining Kumo fold was not admitted by an earlier in-memory cutoff, and its late recovery did not complete. Those are operational incomplete results, not predictive negatives; no missing fold was filled with another seed or fallback.

All experiment workers and active campaign units were verified stopped on all four hosts. The actual stop was later than the preferred 3–4 minute lead, and the authorized ecommerce submission still reached Kaggle before the hard cutoff. No further experiment, upload retry, or automatic restart is queued. Existing completed campaign clocks and sources remain archived. Sources, parameters, logs, predictions and failed/partial evidence remain private with aggregate records and hashes published. Newly fitted weights/KV caches were not serialized, as requested.
