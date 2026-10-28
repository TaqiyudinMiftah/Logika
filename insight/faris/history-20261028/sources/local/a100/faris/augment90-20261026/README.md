# Baseline90 plus actuarial additions — 26 October 2026

This is the user-approved follow-up to the negative horizontal feature-pack experiment. Every new competition-model input keeps **all 90 original baseline columns in their original order**, then appends at most ten actuarial features. The original pack filter had reduced the previous nominal 90-column control to 89; this campaign trains fresh 90-column controls instead of reusing those controls as a matched comparison.

## Frozen design

The existing 19,106 source-backed formulas are screened within each original fold using only one 3,000-row half of the 6,000 correction rows. The score measures the association between a candidate's ranks or missingness and the original Kumo model's residual. All context statistics and exact duplicate checks use the 50,000 context rows. Highly redundant candidates are excluded; each of 11 actuarial families contributes up to ten additions. A twelfth mixed group is diagnostic. This is a heuristic conditional-screening extension, not an implementation of CSIS theoretical guarantees.

The 11 families cover amortization, cash-flow stress, debt service, debt-to-income, liquidity relative to expenses, income and payments, net debt, repayment credibility, temporal changes, and volatility. The same research and formula catalog from the preceding experiment is reused. No new external training records or labels are introduced.

All groups receive Medium and Large Kumo runs, each with eight estimators and the same declared seed within each outer fold. Each also has a fresh full90 control of matching size. There are 130 planned configurations. Two fold-family groups have no qualifying addition; their four size-specific configurations reuse their identical controls, leaving **126 distinct main fits**, plus one old89 replay pilot.

The primary prediction adds a small, paired probability increment to the original raw baseline. Each increment is the augmented-model probability minus its new full90 control. The primary direction averages both model sizes and all 11 actuarial families. The mixed group is excluded from this average. Only a weight from 0, 0.1, 0.2 or 0.4 is selected on the other 3,000 correction rows; a positive weight must improve KS on the complete choice half and on at least four of five predeclared stratified 80% subsamples. Weight zero reproduces the original baseline exactly. The original maturity correction and spline remain the downstream evaluation procedure.

The primary recipe is saved before loading outer evaluation labels. Family-specific results are diagnostics and do not select the primary candidate. The primary comparison is the original matched maturity pooled KS **0.5306361042400267**; the target remains **0.54**.

## Evidence gates and limits

LOGIC_GATE.json checks known KS examples, exact weight-zero recovery, preservation of all90 values and order even with constant/duplicate baseline columns, and weight-selection invariance to perturbing outer predictions. FEATURE_GATE.json records the five fold-local selections and their fingerprints. GATE.json must then record a successful replay of the earlier Medium/eight-estimator 89-column control before broad admissions.

All original outer labels have prior development exposure. The reported fivefold pooled score is reused development evidence. Historical timestamp/DPD consistency checks passed, but upstream repayment snapshot capture semantics remain an assumption. Residual screening can miss interaction-only effects; redundancy thresholds and support minima are predeclared design choices. A first-seed favorable result needs another paired training seed before promotion. This runner does not refit or submit a candidate automatically.

## Compute and retention

- Node2001: /mnt/home/faris/logika-augment90-20261026; CPU feature selection, finite coordinator, scoring and durable results. Coordinator unit: logika-augment90-coordinator-20261026.
- Node2002: /home/faris/logika-augment90-20261026; uses the already verified numerical runtime and checkpoint cache under the preceding horizontal experiment.
- Node2003: /mnt/home/faris/logika-augment90-20261026; uses the existing verified runtime and cache.
- Faris A100 is excluded. The dispatcher uses only idle GPUs and adequate available RAM, with at most four simultaneous fits. Busy GPUs belonging to other work are left occupied.

The bounded run has a six-hour ceiling, including a two-hour evaluation reserve. The original absolute deadlines for this run are **13:29 WIB for new admissions** and **15:29 WIB for completion**, both on 26 October 2026. Resume does not reset them. Worker fits have their own 20-minute caps and 22 GiB memory limits.

The program retains recipes, selected feature definitions, source/checkpoint hashes, parameters, seeds, compact predictions and metrics. It saves no fitted weights or KV caches. Existing immutable input tables and pretrained checkpoints are reused rather than copied per fit.

Read LIVE_STATE.json for the current stage. Finished raw predictions are under runs/, and a completed evaluation is under evaluation/RESULT.json with its frozen recipe and OOF array. Old89 replay results are pilot evidence rather than main-fit results. The local directory mirrors evidence and may lag the live archive on node2001.

## Sources

The formula-specific references and proposed extensions are in SOURCE_LEDGER.json. Additional screening references are:

- Barut, Fan and Verhasselt, *Conditional Sure Independence Screening*, arXiv v2 (2012): https://arxiv.org/abs/1206.1024 . Supports evaluating candidate contributions in the presence of known useful predictors; our residual-rank approximation is a proposed extension.
- SciPy, *rankdata* documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html . Average ranks handle tied feature values.
