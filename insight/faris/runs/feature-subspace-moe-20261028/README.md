# Feature-subspace mixture of experts — 28 October 2026

The user authorized a fresh **12-hour ceiling**. Its immutable clock starts before
the first actual-data qualification fit, including feature selection and nested
anchor calibration. Admission closes after7.5hours, leaving evaluation and refit
reserve. Previous completed campaigns and their clocks are unchanged.

The hypothesis is that source-backed feature expansion helps specialized experts
whose usefulness varies with borrower history and data quality. The design crosses
existing versus expanded semantic views with fixed versus learned soft gating.
The four domains are cash flow, credit history, commerce, and cross-domain capacity.
Each expert receives up to99 predictors, with20 shared core features. Expanded
views use supervised feature selection on the50000 context rows only; all6000
calibration/gate and14000 outer-validation labels are excluded from that selection.
The feature universe includes audited historical definitions and explicit liquidity
stress/monthly-spend proxies. No arbitrary massive interaction sweep is introduced.

There are260 predeclared expert fits:80 Kumo Large/eight-estimator GPU fits and180
LightGBM/CatBoost CPU fits, across5 matching outer folds and2 seeds. CPU families
also test the wide union of the domain features. A predefined hybrid is secondary.
Feature selectors, calibration and gates are additional pipeline fits, recorded
separately from these expert-fit counts. Pilots may be reused as the identical
planned expert fits; they are not independent replications.

The gate is a small regularized softmax network. Its prior assigns90percent to the
matched anchor and2.5percent to each specialist. Fixed controls use those exact
constant weights. Domain predictions are calibrated with3fold held-out predictions
on the6000 correction rows. The anchor correction heads train on4000 rows; anchor
splines train on54000 rows, excluding every correction case whose prediction is
used to train the gate. This avoids reusing an in-sample56000-row spline prediction
as if it were held out. Final heads/calibrators fit6000; the gate sees only their
cross-fitted training predictions. All preprocessing and fitting stays inside each
outer-training partition. No candidate outer score is read before FROZEN.json.

The primary is the two-seed mean `kumo__expanded__gated`; secondary cohorts cannot
replace it after terminal outer scoring. Supporting a development gain requires
KS improvement>=0.0005, paired conditional95percent lower bound>0, positive deltas
in at least4of5folds, and positive deltas in both seeds. Local0.54 attainment is
reported separately. Routing and expansion receive their own matched contrasts.

The local reference is the audited maturity-adjusted fold-local anchor,
KS0.5306361042400267. Public V25 remains a separate exact artifact at0.56822.
All70000 examples have previous development exposure. New nested implementation
boundaries do not create an untouched audit, remove historical selection bias or
guarantee a public improvement. Source snapshot completeness remains inherited.

The original lab backends are2001 and2003, under /mnt/home/faris. After all180
CPU fits completed and unrelated users occupied every lab GPU, the separately
preregistered EXECUTION_SPILLOVER_PLAN.json admitted user-authorized Colab A100
account2 as an execution-only fallback, gated by label-free prediction replay.
All80 Kumo fits use that consistent replacement cohort; old lab attempts are
retired with hashes. Original source, config, contract and clock remain sealed.
GPU admission on shared lab nodes still requires idle capacity; CPU Kumo is forbidden. Unrelated workloads are never stopped. Source,
hyperparameters, seeds, preprocessing recipes, compact predictions, errors and
hashes are retained. No newly fitted weights or KV caches are serialized.

Canonical state: /mnt/home/faris/logika-feature-subspace-moe-20261028/LIVE_STATE.json
on faris@100.68.29.126. Service: logika-feature-subspace-moe-20261028.service.
Use source/launch.py --hours 12 exactly once. Reconciliation/resume uses the original
CLOCK.json; no extension or new launch is allowed. A supported primary creates a
REFIT_REQUIRED.json for frozen refit within the remaining original reserve; model
selection cannot change. A refit is incomplete until its provenance and predictions
are audited. No submission follows an unfinished run or a development point alone.

The public namespace is insight/faris/runs/feature-subspace-moe-20261028/. Every
planned configuration is logged before execution; aggregate fit records and
private prediction hashes are synchronized afterward, including failures.
