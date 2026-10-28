# Frozen shadow-loan modelling campaign

The user approved implementation of the four ideas in `USER_PROPOSAL.txt`, followed by an immutable feature set and a 24-hour modelling ceiling. **The campaign is complete: all 340 fits were audited, but no gain or KS 0.54 was established.** The preselected ensemble scored 0.5303323155 versus the matched baseline 0.5306361042. No refit or submission was made. See [RESULTS.md](RESULTS.md), [ANALYSIS.json](ANALYSIS.json) and [TERMINAL_AUDIT.json](TERMINAL_AUDIT.json). Execution records below preserve the historical sequence; the live service has exited successfully.

## Evidence driving the design

Authenticated Kaggle history confirms V25 is the best observed public submission, KS **0.56822**. The maturity-only reference scored **0.56818**. Two-seed periodicity improved local KS to **0.531403805** but scored **0.56656** publicly. A local improvement alone does not establish a public improvement.

The broad horizontal feature campaign scored **0.528137239**, versus its matched baseline **0.530636104**. The exact-90-plus-actuarial follow-up scored **0.530530919**, also below that baseline. Neither was submitted. `evidence/SUBMISSION_REVIEW.json` records these comparisons and the authenticated submission identities.

## What is frozen

The master table contains **116 columns**. Predeclared views are `champion90`, `shadow91`, `latent99`, `full100`, and `master`. The original 90-column model input, including its duplicate column, is preserved exactly. Kumo views contain at most 100 columns. The original 18-column correction head, including six coverage measurements, is preserved.

Two learned shadow-loan scores use historical contracts, conservative 12-month labels and interval/right-censored repayment evidence. Of 159,323 historical approved loans, 85,972 have conservative usable 12-month labels; unknown outcomes remain unknown. Training excludes the borrower whose query representation is being constructed. Historical event times are bounded rather than invented. Upstream lifetime snapshot completeness remains an explicit assumption.

The joint representation contributes eight latent coordinates and posterior variance. Its encoders use context customers only and combine four financial views with reconstruction, historical-risk and current-risk objectives. The coordinates have no asserted psychological interpretation.

All five fold tables are sealed by `FEATURE_FREEZE.json`, SHA-256 `071c75e0f59cf92429bde9c76a64e27931e1eea03bbbbeac2e2589b5fee68dd6`. Do not regenerate values, add columns or select new features during this campaign. Context-only model preprocessing and the already declared ablation views are permitted. Sources and distinctions between published methods and our adaptations are in `SOURCES.json`.

## Validation and modelling

The existing five-fold protocol uses 50,000 context, 6,000 correction and 14,000 validation customers per fold. Feature gates passed in every fold: exact original-90 preservation, finite learned values, borrower separation, known metric/likelihood cases, and baseline reconstruction within `1.12e-16`.

`CONFIGS.json` fixes 68 configurations and 340 five-fold fits: global Kumo, clustered local-context Kumo, LightGBM, CatBoost, XGBoost, ExtraTrees, logistic regression, MLP, and a BCE/pairwise/consistency ranking correction. Global Kumo and ranking corrections include second-seed experiments. Other families are not all second-seed confirmed. No claim is made that this completes the older exhaustive foundation-model or actuarial-model lists.

Models rank on one correction half and mixture choices use the other. Early stopping and ranking-student training also use the ranking half, so its rankings are training-informed development evidence. The final fold recipes are saved before terminal outer scoring. The original correction head and maturity spline recipe remain fixed. Only configurations completing all five folds enter selection. Failed and unfinished fits remain visible.

The outer data have been used in earlier research. Paired stratified bootstrap intervals are conditional development evidence, not an untouched audit. Target KS 0.54, local improvement and public improvement are separate outcomes. Promotion additionally requires stable evidence across folds and a second seed where applicable; the evaluator never marks a candidate automatically eligible for submission.

## Execution and storage

Canonical live root: `/mnt/home/faris/logika-frozen-shadow-20261026` on `faris@100.68.29.126` (node 2001). Active hosts are now **2001 and 2003 only**, following the user's confirmation that 2002 is down. Node 2003 uses `/mnt/home/faris/logika-frozen-shadow-20261026`. The former node-2002 root `/home/faris/logika-frozen-shadow-20261026` is retired. A100 is excluded. Use only each user's actual home directory.

The immutable clock started **26 October 2026, 17:36:20 WIB**. New search admission ends **27 October, 11:36:20 WIB**; the total ceiling ends **27 October, 17:36:20 WIB**. Full queue dispatch began **26 October, 19:14:17 WIB**, after the two pilots passed. Pilot/launch preparation consumed part of the existing ceiling; it was not reset.

Service: `logika-frozen-shadow-coordinator-20261026.service`. `LIVE_STATE.json` is an atomic runtime snapshot. `MODEL_RUN_AUDIT.json` and `evaluation/RESULT.json` are terminal outputs. `source/coordinator.py` admits only idle GPUs, accounts for other GPU workloads when allocating CPUs, and enforces bounded per-fit CPU/RAM/runtime limits. Per-fit isolated services continue independently of the local chat.

Each fit retains its parameters, seed, source/input hashes, timing, status and compressed predictions. Fitted model weights and KV caches are not saved. Existing original foundation checkpoints are shared for execution. Do not stop other users' workloads or widen the deadline. If interruption occurs, reconcile active remote services and collect finished outputs before resuming, to avoid duplicate execution.

The local `LIVE_STATE.json` and run receipts are copied snapshots; query the node-2001 file for current state. Frozen feature tables remain on all three nodes. No automatic refit or Kaggle submission is built into the coordinator.

## Execution recovery on 26 October

Node 2002 became unreachable through SSH and was offline on Tailscale. At 19:34:59 WIB, a failed result transfer stopped the original coordinator. The follow-up diagnosed the outage and resumed the coordinator at 19:56:27 WIB, with **180 verified results** preserved. Five earlier node-2002 dispatches remain unresolved: three ExtraTrees folds and two Kumo folds. They are held for reconciliation, not silently counted as complete or retried on another host.

The repair adds offline-node backoff, result collection retries and adoption of earlier dispatches. An inactive unit with no result or failure is retried only after its partial artifacts are archived. Existing MLP and ranking workers also run on CPUs when GPU capacity is unavailable; their original objectives, parameters, seeds and feature values are unchanged. CPU allocation now reserves the full per-fit RAM allowance when admitting a batch. The immutable clock, feature freeze, configuration list and scientific selection rules are unchanged; `EXECUTION_REPAIR.json` records their hashes and the execution-source change.

The first repair retried node 2002 automatically. This was superseded by the user's instruction to use 2001 or 2003. Temporary transport credentials were removed when the old coordinator stopped; they must never be copied into experiment evidence. The five unverified attempts on node 2002 were explicitly replaced, with local partial artifacts and logs preserved under `retired_node2002/`. Their inaccessible original outputs must not be merged into replacements. See `NODE2002_RETIREMENT.json` and `NODE2002_MIGRATION_EXECUTION.json`.

## Current execution policy after node-2002 retirement

The three missing ExtraTrees folds completed on node 2001. **245 of 340 planned fits are verified** at this recovery point. The remaining 95 Kumo fits require idle GPU capacity on 2001 or 2003. Other users' GPU work is left running. No outer evaluation or new submission has occurred.

A 1,024-context, 128-query, two-estimator CPU Kumo smoke test passed in 49.7 seconds. This did not establish full-size feasibility. The full 50,000-context, eight-estimator CPU attempt exceeded its 22 GiB cgroup memory limit. The recovery handler incorrectly treated missing-result memory kills as interruptions, causing 19 recorded node-2001 memory-kill events before the route was disabled. These were execution failures, not validation evidence against Kumo; no complete prediction or metric resulted. Journals and partial attempt evidence are retained under `evidence/cpu_resource_limit/` and `interrupted/`, with the incident recorded in `CPU_RESOURCE_LIMIT.json`.

CPU Kumo admission is now **disabled** in `EXECUTION_POLICY.json`; its unchanged configuration has returned to the GPU queue. The retry fix classifies missing-result failures from attempt journals, records dispatch device/start time, disables CPU Kumo after a resource failure, and permits at most two retries for an unknown interruption. A separate `PAUSE_ADMISSION` file can pause admission without triggering terminal evaluation. `RETRY_HANDLER_FIX.json` records this execution change. Feature tables, model configurations, scientific selection rules and the original clock remain unchanged.

CPU and GPU execution use different arithmetic and random-generator backends. The CPU code records float32/CPU RNG and GPU code retains float16 autocast/CUDA RNG; they are not claimed to be bitwise reproductions. All successful Kumo results to this point are GPU results. Existing successful MLP/ranking CPU results remain part of the planned campaign.
