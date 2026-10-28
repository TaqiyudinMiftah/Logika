# Logika 2026 — handoff for the next agent

Prepared **28 October 2026, Asia/Jakarta**. Workspace: `/home/parasite/Project/Competition/logika-2026`.

This document consolidates the verified experiment state and operating constraints. It supersedes stale operational instructions in the older handoff, while preserving that document as historical evidence. Creating this handoff did not start training, change feature tables, submit to Kaggle, delete artifacts or restart automations.

## 1. Current decision

**Retain the incumbent. No recent campaign established a supported improvement, and the 0.54 local pooled-KS target remains unmet.**

| Quantity | Result | Interpretation |
|---|---:|---|
| Best authenticated public submission in retained evidence | **V25: 0.56822** | Public incumbent; latest retained authenticated review is from 26 October, not a new Kaggle lookup on 28 October. |
| Matched local baseline for recent campaigns | **0.5306361042400267** | Maturity-only original Large90 pipeline; use this exact comparator for the recent frozen-feature results. |
| Selected two-seed periodicity local recipe A | **0.5314038049246252** | Higher local KS, but public **0.56656**, below V25. |
| Highest older local point, H016 | **0.531505932** | Not promoted: uncertainty, fold consistency and seed replication did not support its tiny gain. |
| Latest architecture campaign selected blend | **0.5291772103794308** | Below its matched baseline; all 160 planned fits completed and were audited. |

The public incumbent, local selected recipe, highest exploratory point and matched experimental comparator are different objects. Do not substitute one score for another. A higher local score has already failed to transfer to Kaggle here.

**There is no active campaign to resume.** The architecture and frozen-shadow campaigns are complete; their clocks are expired. Their follow-up automation is paused. The six-hour architecture allowance was a ceiling, not six hours of mandatory computation or reusable remaining budget.

## 2. Read these artifacts first

1. [Latest architecture results](/home/parasite/Project/Competition/logika-2026/a100/faris/architecture-20261027/RESULTS.md) and [analysis](/home/parasite/Project/Competition/logika-2026/a100/faris/architecture-20261027/ANALYSIS.json).
2. [Architecture terminal audit](/home/parasite/Project/Competition/logika-2026/a100/faris/architecture-20261027/TERMINAL_AUDIT.json), [frozen selection](/home/parasite/Project/Competition/logika-2026/a100/faris/architecture-20261027/evaluation/FROZEN.json), and [completion manifest](/home/parasite/Project/Competition/logika-2026/a100/faris/architecture-20261027/COMPLETION_MANIFEST.json).
3. [Parent frozen-shadow results](/home/parasite/Project/Competition/logika-2026/a100/faris/frozen-shadow-20261026/RESULTS.md), [feature freeze](/home/parasite/Project/Competition/logika-2026/a100/faris/frozen-shadow-20261026/FEATURE_FREEZE.json), and [resume state](/home/parasite/Project/Competition/logika-2026/a100/faris/frozen-shadow-20261026/RESUME_STATE.json).
4. [Authenticated submission review](/home/parasite/Project/Competition/logika-2026/a100/faris/frozen-shadow-20261026/evidence/SUBMISSION_REVIEW.json) and [submission/validation lineage](/home/parasite/Project/Competition/logika-2026/a100/faris/transfer-20261025/README.md).
5. [Earlier handoff](/home/parasite/Project/Competition/logika-2026/HANDOFF_LOGIKA_20261023.md), only for older experiment details and provenance. Its old paths, model-retention claims, quota observations and active-job instructions are not current authority.

## 3. User intent and non-negotiable boundaries

- The practical objective is better competition performance; the explicit research target is **0.54 local pooled OOF KS**. These are separate success claims.
- **Keep the existing features frozen.** The latest request authorized architectural/model changes, not new feature engineering. The earlier request to try Shadow Loan Learning was implemented before this freeze.
- Use **node2001 and node2003 only**, under the actual Faris home directory **`/mnt/home/faris`**. The user retired node2002 after it went down. Do not restore it, use A100, or use other machines merely because older instructions list them.
- Preserve **parameters, seeds, source, environment identities, preprocessing recipes, compact predictions and evidence**. Do not serialize fitted weights or KV caches. In-memory training states are permitted and released afterward.
- Do not stop unrelated workloads. Check actual idle capacity and resource limits before any newly authorized computation.
- The user previously authorized autonomous submissions, including a useful probe, but explicitly demanded convincing evidence. That is not a reason to submit a losing or unaudited candidate. Verify exact file identity, row alignment, provenance and current quota before any future upload.
- Preserve completed campaign seals, original clocks, failed attempts and exposure history. Do not overwrite outputs to rerun them.
- The user dislikes unnecessary permission questions. Make routine reversible decisions autonomously, but an expired campaign's duration is not a new training budget. A handoff request does not authorize another campaign.
- Prefer **RTK** for supported commands when reduced output is sufficient. Use a narrowly scoped filter only when recurring verbosity justifies it; preserve errors and exit status, and validate against captured output without repeating remote actions. Use ordinary commands when exact structured evidence is needed.

Credentials are intentionally omitted. Use configured SSH authentication and existing private credential handling; do not copy passwords or tokens from chat into this document, commands or logs.

## 4. Access and current machine state

### Node2001: archive, coordinator and jump host

```sh
ssh faris@100.68.29.126
```

Verified login home on 28 October: `/mnt/home/faris`.

| Purpose | Remote path |
|---|---|
| Latest campaign | `/mnt/home/faris/logika-architecture-20261027` |
| Parent frozen campaign and immutable tables | `/mnt/home/faris/logika-frozen-shadow-20261026` |
| Earlier experiment archive | `/mnt/home/faris/logika-faris-exploration` |
| Latest campaign Python | `/mnt/home/faris/logika-model-arena-20261025/env311/bin/python` |

### Node2003: through node2001

After entering node2001:

```sh
ssh -p 2003 faris@gpu-labkc.ub.ac.id
```

Verified login home on 28 October: `/mnt/home/faris`.

- Campaign directory: `/mnt/home/faris/logika-architecture-20261027`.
- Python used by this campaign: `/mnt/home/faris/faris/kumo-coverage-placement-20261021/.venv/bin/python`.
- Frozen input archive: `/mnt/home/faris/logika-frozen-shadow-20261026/frozen`.

The local directories containing `a100/` are archive names; they do not grant permission to execute on A100. GPU availability is transient and was not assumed free when writing this handoff.

### Services, automation and deadlines

| Item | State / boundary |
|---|---|
| `logika-architecture-20261027.service` | Completed; its transient unit is no longer loaded. |
| `logika-frozen-shadow-coordinator-20261026.service` | Parent campaign complete; no restart needed. |
| `logika-frozen-modelling-follow-up` | **PAUSED**, verified in the saved automation configuration on 28 October. |
| Architecture clock | Started 27 October **07:11:58 WIB**; admission ended **11:59:58 WIB**; total ceiling ended **13:11:58 WIB** that day. |
| Parent clock | Started 26 October **17:36:20 WIB**; original total ceiling ended 27 October **17:36:20 WIB**. It completed earlier. |

The recorded architecture/frozen-shadow campaign unit inventories on both nodes were empty at the 28 October check. Inspect current units before any future recovery action; do not infer running work from an old JSON field. Never repeat the completed architecture launch with `--hours 6` or reopen either expired clock.

## 5. Exact frozen data and validation protocol

Feature-freeze SHA-256:

```text
071c75e0f59cf92429bde9c76a64e27931e1eea03bbbbeac2e2589b5fee68dd6
```

Local immutable packets are under [the frozen archive](/home/parasite/Project/Competition/logika-2026/a100/faris/frozen-shadow-20261026/frozen). Each of the five fold folders retains its feature table, row roles, baseline predictions and manifest.

- **Master table: 116 columns.** Declared views are `champion90`, `shadow91`, `latent99`, `full100`, and `master`.
- Preserve the **exact original 90-column view, including its duplicate column**. An 89-column deduplicated control is not the matched baseline.
- Kumo views remain at most 100 columns. The architecture comparison used the already declared original90 and master116 views.
- Preserve the original **18-column correction head**, including its six coverage measurements.
- Shadow/latent learned feature values were built before the freeze with context-only selected-loan labels and borrower-disjoint historical supervision. Do not regenerate them during model-only experiments.

The task is default ranking, measured by **pooled two-sample KS**, maximized over 70,000 out-of-fold positive-class probabilities. Every applicant has exactly one outer prediction. Mean fold KS and AUC are diagnostics, not substitutes for the pooled objective.

Per original outer fold:

| Role | Applicants | Permitted use |
|---|---:|---|
| Context | 50,000 | Architecture training and preprocessing. |
| Correction | 6,000 | Fixed downstream correction and predeclared model selection. |
| Outer validation | 14,000 | Terminal evaluation after the candidate recipe is frozen. |

For the architecture campaign, the stopping epoch was selected on a fixed stratified **45,000/5,000 split inside context**, followed by fresh initialization and fitting all 50,000 context rows for the selected epoch count. Correction labels did not train the architectures or choose their stopping checkpoints.

Internal numeric normalization, missing masks, category embeddings and network tokens were declared model preprocessing. They did not overwrite source feature values. Preprocessing was fitted on context customers only. The master table contains learned representations already fitted on full context, so its inner stopping split is an intra-training device, not independent validation of those representations.

The two fixed seeds were averaged within each design. All ten fits were required before that design could enter selection. One correction half ranked designs; the other chose the ensemble size and baseline blend weight. Selection was frozen before outer scoring. The downstream original correction head and `.75 corrected + .25 maturity spline` mixture remained fixed.

**All 70,000 outer labels have historical experiment exposure. There is no untouched audit.** Bootstrap intervals are conditional reused-development evidence; a new seed or reshuffle does not manufacture independent confirmation. Historical repayment snapshot capture semantics also remain a documented assumption. Passing the artifact/evaluator audit does not resolve that source-provenance limit.

## 6. What has been completed since the older handoff

| Campaign | Verified outcome | Primary pooled KS | Matched delta |
|---|---|---:|---:|
| Horizontal feature expansion, 26 October archive | 1,510 completed fits; audited; no promotion | 0.5281372386 | −0.0024988656 |
| Exact90 plus small actuarial additions | Terminal evaluation exists; no promotion; no second-seed confirmation | 0.5305309186 | −0.0001051857 |
| Frozen shadow/latent + model comparison | 68 configurations, **340/340 fits**, terminal audit passed | 0.5303323155 | −0.0003037887 |
| New architecture comparison | 32 configurations, **160/160 fits**, terminal audit passed, zero failures | 0.5291772104 | −0.0014588939 |

These campaigns made **no new Kaggle submissions**. Their failure to improve does not prove that every possible feature or model design is ineffective.

Supporting artifacts:

- [Horizontal results](/home/parasite/Project/Competition/logika-2026/a100/faris/horizontal-20261026/RESULT_SUMMARY.md) and [audit](/home/parasite/Project/Competition/logika-2026/a100/faris/horizontal-20261026/reports/FINAL_AUDIT.json).
- [Exact90 augmentation terminal result](/home/parasite/Project/Competition/logika-2026/a100/faris/augment90-20261026/evaluation/RESULT.json). Its local `RESUME_STATE` and `LIVE_STATE` snapshots still say training; the terminal result is later. Do not restart based on those stale snapshots or treat this row as a newly audited two-seed success.
- [Frozen-shadow analysis](/home/parasite/Project/Competition/logika-2026/a100/faris/frozen-shadow-20261026/ANALYSIS.json) and [terminal audit](/home/parasite/Project/Competition/logika-2026/a100/faris/frozen-shadow-20261026/TERMINAL_AUDIT.json).

The horizontal campaign screened 21,514 feature definitions in 290 packs. The frozen-shadow round compared global/local-context Kumo, LightGBM, CatBoost, XGBoost, ExtraTrees, logistic regression, MLP and ranking corrections. No candidate exceeded the matched baseline in the frozen-shadow terminal comparison. Different model families did not all receive the same seed replication; inspect the configuration matrix before making a family-wide claim.

## 7. Latest architectural modelling: exact result and limits

Six architecture families were compared: embedded MLP, ResNet, stacked DCN-V2 adaptation, parallel DCN-V2 adaptation, official TabM 0.0.3, and an FT-Transformer adaptation. BCE/pairwise variants for ResNet and TabM, two frozen input views, two seeds and five folds produced **16 paired designs and 160 fits**.

Source motivation and adaptations are documented in [the source ledger](/home/parasite/Project/Competition/logika-2026/a100/faris/architecture-20261027/SOURCES.json); exact settings are in [CONFIGS.json](/home/parasite/Project/Competition/logika-2026/a100/faris/architecture-20261027/CONFIGS.json). Published benchmarks are motivation, not evidence of a Logika gain.

| Predeclared diagnostic | Original90 KS | Master116 KS |
|---|---:|---:|
| MLP / BCE | 0.52044041 | 0.52316211 |
| ResNet / BCE | 0.52297582 | **0.52496073** |
| ResNet / BCE + pairwise | 0.52488615 | 0.52475654 |
| Stacked DCN / BCE | 0.51990303 | 0.52279336 |
| Parallel DCN / BCE | 0.52086185 | 0.52422667 |
| TabM / BCE | 0.52280277 | 0.52238467 |
| TabM / BCE + pairwise | 0.52396621 | 0.52317892 |
| FT-Transformer / BCE | 0.52341465 | 0.52219121 |

The larger frozen view improved four of eight paired-design point estimates and reduced four. Pairwise ranking improved three of four matched objective comparisons. None of the 16 paired designs or 32 individual seed configurations exceeded the baseline.

The primary correction-selected blend was **0.5291772103794308**, delta **−0.0014588938605959445**. It improved only **1/5 fold** point estimates. Its 2,000-replicate paired class-stratified 95% interval for the change was **[−0.0034979203807281867, +0.0007581955395255267]**. Fixed-recipe seed deltas were **−0.0022356832894546708** and **−0.0013914159768170231**. No supported gain was established; the interval also does not establish a definite population loss.

The frozen baseline-blend weights were `[0.05, 0.70, 1.00, 0.05, 0.05]` across folds. They were not changed after seeing outer results. Do not retrospectively substitute the best outer family or weight and call it the primary result.

Training/evaluation finished 27 October **07:47:57 WIB**, about **35.98 minutes** after launch. The queue used 129 CPU fits and 31 CUDA fits on the two permitted nodes. No fitted weights were serialized. Full verification followed within the original six-hour ceiling.

The independent audit checked all five data seals, all 160 output hashes, exact70,000-row coverage, configs, seeds, training boundaries, selected epochs, source identities and selection order. All 52 reported KS/AUC summaries independently reproduced with maximum error `3.33e-16`; all 260 downstream correction fits reproduced the saved predictions exactly. Remote original outputs and their local copies were hash-verified.

The completion manifest contains **850 files**, totaling 48,272,108 bytes, excluding its own file. Its digest, rechecked on 28 October:

```text
3d6d5d23755806302321cd1842c9ca4a317df4e815ac68b0b6930b7bed3981d2
```

Key replay evidence: [local independent audit](/home/parasite/Project/Competition/logika-2026/a100/faris/architecture-20261027/LOCAL_TERMINAL_AUDIT.json), [remote receipts](/home/parasite/Project/Competition/logika-2026/a100/faris/architecture-20261027/REMOTE_RECEIPTS.json), and [audit source](/home/parasite/Project/Competition/logika-2026/a100/faris/architecture-20261027/audit/terminal_audit.py).

This was a finite comparison with fixed widths, learning schedules and two seeds. It does not exhaust architecture optimization. The unchanged correction head may favor its original baseline family. Do not turn the result into a universal claim that neural models cannot help.

## 8. Submission identity: do not trust the generic filename

| File / recipe | Retained public KS | Kaggle reference |
|---|---:|---|
| V25 | **0.56822** | `57444414` |
| B, maturity-only | 0.56818 | `57604838` |
| Prior root periodicity+maturity | 0.56678 | `57482053` |
| A, two-seed periodicity+maturity | 0.56656 | `57605161` |

The user's earlier uncertainty about `submission.csv` was resolved by exact downloaded-file identity. The root file is the prior periodicity+maturity refit, not V25 or candidate B.

- [Verified V25 CSV](/home/parasite/Project/Competition/logika-2026/submission_best_public_V25_056822_verified.csv), SHA-256 `e7f0748c6c8ded776f0a6a03bdebcd0787cf6c0057d0b078c31196fb34f58efe`.
- [Root submission.csv](/home/parasite/Project/Competition/logika-2026/submission.csv), SHA-256 `922788d13cd8ec53b7cbd732496a9200150a499fd4363455414e2cc6230a6b06`.

Both local hashes were rechecked on 28 October. Their historical receipt/public-score association comes from retained authenticated records; this handoff did not access the live leaderboard or refresh quota. Do not reuse old statements such as “three slots left” as current quota.

The repaired matched local V25 replay may differ from its older reported OOF value because preprocessing/protocol versions differ. Preserve both identities rather than inventing a conversion. A full-data refit does not receive a fresh independent validation score from its earlier OOF recipe.

## 9. Broader requests that remain incomplete

The user previously asked to try every listed foundation model and actuarial method. **Do not report those requests as exhausted.**

- [Foundation/model registry](/home/parasite/Project/Competition/logika-2026/a100/faris/model-arena-20261025/MODEL_MATRIX.json): the 25 October snapshot contains 98 entries representing 52 unique methods; 14 entries are marked five-fold complete and 84 not attempted. Later experiments overlap some families; this is not a synchronized present-day total of every run.
- [Actuarial method registry](/home/parasite/Project/Competition/logika-2026/a100/faris/actuarial-20261025/METHOD_MATRIX.json):165 entries. Snapshot statuses include 69 with unmet data requirements, 27 feature/valuation candidates, 20 pending auxiliary-target designs, 16 pending risk-feature designs, 15 pending predictive experiments, 11 credibility/binary-adaptation candidates and 7 related implementations completed. A related implementation is not proof that the exact listed actuarial model was tested.

Missing methods must be reconciled against actual experiment evidence, required data, implementation availability and applicable terms before scheduling. Do not manufacture irrelevant inputs to claim coverage. Feature-producing actuarial methods conflict with the present feature freeze unless the user changes that scope.

Older training archives and citations remain in the earlier handoff. Retained registry snapshots, generated configs, completed fits, aliases, candidate assessments and source references are different counts; do not add them together as independent experiments.

## 10. Known operational traps and continuation guidance

1. **No automatic restart.** Both relevant campaigns are complete and their clocks are expired. For new user-requested training, establish a fresh finite budget and separate campaign directory; preserve the previous scientific seals and exposed results.
2. **Do not repeat CPU Kumo.** The parent campaign's full-size CPU pilot exceeded its 22GiB cgroup limit. An interrupted-attempt retry bug produced 19 recorded memory kills; it was fixed and CPU Kumo disabled. These are resource failures, not evidence that Kumo predicts poorly. See [incident evidence](/home/parasite/Project/Competition/logika-2026/a100/faris/frozen-shadow-20261026/CPU_RESOURCE_LIMIT.json) and [retry repair](/home/parasite/Project/Competition/logika-2026/a100/faris/frozen-shadow-20261026/RETRY_HANDLER_FIX.json).
3. **Retired node2002 outputs need provenance.** Verified results from before the outage were retained; five unavailable attempts were explicitly replaced on permitted nodes. Do not merge old outputs into their replacements. See [retirement record](/home/parasite/Project/Competition/logika-2026/a100/faris/frozen-shadow-20261026/NODE2002_RETIREMENT.json).
4. **Stale metadata is not live execution.** Examples include the augmentation snapshots and the architecture resume state's preparation-stage pilot-pending label. The later passed runtime gate, terminal result, audit and actual unit inventory take precedence.
5. **Old model paths are not available-weight guarantees.** A separate 25 October cleanup removed models from its expressly authorized remote scope; newer campaigns deliberately retain parameters and predictions only. The old handoff's earlier “all models retained” text is superseded. Do not rerun cleanup or assume a historical model hash means its weight file still exists.
6. **Evidence before a submission.** Require complete row/fold coverage, exact feature/config/source identity, reproducible scores, seed/fold assessment and a defensible public-transfer case. The recent negative rounds supply no such candidate.
7. **No independent-holdout claims.** New designs informed by these results use exposed development evidence. Record that exposure and freeze the next comparison before scoring; do not relabel the same rows as a fresh audit.

For a successor asked only to understand the project: read sections 1–5 and the linked terminal artifacts, then answer from the retained evidence. For a successor asked to improve it: preserve the freeze and incumbent, diagnose a specific untested modelling limitation, reconcile the historical registry, declare a new comparison/budget, and distinguish a development win from a public improvement. No new hypothesis is pre-approved by this handoff.

## 11. Verification scope of this handoff

On 28 October the handoff author reread the final local results, both recent campaign states, submission review, model/actuarial registries and paused automation configuration; revalidated all 850 files covered by the architecture completion manifest; rehashed the V25 and root CSVs; and connected to node2001 and node2003 to verify their actual home directories. The canonical architecture live state still reports160 completed fits, zero failures and no submission. The earlier remote evidence mirror was verified by the completed campaign's receipts.

No live Kaggle scores/quota, new GPU-capacity guarantee, new model fit or new public result is asserted. Keep the earlier handoff and completed campaign packages unchanged; this is a new standalone handoff document.
