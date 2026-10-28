# Architectural modelling on the frozen Logika features

Status: **complete; 160/160 fits, zero failures, independently audited.** The frozen selected blend scored KS **0.5291772104** versus matched baseline **0.5306361042**. Target 0.54 was not reached and no development gain is supported. No refit or submission was made. See [RESULTS.md](RESULTS.md), `ANALYSIS.json` and `TERMINAL_AUDIT.json`.

The user requested another architectural modelling round while preserving the features. The existing feature freeze remains exact: SHA-256 `071c75e0f59cf92429bde9c76a64e27931e1eea03bbbbeac2e2589b5fee68dd6`. All five original feature tables, column values and row roles are reused from `/mnt/home/faris/logika-frozen-shadow-20261026/frozen`. There is no new feature construction, feature selection or representation regeneration.

The completed previous round remains immutable. Its selected ensemble scored KS 0.5303323155 against the matched baseline 0.5306361042; none of its 68 individual configurations exceeded the baseline. Those outer results have been retired into development evidence, as recorded in `EXPOSURES.jsonl`. No untouched final audit is available.

## Architecture comparison

There are **16 designs, two fixed seeds per design, five folds: 32 configurations and 160 planned fits.** The two input views are the already frozen original 90 columns and the complete 116-column master table.

| Architecture | Mechanism | Objectives |
|---|---|---|
| Embedded MLP control | Categorical embeddings and dense network under the same new training pipeline | BCE |
| Residual network | Three normalized residual blocks | BCE; BCE plus pairwise ranking |
| Stacked DCN-V2 adaptation | Three rank-32 cross layers followed by a deep branch | BCE |
| Parallel DCN-V2 adaptation | Cross and deep branches combined at the output | BCE |
| Official TabM 0.0.3 | Shared network with 16 ensemble members; train each member, average probabilities | BCE; BCE plus pairwise ranking |
| FT-Transformer adaptation | Numerical/categorical feature tokenizer, two attention blocks and a classification token | BCE |

The original papers and fetched Consensus records are documented in `SOURCES.json`. ResNet, DCN-V2 and FT-Transformer use explicitly documented task-specific adaptations. TabM uses the installed official implementation; width, member count and training settings differ from the paper benchmarks. Published performance does not establish a Logika gain.

Internal normalization, missingness masks, categorical embeddings and attention tokens are model preprocessing or learned network operations. They do not add to or overwrite the frozen source feature tables. Their parameters are fitted using context customers only. The MLP control shares this preprocessing, allowing architecture comparisons under a common pipeline.

## Training and selection

- Preserve the original 50,000 context / 6,000 correction / 14,000 validation customers per fold.
- Choose the stopping epoch using a fixed 45,000 / 5,000 split entirely inside the context customers, then initialize a new model and fit all 50,000 context customers for that epoch count.
- Do not use correction labels for architecture training, checkpoint selection or pairwise training. This addresses the training-informed candidate ranking in the previous round.
- The 116-column view contains previously learned frozen representations trained on the full context. Its inner split is an intra-training stopping device, not an independent evaluation of those representations.
- Average the two fixed seeds for each design, requiring all ten fits before a design enters selection. Rank on one correction half; choose ensemble count and baseline blend weight on the other. The original 18-column correction head and maturity spline remain fixed for matched comparisons.
- Freeze the entire selection recipe before terminal outer scoring. Apply that same recipe to each seed separately for stability diagnostics; do not reselect after viewing outer results.
- Target KS 0.54 is separate from a supported development gain and public improvement. Promotion requires a positive primary delta, a paired bootstrap lower bound above zero, at least three positive fold deltas and positive deltas for both fixed-seed recipes. No independent-confirmation claim is available.

The exact contract is `VALIDATION_CONTRACT.json`; the full parameter matrix is `CONFIGS.json`.

## Checks completed

`GATE_PREP.json` verifies the inherited feature, manifest and row-role hashes in all five folds, metric known cases, class support, context-only preprocessing under a held-out-row counterfactual, safe unknown-category handling, and finite gradients plus parameter updates for all six architecture families. These preparation checks use synthetic optimizer inputs, not a scored competition-data pilot.

The bounded official TabM pilot passed before broad admission: three internal epochs, a fresh context refit for the selected three epochs, 20,000 finite probabilities and repeat prediction agreement. `GATE.json` records the pilot result and prediction hashes, verified runtime environments, and the preparation checks. The pilot is stored separately from the 160 planned main fits.

## Duration and execution

The user supplied a **six-hour ceiling**. The immutable `CLOCK.json` starts at 00:11:58 UTC, closes admission at **04:59:58 UTC / 11:59:58 WIB**, and ends all computation by **06:11:58 UTC / 13:11:58 WIB on 27 October 2026**. It includes qualification, pilot, training and evaluation; research and source preparation are recorded separately. The final 20% is reserved for draining workers and evaluation. The old campaign's allowance is not reused. Never reset or extend this clock on resume.

Only nodes **2001 and 2003** are enabled. Node 2002 and A100 remain excluded. Work stays under `/mnt/home/faris/logika-architecture-20261027` on both permitted hosts. Node 2001 uses `/mnt/home/faris/logika-model-arena-20261025/env311/bin/python`; node 2003 uses `/mnt/home/faris/faris/kumo-coverage-placement-20261021/.venv/bin/python`. The frozen inputs remain in their original archive and are read only.

The campaign was launched with `source/launch.py --hours 6` as the bounded user service `logika-architecture-20261027.service`. Its systemd runtime is capped at six hours. The one-time `--hours` launch must never be repeated. This completed campaign must not be resumed. During an interrupted active run, the recovery policy was to reconcile actual workers and use `source/launch.py --resume`, preserving the existing clock. Admission honors idle GPU detection, CPU reservations, RAM budgets and per-fit limits. The prepared coordinator does not interfere with other users' workloads.

The coordinator preserves isolated worker logs and explicit failures. Missing outputs do not trigger an unbounded retry loop. Any failed design must be diagnosed or reported incomplete; its execution failure does not establish negative predictive evidence.

Each successful fit retains hyperparameters, seeds, source/input identities, preprocessing recipes, training traces and compact predictions. In-memory training states are released; fitted weights and KV caches are not serialized. No automatic refit or Kaggle submission is part of this queue.

`PREPARATION_MANIFEST.json` is the preserved pre-launch snapshot. Launch fills in the authorized duration in `VALIDATION_CONTRACT.json` before creating the clock. Therefore that preparation snapshot is not the current contract seal; use the hashes recorded in `CLOCK.json` and `GATE.json` for the running campaign.
