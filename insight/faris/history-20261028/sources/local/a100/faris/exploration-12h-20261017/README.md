# Twelve-hour Logika exploration

The frozen champion is the five-model trimmed ensemble in the existing root `submission.csv`, SHA-256 `be059078ade95bc6758522e8d72849e3f793e57380e2c083e05c77c64335e192`. Its public score **0.56188 is user-reported**; OOF KS is 0.525771439 and historically reused confirmation KS is 0.530496149. The stretch target is 0.55 development KS, not a promised score.

This campaign produces research evidence and retained models. It never creates competition-test predictions or another submission. The champion is separately protected in the PC2 archive.

## Search and evaluation

The explicit registration catalogue contains 4,096 configurations over 192 new representations plus v3 and five champion views. Resolved no-op/equivalent configurations are removed before per-host queues are sealed. Historical record IDs are inventoried separately and are not claimed to be independent experiments. Five champion controls intentionally repeat earlier recipes. Report unique recipes, seed repeats, fold fits, retries and ensembles separately.

The actuarial branch adds binomial logit/cloglog GLMs, cross-fitted WOE scorecards and beta-binomial credibility scorecards, plus the spline/EBM additive controls. It reports default-probability reliability, Brier loss and observed/predicted defaults alongside KS. Severity and survival models are excluded because their outcome fields are absent.

Families: CPU LightGBM and RX6600LE OpenCL LightGBM, native-categorical CatBoost, CUDA XGBoost, elastic-net/spline/kernel statistical models, EBM, ExtraTrees, official-package TabM, local residual MLP and FT-style transformer, a **TabR-inspired retrieval MLP**, GRU and DeepSets. The local retrieval model is not an exact official TabR reproduction; that port is excluded until tested. Node380 optionally adds cached Kumo Large and newly staged official TabICLv2. TabPFN-2.5 is excluded when checkpoint access/terms are unresolved; there is no unattended acceptance of gated terms.

New feature blocks cover repayment deterioration/maturity/exposure, bureau provider disagreement/staleness, income and balance changes, shopping regularity/category-relative prices/cancellation behavior, cross-source affordability and representation learning. Actual true-gain importance is compared with split-count importance. Original champion `gain` remains split-count importance so controls reproduce the original pipeline. Signed logs, scaling, encodings, selection, PCA/SVD/grouped PCA, clustering and autoencoders are fitted only on training partitions. Extra history-derived shrinkage priors use only training-partition counts.

Every completed screening recipe covers all 56,000 development rows exactly once over the original three folds. Inner early stopping refits from scratch on the full outer-training partition. Target encoding is inner-cross-fitted. Retrieval excludes self-neighbors; query labels never enter contexts. Event encoders use only pre-decision observations. Fixed combinations require aligned complete OOF arrays. Learned ensemble jobs construct new inner OOF arrays inside each outer training partition; ordinary global OOF never fits their meta weights.

The CPU/GPU queues are independent and host-sharded. Screening ends at hour 9; up to 12 nested ensemble jobs can enter after hour 7.5. At hour 9, evaluate up to 4,096 fixed combinations and freeze up to six stability nominees. Compare nominees and all five champion components on a second seed, a second three-fold assignment seeded 20261018, and a separate chronological diagnostic. At hour 10.5, freeze at most three finalists and the primary nominee. At hour 11.5, terminal confirmation is assessed with paired bootstrap uncertainty and Holm-adjusted secondary comparisons. The last half hour reconciles artifacts and writes reports. Jobs that do not fit their phase remain partial/unstarted; they are never called completed negative findings.

All validation has historical/adaptive label reuse. Changing folds does not create untouched labels. Positive local scores and conditional bootstrap uncertainty do not prove a public leaderboard gain or transfer to a larger model.

## Operation and resources

Local control from the repository root:

```sh
python a100/faris/exploration-12h-20261017/control.py status all
python a100/faris/exploration-12h-20261017/control.py collect all
python a100/faris/exploration-12h-20261017/control.py report all
python a100/faris/exploration-12h-20261017/control.py stop all
python a100/faris/exploration-12h-20261017/control.py resume pc2
python a100/faris/exploration-12h-20261017/control.py verify pc2 --job JOB_ID
python a100/faris/exploration-12h-20261017/control.py replay pc2 --job JOB_ID --output NEW_DIRECTORY
```

`prepare`, `validate`, and one-time `launch` are also supported. Resume preserves the original clock and cannot revive a finished/expired campaign. A replay is new training and requires an explicit request after handoff. PC3 access always originates on PC2; PC1 uses a dedicated forced-command key with forwarding disabled. Credentials remain under private `.credentials/`, outside source and snapshots.

- PC1: `/home/agung/faris/exploration-12h-20261017`, 16 CPU threads, one RX6600LE OpenCL queue.
- PC2: `/mnt/home/logika-faris-exploration/exploration-12h-20261017`, 12 CPU threads, both RTX3060s, durable archive and collection service.
- PC3: `/mnt/home/faris/faris/exploration-12h-20261017`, 12 CPU threads, both RTX3060s.
- Node380: `/home/taqiyudinmiftah/faris/exploration-12h-20261017`, GPU-only foundation-model work after fresh preparation/admission; the original deadline applies.

All three PC services share a host-local `explore12h.slice` enforcing their aggregate CPU/RAM quotas and zero swap. RAM limits use measured available capacity with an OS reserve. GPU jobs start exclusive per device; multiplexing remains disabled without throughput evidence. The new 12-hour clock begins only at explicit launch, after dependency/data staging and synthetic verification. Actual-data feature construction and calibration are inside the clock. Node380 may join later and receives no additional time.

The PC2 dedicated storage has a 300 GiB campaign ceiling. Admission pauses before archive headroom is exhausted. Worker models are pruned locally only after archive hashes match, an archive receipt is durable, and the source still matches its registered artifact hash. Every distinct model remains in PC2 storage; identical immutable large weights are content-shared. Prediction evidence, configuration, logs, metrics and model indexes stay on workers. Interrupted transfers never authorize pruning. A stopped collector cannot silently delete local artifacts; workers stop admitting before local spool limits.

Cleanup of prior work removed only a rebuildable dependency download cache. See `evidence/cleanup.json`; prior experiment models, OOF arrays, source, negative findings and champion artifacts remain preserved.

## Evidence and output

`clock.json` contains the authoritative start/deadline. `logs/` records installation, preflight, controller, collection and supervisor behavior. Each campaign retains its immutable runtime/source/evidence seal, complete package inventory, SQLite ledger, per-attempt parameters, identities, stdout/stderr, fitted preprocessing, models, validation outputs, resource samples and statuses. `archive/` holds verified consolidated copies and protection receipts.

Automatic reports include individual, fixed-ensemble and nested scoreboards; feature effects; coverage; cohort errors; seed/fold stability; frozen finalists; terminal confirmation; A100 handoff and the final Markdown report. Missing confirmations and unsuccessful gates are explicit. Source and data hashes are scientific provenance, not proof against selection bias.

Synthetic checks cover tied KS, point-in-time rejection, label-free feature construction, fold-local transforms, nested label isolation, CPU/GPU queue behavior, archive-pruning safety, tampering, failures/timeouts, restart/deadline semantics, detached SSH survival and real AMD/NVIDIA training. The excluded 1,000-job stress test was not rerun. Node380 cannot be represented as tested/running until its endpoint recovers and its own readiness check passes.

After one verified launch handoff, ordinary scripts run this campaign. There are no AI agents, AI monitoring loops or recurring chat automations.

Startup repairs preserve the clock and model catalogue. They cover resumable feature distribution, bounded node transfers, the public TabICL checkpoint repository (`jingang/TabICL`), an empty optional weights directory, compact control status and complete catalogue status counts. These orchestration edits received syntax/targeted live checks; the full 31-test run preceded the final orchestration edits. Feature-effect reports require the same CPU/OpenCL execution backend; unmatched anchors are reported as coverage gaps instead of attributing backend differences to features.

The champion reproduction gate passed exactly (maximum OOF prediction difference 0.0; KS 0.525771438885202) after pinning its three LightGBM components to PC1 and its two logistic components to PC2, matching their historical execution hosts. PC1 selected Cooperlake BLAS kernels and PC2 selected Zen kernels; moving the PCA/SVD components between them changed predictions despite identical source, data and package versions. The initial mismatched runs remain retained as portability evidence. Keep fitted reductions and their execution provenance in any later A100 handoff. All resume operations preserve the original 23:55:35 WIB deadline.
