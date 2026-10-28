# Second-wave feature research

This adds **4,096 new representations and 16,384 new pipeline recipes**, with four
repeated logical controls counted separately. Each representation uses the same
small LightGBM, CatBoost, XGBoost, and regularized linear probes. The generator
rejected 5,135 previous or structurally equivalent candidates. Its signatures
include resolved columns, active priors, and preprocessing behavior. The exact
catalogue includes uncovered factor pairs, matched single-change contrasts,
source/view combinations, and seeded higher-order combinations.

The first wave already used its campaign confirmation partition. Its nominee and
evidence remain frozen. This second wave provides **development-only research**;
it does not claim new independent confirmation. The worker rejects confirmation
tasks and blocks loading confirmation arrays. Fitting, features, metrics, and
ensemble implementations match the parent source. Runtime changes route work
through the guard and check estimated time before admitting a job.

Each PC stores its whole owned queue at
`~/faris/logika-meta-experiments/meta-20261016-v1-wave2/`. PC3's home is
`/mnt/home/faris`, and its connections originate on PC2. Banks and environments
refer to each machine's verified parent files; no private key was copied.

Limits remain PC1: 16 workers and a 2 GiB OS reserve; PC2/PC3: 7 workers, seven
CPU-equivalents, 20 GiB RAM, zero swap, and one visible GPU each. GPU jobs count
toward worker limits. PC1 keeps its 3 GiB run admission budget and the parent plus
extension stays below its 10 GiB project budget. Verified completed evidence is
archived to PC2 before local compaction. PC2 admits up to 48 GiB of additional
run artifacts with a 2 GiB filesystem reserve.

Parent executor groups stop before extension groups start. Rebuilding the first
wave's report happens inside the new PC2 aggregate resource group and reads its
already frozen confirmation evidence. That report is separate from this search.
Other workloads are untouched.

The shared clock is unchanged:

- Original start: **16 October 2026, 17:29:39 WIB**.
- Search admission ends: **17 October, 02:59:39 WIB**.
- Hard deadline: **17 October, 05:29:39 WIB**.

Every ranked fitted recipe needs complete three-fold OOF. Failed, partial, and
unstarted work remains explicit. Later stages compare fixed and genuinely nested
ensembles and perform bounded repeat/temporal checks. Folds, retries, and repeated
controls do not inflate new-recipe counts.

All three PCs passed the original 16 acceptance tests, four extension checks,
source/data/environment checks, backend probes, and resource-limit readback. PC2
also passed a complete three-fold run through the guarded worker, independently
verified from saved predictions. See `preflight/`, `parent_evidence.json`,
`source_manifest.json`, and `scheduler_revision.json`.

Read current status from the extension directory on PC2:

```sh
.venv/bin/python -m metaexp status
```

The service is `metaexp-meta-20261016-v1-wave2-coordinator`. Automatic restart
preserves its entry point and deadline. Do not replace it with the original
generic launch command. The parent's `research_extension.json` points here.

Outputs include `reports/summary.json`, `reports/progress.md`,
`reports/final_report.md`, `reports/feature_effects.json`, separate pipeline and
ensemble scoreboards, `reports/untested_recipes.json`, and
`reports/a100_handoff.json`. Completed pipelines retain their resolved recipes,
preprocessing audits, OOF evidence, logs, timing, and resource samples.

Deployment uses the parent directory's `deploy_research_extension.py`: stage,
prepare, seal, preflight, then launch. It preserves the parent's time window.
