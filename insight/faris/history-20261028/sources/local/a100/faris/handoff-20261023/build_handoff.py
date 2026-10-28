"""Build a dated, source-linked handoff without starting any experiment."""
from pathlib import Path
import json,csv,hashlib,datetime

PROJECT=Path(__file__).resolve().parents[3]
BASE=PROJECT/'a100/faris';OUT=Path(__file__).resolve().parent
PC2='/mnt/home/logika-faris-exploration'
def read(p):return json.loads(Path(p).read_text())
def link(p,label=None):
 p=Path(p).resolve();return f'[{label or p.name}]({p})'
live=read(OUT/'PC2_LIVE_PATHS.json');node=read(OUT/'NODE380_LIVE_PATHS.json')
special={
 'depth-20261017':('PC2','/home/faris/faris/depth-20261017'),
 'logika-meta-experiments':('PC2','/home/faris/faris/logika-meta-experiments'),
 'submission-research-20261017':('PC2','/home/faris/faris/submission-research-20261017'),
 'submission-pass2-20261017':('PC2','/home/faris/faris/submission-pass2-20261017'),
 'kumo-large-node380-20261016':('node380','/home/taqiyudinmiftah/faris/kumo-large-node380-20261016'),
 'kumo-quantile-mean-20261021':('node380','/home/taqiyudinmiftah/faris/kumo-quantile-mean-20261021'),
 'tabicl-compact-a100-20261021':('node380','/home/taqiyudinmiftah/faris/tabicl-compact-a100-20261021'),
 'tabicl-conditional-mean-20261021':('node380','/home/taqiyudinmiftah/faris/tabicl-conditional-mean-20261021'),
}
index=[]
for p in sorted(BASE.iterdir()):
 if not p.is_dir()or p==OUT or p.name.startswith(('__','model-cleanup-','home-migration-')):continue
 seen=live['studies'].get(p.name,{});host,path=special.get(p.name,('PC2',PC2+'/'+p.name))
 anchors=[n for n in ['README.md','RESEARCH.md','REPORT.md','plan.json','result.json','RESULT.json','FINAL_STATUS.json','OOF.npz','oof.npz','models','folds','reports','results','source']if(p/n).exists()]
 if p.name=='bureau-occasion-fe-20261021':host='not deployed';path='';scope='Prepared code only; no default-model training/result.'
 elif p.name in special:scope='Remote root verified on 2026-10-23; see host evidence.'
 elif seen.get('exists'):scope='PC2 directory existence verified on 2026-10-23; result completeness requires its terminal evidence.'
 else:scope='Local snapshot retained; generic PC2 location not verified. Check alternate/host paths.'
 fresh=OUT/'evidence'/(p.name+'__result.json')
 index.append(dict(study=p.name,local_path=str(p.resolve()),local_anchors=anchors,primary_remote_host=host,primary_remote_path=path,PC2_standard_root_exists=seen.get('exists',False),verification_scope=scope,fresher_result_snapshot=str(fresh.resolve())if fresh.exists()else None))
(OUT/'EXPERIMENT_PATHS.json').write_text(json.dumps(index,indent=2))
with(OUT/'EXPERIMENT_PATHS.csv').open('w',newline='')as f:
 fields=['study','local_path','primary_remote_host','primary_remote_path','PC2_standard_root_exists','verification_scope'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k]for k in fields}for r in index)

feature=read(BASE/'feature-only-20261021/FEATURE_EVIDENCE.json')
feature_table='| Study / recipe | Pooled KS | AUC | KS fold wins vs repaired baseline |\n|---|---:|---:|---:|\n'
for x in feature['rows']:
 feature_table+=f"| {link(BASE/x['study'],x['study'])} / `{x['recipe']}` | {x['KS']:.9f} | {x['AUC']:.9f} | {x.get('KS_fold_wins','?')}/5 |\n"
path_table='| Study directory (local link) | Primary remote host | Primary remote root |\n|---|---|---|\n'
for x in index:path_table+=f"| {link(x['local_path'],x['study'])} | {x['primary_remote_host']} | `{x['primary_remote_path'] or 'No deployment; local draft'}` |\n"

doc=r'''# Logika 2026 — experiment and model handoff

Prepared from **23 October 2026** inspections; finalized **24 October 2026, Asia/Jakarta**, for another model working in the same workspace. Remote paths and access were not rechecked on 24 October. This is a handoff of retained evidence and access routes. No training, resubmission, cleanup, reboot or process termination was performed to create it.

## 1. Read these first

1. The current delivered CSV is `/home/parasite/Project/Competition/logika-2026/submission.csv`, SHA-256 **`922788d13cd8ec53b7cbd732496a9200150a499fd4363455414e2cc6230a6b06`**. The named identical copy is `submission_ks05309665_periodicity_maturity_full.csv` in the same directory. Both hashes were rechecked on 23 October.
2. It is a **new full-data refit of `shopping-periodicity-fe-20261021/true`**, using maturity-aligned repayment features. The selected research recipe had **pooled OOF KS 0.5309665314756384 / AUC 0.8469295983174442**, on 70,000 rows and five folds. The refit itself has **no new validation score or known public score in this chat**. Do not attach that OOF value to an in-sample refit metric.
3. The user explicitly selected the highest-KS recipe despite its lower AUC, then explicitly required a full-data retrain. Preserve that choice. Record AUC honestly; do not silently substitute the older joint-metric candidate.
4. The **0.54 research target was not reached**. Earlier timed searches are expired; do not restart them or reset their clocks. The latest full-data submission job is complete and all five models are retained.
5. **Agung's PC is now accessible only from PC2:** `ssh faris@labkc-comm1`, then `ssh agung@10.201.72.28`. This password login and the important experiment directories were verified on 23 October. The former `agung@100.89.211.119` route is deprecated.
6. The teammate master log is on node380 at `/home/taqiyudinmiftah/adit/Logika/insight/goal054_offset_baseline_20261019/v7/reports/EXPERIMENT_MASTER_LOG.md`, with `.csv` and `.json` siblings. A fresh read on 23 October still found **118 active experiment records**, V00–V117, and best reported public result **V25, 0.56822**. The refreshed record now states an authenticated Kaggle receipt and exact CSV-hash verification; we read that retained record, **did not independently call Kaggle** during this handoff.

Primary entry points:

- Latest full-data delivery: [README](./a100/faris/submission-full-periodicity-20261021/README.md), [delivery receipt](./a100/faris/submission-full-periodicity-20261021/LOCAL_DELIVERY.json), [independent audit](./a100/faris/submission-full-periodicity-20261021/FINAL_INDEPENDENT_DELIVERY_AUDIT.json).
- Latest feature evidence: [table](./a100/faris/feature-only-20261021/FEATURE_EVIDENCE.md), [machine-readable results](./a100/faris/feature-only-20261021/FEATURE_EVIDENCE.json), [portable maturity factory](./a100/faris/feature-only-20261021/handoff/README.md).
- Historical reconstruction: [inventory](./a100/faris/non-dominated-research-20261021/inventory/REPORT.md), [reconciliation](./a100/faris/non-dominated-research-20261021/inventory/reconciliation.json).
- All local studies and remote mappings: [JSON](./a100/faris/handoff-20261023/EXPERIMENT_PATHS.json), [CSV](./a100/faris/handoff-20261023/EXPERIMENT_PATHS.csv).
- Current master state: [research-state.json](./a100/faris/non-dominated-research-20261021/research-state.json). It contains many **old auxiliary status fields**. Its latest phase is `full_data_highest_KS_submission_delivered`; prefer final artifacts and current process identity over old PIDs.

## 2. Machine access and operating boundaries

### PC2 — primary archive, training host and jump host

```sh
ssh faris@labkc-comm1
```

- Primary archive: `/mnt/home/logika-faris-exploration/`.
- Older projects: `/home/faris/faris/` (`~/faris/` may resolve through the home mount).
- Pinned environment for recent 70k work: `/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python`.
- Training hardware: two RTX 3060 GPUs and 12 CPU threads in the campaign records. Both GPU names and archive paths were checked live on 23 October. Available archive disk space at inspection: **__FREE_GIB__ GiB**, a snapshot, not a quota or future guarantee.
- The latest final refit ran here in `submission-full-periodicity-20261021/`; unit `logika-full-periodicity-refit-20261021.service`.

### PC1 — Agung, new route

```sh
# First enter PC2.
ssh faris@labkc-comm1

# Run this on PC2, not on the laptop.
ssh agung@10.201.72.28
```

- Verified hostname: `labkc-ThinkCentre-M75t-Gen-5`; home `/home/agung`.
- Latest user-provided password is stored in the **private local file** `/home/parasite/.config/logika-2026/handoff-access-20261023.json`, mode `0600`, parent directory `0700`. Read it locally when an interactive SSH prompt requires it. It is deliberately outside the repository and absent from this Markdown. If handing off outside this machine, provide that credential separately.
- Passwordless `BatchMode=yes` authentication did **not** work for the new destination; password authentication did. Do not mislabel this as a network outage.
- Historical hardware: 16 CPU threads, AMD RX 6600 LE, plus integrated AMD graphics. No CUDA; old AMD LightGBM jobs use OpenCL after backend checks.
- Verified roots: `/home/agung/faris/logika-meta-experiments/meta-20261016-v1`, its `-wave2` sibling, `/home/agung/faris/depth-20261017/campaign`, `/home/agung/faris/depth-20261017/campaign-amd`, `/home/agung/faris/exploration-12h-20261017`.
- Old controllers/configurations may hardcode `100.89.211.119` or an old forced-command SSH key. Keep immutable research snapshots unchanged. A newly authorized execution bundle must use the new hop; do not blindly run an old `control.py ... all` command.

### PC3 — only through PC2

```sh
ssh faris@labkc-comm1
ssh -p 2003 faris@gpu-labkc.ub.ac.id
```

- Verified hostname: `labkc-comm3`; home `/mnt/home/faris`.
- Research root: `/mnt/home/faris/faris/`.
- Relevant retained directories verified: `depth-20261017`, `exploration-12h-20261017`, `kumo-coverage-placement-20261021`, `catboost-convergence-20261021`, `kumo-supervised-view-20261021`, `nested-correction-20261021`, `measurement-risk-20261020`, `context-model-retention-20261021`, `non-dominated-research-20261021`.
- Historical hardware: two RTX 3060s, 12 CPU threads. Recheck current capacity before any authorized work.

### Node380 — Jupyter API, not ordinary SSH

Use the explicit node380 profile from the local project root:

```sh
python3 a100/profile.py node380 term 'whoami; pwd'
python3 a100/profile.py node380 term 'nvidia-smi'
python3 a100/profile.py node380 sessions
```

- Profile: `/home/parasite/Project/Competition/logika-2026/a100/profiles/node380/profile.json`.
- Endpoint: `https://node380-ai-hub.ub.ac.id/user/taqiyudinmiftah/`; account `taqiyudinmiftah`.
- Root: `/home/taqiyudinmiftah/faris/`.
- Credentials: the profile's existing `sysinfo.txt` / `A100_NODE380_API_TOKEN`. Do not print tokens or copy them into logs or this document.
- Reuse the persistent **`term` kernel**, and issue its commands sequentially. Do not create a new helper/control kernel for every read. An idle kernel is not a failed training job.
- Node environment for newer studies: `/home/taqiyudinmiftah/faris/exploration-12h-20261017/.venv/bin/python`. Plain `python3` is a different interpreter and may lack ML dependencies.
- Historical accelerator: A100 80GB. Node CPU preprocessing is slow; later campaigns staged CPU-heavy feature construction on PCs. Check live cgroup RAM and GPU memory rather than assuming the host's resources equal the container's allowance.
- Do not confuse node380 with profile `faris`, which is the older node480 account `fariselhakim`. Old tools can default to node480; always select the profile explicitly.

### User constraints

- No poweroff or reboot. Never kill or interfere with unrelated users' processes.
- Keep new project code under a Faris-owned directory. Preserve all trained models and failed/partial evidence; remove redundant copies only after archive hash verification and within authorized cleanup scope.
- Early campaigns had 7-worker / 20GiB / 12GiB GPU limits on PC2/3. The user later authorized maximum available compute for later runs. Those old values are **historical bundle limits**, not universal current authorization. Do not edit old cgroup manifests to disguise how an experiment ran. Check actual free capacity and unrelated workloads first.
- No agents, recurring AI monitoring or automatic campaign restarts were part of the final feature campaign. No new job was started during this handoff.
- User asked for Markdown handoff now. The safe next action is reading evidence; this document is not a request to restart training or upload a CSV.

## 3. Current delivered artifact and complete training graph

Workspace: `/home/parasite/Project/Competition/logika-2026`.

| Artifact | Absolute local path |
|---|---|
| Current competition CSV | `/home/parasite/Project/Competition/logika-2026/submission.csv` |
| Descriptive identical CSV | `/home/parasite/Project/Competition/logika-2026/submission_ks05309665_periodicity_maturity_full.csv` |
| Final run directory | `/home/parasite/Project/Competition/logika-2026/a100/faris/submission-full-periodicity-20261021/` |
| Saved source | Same run, `source/{prepare.py,train.py,finish.py,baseline.py,feature_head.py,spline.py,gpu_backend.py,maturity_features.py}` |
| Data/prior builders | Same run, `feature_build/source/build.py`, immutable `feature_build/vendor/` |
| Five local model bundles | Same run, `models/fold0/` through `models/fold4/` |
| Full archive including feature tables/workspaces | PC2 `/mnt/home/logika-faris-exploration/submission-full-periodicity-20261021/` |

Every model bundle retains `kumo_model.joblib` (canonical context and labels, feature list, seed, checkpoint hash), `head.joblib`, `spline.joblib`, `maturity_transformer.json`, `test_predictions.npz`, and `complete.json`. PC2 additionally retains `raw_predictions.npz` and `test_features.parquet`. Do not confuse these final models with the earlier OOF models.

In **each** of five final runs:

1. Keep the original 6,000 correction IDs and use the other **64,000** labeled applications as Kumo context. This is a refit on all **70,000** labels across the full graph; Kumo alone does not use 70k rows. The context and correction roles remain disjoint.
2. Rebuild label-free shared feature statistics using only that 64k context. Refit maturity priors on the same IDs.
3. Use **frozen Kumo Large**, 90 original core inputs, eight estimators, seeds **2026–2030**, FP16, estimator batch size 1 and query batches of 512. No gradient tuning of the pretrained backbone.
4. Train the existing additive offset correction on the disjoint 6k rows using their out-of-context Kumo predictions. Inputs: **29**. Median imputation, missing indicators, three-knot quadratic spline basis, standardization; L2 **0.003**, correction strength **0.5**, L-BFGS-B maxiter **500**. Full settings and fitted preprocessing are retained.
5. Train the spline/logistic branch on all **70,000** labels, inputs **88**, five-knot cubic splines, `C=.05`, `solver='newton-cholesky'`, `max_iter=80`, `tol=1e-5`.
6. Per-run prediction = **75% corrected Kumo + 25% spline**. Final prediction = equal arithmetic average of the five runs.

Feature placement matters:

- **Kumo stays at 90 core inputs**, using the inherited repaired feature definitions. Neither the 11 periodicity additions nor the six V25 coverage additions are injected into its context in this selected recipe.
- Correction = original 12 fields + six V25 fields + 11 periodicity fields = **29**.
- Spline = original 77 fields + 11 periodicity fields = **88**. The six V25 additions are a correction-stage block; do not silently append them to the spline.
- The maturity change replaces **only two existing posterior means** in correction and spline. Kumo predictions are generated before those stage-specific substitutions. Do not accidentally replace those fields in Kumo too.

The retained Kumo model is the frozen pretrained weights, exact context/labels/schema, seed and code. Attention caches are reproducible inference indexes; the final refit retained canonical context models rather than duplicating large cache files. Historical CV caches remain archived separately.

Validation/delivery receipts:

- `FINAL_STATUS.json`: five complete full-data runs, 30,000 finite test predictions.
- `FULL_DATA_ROLE_AUDIT.json`: 64k + 6k = 70k per run, correct context prior IDs, no test overlap.
- `FINAL_INDEPENDENT_DELIVERY_AUDIT.json`: context rows and labels checked; spline fit rows and correction settings checked; sample order verified; independently reaggregated five-run predictions, max error **3.33e-16**.
- `LOCAL_DELIVERY.json`: final SHA, named path and previous SHA.
- `ENVIRONMENT.json`, `SDM_SOURCE.json`, `requirements-resolved.txt`: actual environment and source hashes. Training used Python 3.11.13, NumPy 2.2.6, pandas 2.3.3, SciPy 1.16.3, sklearn 1.7.2, torch 2.7.1+cu126, joblib 1.6.0, pyarrow 22.0.0. SDM was imported from pinned source; use its source manifest rather than assuming package-manager metadata exists.

Previous root CSV SHA: `1ba5b8677549f5d05fdda6d63fa5f978e731247275a74ff91f3bdd39ac068036`. Backup:
`/home/parasite/Project/Competition/logika-2026/a100/faris/submission-full-periodicity-20261021/previous_submission_1ba5b8677549f5d05fdda6d63fa5f978e731247275a74ff91f3bdd39ac068036.csv`.

CSV contract: **30,000** unique application IDs, exact columns **`application_id,default_probability`**, official sample order, probabilities strictly within (0,1). This thread did not upload the new CSV. No subsequent public score was provided here.

## 4. Validation protocols — keep them separate

| Protocol | Data and use | Critical limitation |
|---|---|---|
| Legacy meta/depth/exploration | 56k development with frozen three-fold OOF; 14k confirmation from original 20% split, stratified target/month, seed 20261016; development seed 9173 | Confirmation and development labels had historical exposure. This is not an untouched private holdout. |
| Original teammate / later matched studies | All 70k, original five folds, seed 31027; per fold 50k Kumo context, 6k correction, 14k validation; spline on 56k outer train | Score **pooled** complete OOF once per row. Some historical feature statistics were estimated globally; compare only against a matched version. |
| Repaired V25 studies | Same 70k/5-fold roles; unsupervised shared statistics refit within each 50k context | Matched baseline **KS .5302185366909258 / AUC .847025849979685**. This is the baseline for the 18 recent feature contrasts. |
| Final submission refit | Five runs with 64k context, 6k correction, 70k spline; 30k test | No outer validation left in the final fits. Their selection score comes from the earlier OOF recipe, not an in-sample final-fit score. |

- Historical V25: **.5302846600440411 / .8470292789071159**, from inherited globally estimated covariate priors. Do not silently interchange it with repaired V25.
- Original teammate pre-V25 baseline: **.5292994659005608 / .8472026883917805**. Several earlier studies compare against this, not repaired V25.
- Tie-aware KS is maximum absolute difference between empirical class CDFs at tied-score boundaries. Directional KS is also retained. AUC is rank/tie-aware. Independent implementations were checked.
- Pooled OOF KS and mean-fold KS are different quantities. Some old candidates improved mean-fold KS while lowering pooled KS. Always name the aggregation used.
- All completed main-screening recipes require complete aligned OOF, not one good fold. Folds/seeds/retries/assessment rows are not independent experiments.
- Learned preprocessing and priors belong inside the relevant training roles. Target encoding needs inner cross-fitting. Learned stacking/routing needs actual nested predictions; global OOF filtering is insufficient.
- Raw `max_days_past_due` / `amount_paid` snapshot provenance remains unresolved. Calendar eligibility checks do not prove those snapshot values were available at a past underwriting instant. This limitation is inherited by the baseline and chosen model.
- Source semantics: deduplicate order totals; use completed orders for realized spending; cancelled baskets are cancellation behavior; provider debt overlaps and must not be summed as identified total debt; observed last-allocation time does not prove full/timely repayment.
- Do not use borrower IDs, row order, future observations, hidden-label/RNG reconstruction or external borrower labels as predictors.

## 5. Data, split, bank and checkpoint paths

### PC2 authoritative inputs

- Raw CSVs: `/mnt/home/logika-faris-exploration/exploration-12h-20261017/raw/`.
- Important files: `train_applications.csv`, `test_applications.csv`, `customer_profiles.csv`, `previous_applications.csv`, `repayment_history.csv`, `bureau_observations.csv`, `account_behavior.csv`, `ecommerce_transactions.csv`, `sample_submission(3).csv`.
- Original 70k features/labels and lock: `/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/input/data/train_features.parquet` and `input/BASELINE_LOCK.json`.
- Original fold roles: same `input/artifacts/fold0/indices.npz` through fold4; arrays `context`, `correction`, `train`, `validation`.
- Large immutable numerical bank: `/home/faris/faris/logika-meta-experiments/meta-20261016-v1/bank/`. `numeric.npy` is a memory-mapped float32 matrix, 100k rows × 2,329 numeric columns; categorical table has four variables. Read `manifest.json`, `metadata.parquet`, `categorical.parquet`; verify source/split hashes.
- Five context-local repaired feature tables: `/mnt/home/logika-faris-exploration/fold-local-priors-20261021/foldF/features.parquet`, with `complete.json`, `prior_audit.json`, source/vendor snapshots and temporary builder workspaces.
- Sanitized role packets: `/mnt/home/logika-faris-exploration/kumo-temperature-adaptation-20261021/full_data/foldF/`: `context.parquet` 50k labels; `query.parquet` 20k without labels; `correction_labels.npy` 6k; hashes in `manifest.json`. The packet `indices.npz` lacks a `train` key; derive/check the union against the original indices rather than assuming it exists.
- Kumo weights: `/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/assets/classifier.pt`, SHA **`cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613`**, 854,850,034 bytes. Model `nvidia/Kumo-Tabular`, large classifier, revision `v1.0.1`.
- Repaired CV Kumo contexts/caches: `/mnt/home/logika-faris-exploration/kumo-coverage-placement-20261021/conditions/foldF/control90/` (`context.parquet`, `fitted_context.pt`, `raw_predictions.npz`, `cache.json`, `complete.json`). Each 50k cache is roughly 6GB. Verify full hashes before reuse; do not casually duplicate all caches.
- Shared repaired spline CV models: `/mnt/home/logika-faris-exploration/kumo-coverage-placement-20261021/shared/foldF/spline.joblib`.
- Final full-data inputs/role IDs: `/mnt/home/logika-faris-exploration/submission-full-periodicity-20261021/input/artifacts/foldF/indices.npz`.

`F` means 0–4. The local `ranking-pipeline-20261020/input/` is a partial evidence mirror; notably do not assume it or PC2 `input/data/` contains a `test_features.parquet`. Newer builders obtain all 100k rows from raw applications/bank and reconstructed fold-local feature tables.

### Local original repo

`/home/parasite/Project/Competition/logika-2026/Logika/` contains the historical repo, including `scripts/build_features_v3.py`, `build_features_v4.py`, `build_features_v5.py`, `extend_features.py`, `reverse_synthetic_patterns.py`, model scripts and `artifacts/optimization/`.

The safe fitted estimators used from `reverse_synthetic_patterns.py` are its price, bureau and repayment latent estimators, invoked from isolated vendor snapshots. The file also contains other diagnostic/reconstruction work: **do not run its main module wholesale or revive RNG/hidden-label probes**.

## 6. V25 reconstruction and selected feature lineage

V25 did not replace the whole model. It added these six correction inputs to the pre-V25 12-field correction, leaving Kumo90 and spline77 unchanged:

```text
risk_conditional_negative_months
risk_conditional_nonnegative_months
aux_account_months
aux_orders90
aux_ecommerce_observed_days
aux_allocation_count
```

Definitions and validation:

- The first two count completed account months containing qualifying observed installment allocations and negative/nonnegative net cash flow. Installment eligibility uses due age >=15 days for this original block. Allocation presence is observation, not proof of settlement.
- `aux_account_months`: observed complete account months in the original recent-three-month window.
- `aux_orders90`: completed unique orders in the preceding 90 days.
- `aux_ecommerce_observed_days`: span back to first recorded order, an observation-coverage proxy, not verified enrollment duration.
- `aux_allocation_count`: eligible installments with a recorded last-payment allocation timestamp.
- [Coverage transfer](./a100/faris/coverage-transfer-20261021/REPORT.md) retains the raw rebuild, independent reference checks, source hashes and **65 correction fits** for the fixed opportunity extension plus attribution. These are multiple fitted contrasts, not 65 independent discoveries.

Maturity replacement:

- Replace `re_late_1_posterior` and `re_late_30_posterior` with separate Beta-binomial posterior means for maxDPD >=1 / >=30, **both measured on installments at least30 days old**.
- Fit Beta priors only on the explicit context IDs, not on test or evaluation applicants. Other repayment aliases and posterior-SD fields remain as in the selected baseline.
- Portable builder: [maturity_features.py](./a100/faris/feature-only-20261021/handoff/maturity_features.py); [factory checks](./a100/faris/feature-only-20261021/handoff/FACTORY_AUDIT.json). It checks all100k rows/folds for parity, serialization, row/ID invariance, future-event exclusion and noncontext perturbation invariance.
- Validation outputs: `repayment-severity-fe-20261021/foldF/features.parquet` (`matched_beta_late_1`, `matched_beta_late_30`) and `transformer.json`; mean-replaced head in `heads/matched_maturity/foldF/model.joblib`; corresponding spline in `maturity-placement-fe-20261021/foldF/spline.joblib`.

Shopping periodicity block:

```text
beh_spectral_spend_power_7d
beh_spectral_spend_power_14d
beh_spectral_spend_power_30d
beh_spectral_spend_periodicity_strength
beh_spectral_spend_dominant_cycle_days
beh_spectral_orders_power_7d
beh_spectral_orders_power_14d
beh_spectral_orders_power_30d
beh_spectral_orders_periodicity_strength
beh_spectral_orders_dominant_cycle_days
beh_spectral_window_supported
```

- Build 180 **full calendar days**, excluding the incomplete decision day. Deduplicate completed-order spending and counts. Require observed-span support plus >=8 completed orders for the ten spectral fields; unsupported values stay missing. The support flag is kept separately.
- Linear detrend, Hann taper, squared rFFT power, zero DC. Predeclared power bands cover 6–8, 12–16, 25–35 days; largest peak/period is selected among 3–90-day frequencies. Normalize by total nonzero-frequency power.
- Exact 100k-row feature table: PC2 `shopping-periodicity-fe-20261021/true_features.parquet`, SHA **`b0a92347fcd1328a7f4bbdb779a005223375f01d4e003ddf23e4445dc1ee1c39`**. Builder reproduced teammate V80 features to maximum error **8.33e-17**. Historical V80 checkpoints were replayed before the new placement comparison.
- The predeclared negative control shuffled the 180-day spending/count vectors **together within each customer**, seed20261127, preserving totals, support and daily distributions. This tests time order, not causal payroll cycles.
- Selected validation models: `shopping-periodicity-fe-20261021/models/true/foldF/{head.joblib,spline.joblib,validation.npz,boundary.json,complete.json}`. Ten selected fitted models are within the 20 real/control models. `OOF.npz`, `result.json`, `INDEPENDENT_AUDIT.json`, `FINAL_STATUS.json` retain the complete evidence.

## 7. Recent feature-only experiments: completed matched comparisons

All rows below have complete original-five-fold 70k OOF. Baseline is repaired V25 **KS .5302185367 / AUC .8470258500**. Fold-win counts come from the consolidated table **against that baseline**, not necessarily against the maturity candidate. Read each study's own report for other contrasts and uncertainty.

__FEATURE_TABLE__

Interpretation:

- **Highest pooled KS:** true periodicity + maturity, **.5309665315**. AUC decreases versus baseline, and only2/5 KS folds improve versus maturity. The simultaneous conditional intervals include no improvement. It was initially not promoted under joint KS/AUC criteria; the user later explicitly selected it for the final submission.
- **Earlier joint-metric point lead:** maturity in both head and spline, **.5306361042 / .8470560913**. Its conditional adjusted improvement intervals also include zero; it is not an independently confirmed gain.
- **Band-only spectral ablation:** KS fell to **.5295255960**, AUC rose to **.8471189857**. The two spectral studies show a metric trade-off. The branch was closed; no frequency/window/subset sweep was admitted.
- Account conditional-forecast and persistent-trait transforms both improved measurement arguments in some respects but failed to improve default KS/AUC jointly. Their source-time Gaussian models, parameters and negative results are retained. Do not reopen them merely to increase experiment count.
- Equal-loan/cluster credibility, expanded mature denominators, preserved missingness masks, source-support counts and Kumo-side coverage replacement also failed to beat the selected recipe. Complexity and a better raw measurement forecast are not sufficient evidence of better default ranking.

Target-free audits that excluded or qualified proposals:

| Local study | What it established / decision |
|---|---|
| `feature-budget-audit-20261021` | Some coverage fields duplicate existing features at model precision; correlated price inputs were explicitly identified before the failed replacement study. |
| `loan-cluster-audit-20261021` | Within-loan dependence supported testing a cluster model; the scored cluster replacement was still negative. |
| `bureau-reliability-fe-20261021` | Proposed persistent customer/provider error variance truncated to zero, with near-zero early/late contrast correlations; no default fit admitted. |
| `bureau-boundary-audit-20261021` | Common lower-censoring story unsupported; edge values were sparse. No default fit admitted. |
| `price-measurement-audit-20261021` | Very sparse price floor observations and negligible rounding-error scale did not justify a broad correction. No default fit admitted. |
| `account-measurement-audit-20261021` | Gaussian random-intercept/AR1 models improved held-out income/outflow measurement prediction; subsequent default-model replacements still failed joint improvement. |
| `bureau-noise-scale-audit-20261021` | No averaging-scale mismatch: 377,680 finite reports had one report per customer/application/provider. Separate shared-occasion diagnostic is described below. |

### Prepared but unexecuted bureau-occasion follow-up

`/home/parasite/Project/Competition/logika-2026/a100/faris/bureau-occasion-fe-20261021/` contains a draft representation, builder and matched-run code. It **was not deployed or trained** before the user selected the submission. There is no default-prediction score and no launch-ready completion claim.

The source audit found positive earliest/latest cross-provider score-change covariances, correlations only .026–.035. A separate 25k-fit /25k-held-out-customer measurement screen found primary hidden-score MSE difference **−.804886**, conditional interval **[−1.242550,−.367222]**, all three provider point differences negative. This is **score-measurement prediction**, not default KS/AUC. A shared economic/reporting component is not identified measurement noise or proven default signal.

Raw diagnostic sources: PC2 `/mnt/home/logika-faris-exploration/bureau-noise-scale-audit-20261021/`, files `RESULT.json`, `OCCASION_RESULT.json`, `SCREEN_RESULT.json`, `MEASUREMENT_SCREEN.md`, `measurement.py`. Fresh copies of the two later result files are in the handoff `evidence/` folder. If a future user authorizes another campaign, this is one specific unfinished hypothesis; finalize its static-posterior parity and role gates first. Do not claim it has improved default prediction.

## 8. Earlier model/fusion studies — avoid repeating blind search

These are background evidence, not permission to change the feature-only model. Confirm the **matched baseline within each study** before comparing deltas.

| Study | Retained finding / important artifact |
|---|---|
| `horizontal-20261019` | Combining the earlier champion with the teammate's stronger Kumo recipe did not beat the teammate. `results/REPORT.md`, `results/result.json`, imported original repo evidence. |
| `teammate-approach-20261020` | 80 correction fits: credit-agreement, consumption-obligation, measurement-reliability blocks and an inner selector. All failed joint KS/AUC improvement. `REPORT.md`, `result.json`, `feature-study/REPORT.md`. |
| `teammate-approach-20261020/feature-study` | 55 heads disentangled V07/V101 regularization, coverage and feature changes. Apparent published gains were confounded or setting-dependent. Do not attribute their whole gain to a feature name. |
| `measurement-risk-20261020` | Ten actuarial/measurement-uncertainty GLMs; integrated/plugin replacements and small additions did not beat the matched baseline. `completed/REPORT.md`. |
| `ranking-pipeline-20261020` | Fixed ranking-loss intervention did not justify replacing incumbent. Also holds the shared pinned environment, original input lock, feature data and Kumo checkpoint. |
| `nested-calibration-20261020` | Nested correction improved AUC but lowered KS; incumbent retained. `results/REPORT.md`. |
| `ri[REDACTED_CREDENTIAL]` | **Fresh PC2 inspection on23October: complete**, unlike stale local `state.json` saying reporting. Teacher/disagreement features: signal KS **.5275823123 / AUC .8469021269**, baseline **.5292498734 / .8472013286**. Negative. Terminal result and independent audit copied into this handoff. |
| `nested-correction-20261021` | **23 October PC2 inspection: complete; independent audit covers30 jobs**. Crossfit56000 **.5287816648 / .8474621497**; heldout6000 **.5297495561 / .8468937319**; fixed-half **.5300270558 / .8473457713**; original control **.5302185367 / .8470258500**. None beats control KS. Local older `COLLECTED_CONTENT_AUDIT.json` verifies only seven collected contexts, not the later whole run; use fresh terminal/independent audit in handoff evidence and PC2. |
| `kumo-coverage-placement-20261021` | Repaired90 baseline plus coverage96 and shuffled96. Coverage in the main context did not improve KS; control90 supplies reusable contexts/caches. |
| `kumo-temperature-adaptation-20261021` | Gradient/temperature adaptation result **.52733117 / .84525661**, weaker than fixed control. Do not call ordinary Kumo `.fit()` gradient fine-tuning. |
| `kumo-full-context-20261021` | 56k context alternative **.52980209 / .84689241**; did not beat its matched50k control. Final user-requested64k submission refit is not a newly validated context-search winner. |
| `kumo-supervised-view-20261021` | Supervised90 selection **.52808253 / .84768378**: AUC up, KS down. Fixed addition also below control KS. |
| `catboost-convergence-20261021` | Full native CatBoost **.52042860 / .84309995**, compact **.52198804 / .84334697**, weaker standalone. Fixed additions did not solve the KS gap. |
| `tabicl-compact-20261021` | PC GPU memory/cache issues preserved. Read incident files; do not count incomplete jobs as negative completed recipes. |
| `tabicl-compact-a100-20261021` + `heldout-margin-fusion-20261021` | A100 TabICL classifier **.52510532 / .84524010**; fixed20% addition **.53006128 / .84692974**; learned margin fusion **.53008935 / .84701535**. No KS improvement over repaired baseline. |
| `tabicl-conditional-mean-20261021` + `heldout-class-margin-20261021` | Conditional-mean regressor was weaker; class-margin-only tiny KS change **.53023495 / .84701167**, with lower AUC. Preserve exact method distinctions. |
| `kumo-quantile-mean-20261021` | Frozen regression-prior/999-quantile Bernoulli-mean hypothesis on node380; result **.52932293 / .84671393**, fixed addition **.5300864 / .8470242**. Not a new winner. Local `plan.json` is an early prepared snapshot; terminal evidence lives on node380. |

Some other studies (`bureau-state-20261020`, `ri[REDACTED_CREDENTIAL]`, `calibration-capacity-20261021`) are retained as their own research/diagnostic branches. Use the complete path index and their actual result/gate files; do not infer completion from a directory or README alone.

## 9. Older broad campaigns, original champions and submissions

### Broad meta experiments and wave2

- Local implementation: `/home/parasite/Project/Competition/logika-2026/a100/faris/logika-meta-experiments/`.
- Campaign1 on PC1: `/home/agung/faris/logika-meta-experiments/meta-20261016-v1/`; wave2 sibling `meta-20261016-v1-wave2/`.
- Campaign1/wave2 on PC2: `/home/faris/faris/logika-meta-experiments/meta-20261016-v1/` and `meta-20261016-v1-wave2/`.
- PC3 also used its own independent shard; locate the exact archived root from manifests rather than guessing a home symlink.
- Key artifacts: `campaign.json`, `catalogue.json`, `history_inventory.json`, `ledger.sqlite`, `runs/<id>/`, `reports/final_report.md`, `reports/feature_effects.csv`, `reports/confirmation.json`, `reports/untested_recipes.json`, `reports/a100_handoff.json`.
- Implementations: `metaexp.pipeline.Preprocessor`, `scheduler_parallel.py`, `autonomous_shards.py`, `node_executor.py`, `rpc_parallel.py`. See `RUNNING.md` and `extensions/wave2/README.md` for historical changes, not current-live-worker claims.
- Original catalogue:1,024 representations /4,096 pipelines; wave2:4,096 representations /16,384 pipelines. **Registration is not completion.** Folds and fixed blends are counted separately.
- The legacy selector named `gain` used default LightGBM **split-count** importance. Later campaigns distinguished true gain importance. Keep this mismatch explicit for reproducibility.

### Depth campaign (06:00 WIB cutoff,17October)

- Local: `/home/parasite/Project/Competition/logika-2026/a100/faris/depth-20261017/`.
- PC1: `/home/agung/faris/depth-20261017/{campaign,campaign-amd}/`.
- PC2: `/home/faris/faris/depth-20261017/campaign-parallel/`.
- PC3: `/mnt/home/faris/faris/depth-20261017/campaign-parallel/`.
- Node380: `/home/taqiyudinmiftah/faris/depth-20261017/campaign-fm2/`.
- Evidence: `state.json`, `heartbeat.json`, `owner.json`, `search.sqlite3`, `runs/*/attempt_*/{stdout.log,stderr.log,spec.json,models.json,validation.npz}`, `reports/{report.md,summary.json,trials.csv,trials.jsonl}`.
- Startup `campaign/` and repaired `campaign-parallel/` / `campaign-fm2/` are not automatically independent new logical recipes. GPU queue fixes, notebook Agg fixes and OOM incidents are retained.

### Exploration campaign anchored to0.56188 (17October)

- Local: `/home/parasite/Project/Competition/logika-2026/a100/faris/exploration-12h-20261017/`.
- PC1: `/home/agung/faris/exploration-12h-20261017/`.
- PC2: `/mnt/home/logika-faris-exploration/exploration-12h-20261017/`.
- PC3: `/mnt/home/faris/faris/exploration-12h-20261017/`.
- Node380: `/home/taqiyudinmiftah/faris/exploration-12h-20261017/`.
- 4,096 training configurations were a registration ceiling. Read actual reports, scores, failures and coverage. Actuarial branch included binomial logit/cloglog GLMs, cross-fitted WOE and Beta-binomial credibility; unavailable severity/survival outcomes were excluded, not fabricated.
- TabM, residual MLP, FT-style transformer, history encoders and a **TabR-inspired** retrieval MLP were explored. The local retrieval model is not an exact official TabR reproduction. TabPFN2.5 was excluded where access/terms were unresolved.
- Exact reproduction of the old champion's PCA/SVD components depended on execution-host BLAS behavior (PC1 Cooperlake versus PC2 Zen kernels). Preserve fitted transforms, host/backend identity and source hashes; moving a reduction across hosts can change predictions despite identical packages.
- Original deadline23:55:35 WIB on17October is expired. Later experiments did not extend that same campaign by silently resetting it.

### Original Kumo Large node380 baseline

- Local: `/home/parasite/Project/Competition/logika-2026/a100/faris/kumo-large-node380-20261016/`.
- Remote: `/home/taqiyudinmiftah/faris/kumo-large-node380-20261016/`.
- `artifacts/node380/full/summary.json`, saved full OOF predictions, `submissions_held/submission_kumo_large_logika_v3.csv`, `launcher.log`.
- Independently recomputed historical OOF KS **.5127016592**, a different10-fold/16,384-context/two-estimator protocol. User-reported public score **.55818**. Not comparable to later70k/5-fold context50k scores without matched reruns.

### Submission lineage

| Artifact or recipe | Recorded result | Local location / caution |
|---|---|---|
| Historical Logika v3 | User-reported public **.55845** | `Logika/submission_v3.csv`, `Logika/artifacts/optimization/`; historical OOF .51901 was under an older protocol. |
| Historical v5 | User-reported public **.55680** | `Logika/submission_v5.csv`; better local historical OOF did not imply better public score. |
| Five-model trimmed champion | User-reported public **.56188**;56k dev KS **.525771439**, reused confirmation **.530496149** | `a100/faris/submission-research-20261017/artifacts/submission.csv`, SHA **`be059078ade95bc6758522e8d72849e3f793e57380e2c083e05c77c64335e192`**. Remote PC2 `/home/faris/faris/submission-research-20261017/`. |
| Seed/research pass | No dev gain; champion retained | `a100/faris/submission-pass2-20261017/result.json`; remote `/home/faris/faris/submission-pass2-20261017/`. |
| EBM/account elastic-net/TabM/two nested stacks logit ensemble | User-reported public **.55353**, below .56188 | `a100/faris/submission-final-20261018/`; artifact SHA **`e809cc0100b39ce2bd6ddc15ac7f28fc30f840bf483c3e8777af2e37e7f478c7`**. Its good reused validation did not transfer. |
| Champion-anchored95:5 elastic addition | Exploratory secondary benchmark, no public score established here | `a100/faris/submission-robust-20261018/REPORT.md`; SHA **`1ba5b8677549f5d05fdda6d63fa5f978e731247275a74ff91f3bdd39ac068036`**. It was the immediate previous root CSV, **not** the .55353 replacement. |
| Teammate V25 | Best reported public **.56822**; historical OOF **.5302846600/.8470292789** | `/home/parasite/Downloads/submission_V25_goal054_offset_v7.csv`, SHA **`e7f0748c6c8ded776f0a6a03bdebcd0787cf6c0057d0b078c31196fb34f58efe`**. Refreshed teammate record cites Kaggle submission ref57444414 and a verified receipt/hash. |
| Teammate V40 | User-reported public **.56464** | Coverage-transfer evidence/runs/V40 plus live teammate master. |
| Current periodicity+maturity full refit | Selected research OOF KS **.5309665315**; refit/public score unknown | Root `submission.csv`, SHA **`922788d13cd8ec53b7cbd732496a9200150a499fd4363455414e2cc6230a6b06`**. Do not confuse its filename with old artifacts also named `submission.csv`. |

The old .56188 champion has three LightGBMs and two regularized logistic regressions over differing feature views; per applicant it removes the highest/lowest probability and averages the middle three. The current delivered Kumo+correction+spline graph is different. “Expand horizontally” motivated earlier research, but adding weak models or more columns repeatedly did not automatically improve scores.

## 10. Teammate sources and sessions

- Repository reference: `https://github.com/TaqiyudinMiftah/Logika`. The GitHub contents were not refetched in this handoff; use local imported snapshots or inspect its actual current revision before assuming it matches an experiment.
- Main current node380 work: `/home/taqiyudinmiftah/adit/Logika/insight/goal054_offset_baseline_20261019/v7/`.
- Master logs: `reports/EXPERIMENT_MASTER_LOG.{md,csv,json}`.
- Important subtrees: `scripts/`, `runs/V25/`, `runs/V40/`, `runs/V80/`, `behavior/data/raw_features.parquet`, `behavior/reports/FEATURE_CATALOG.csv`.
- Older parallel repo `/home/taqiyudinmiftah/Logika/` also exists; do not confuse it with `adit/Logika/` when resolving a run's provenance.
- Fresh source metadata from23October: [NODE380_LIVE_PATHS.json](./a100/faris/handoff-20261023/NODE380_LIVE_PATHS.json). Source master timestamp **__MASTER_TIME__**, SHA **`__MASTER_SHA__`**. This is a read snapshot; the teammate may continue updating it.
- Local imported approach: `/home/parasite/Project/Competition/logika-2026/a100/faris/teammate-approach-20261020/` with `live_master_log.json`, `FEATURE_RESEARCH.md`, `feature_evidence/scripts/{features.py,lifecycle_features.py}`, `feature-study/`, extracted session material and frozen run evidence.
- A previous refreshed118-record snapshot: `/home/parasite/Project/Competition/logika-2026/a100/faris/feature-only-20261021/friend-refresh-20261021/EXPERIMENT_MASTER_LOG.json`. It is older than the live23October log.
- Correct high-scoring session: `/home/parasite/Downloads/codex-session-01a133cc-0500-7a21-bcb6-6fdf628355f0.md` (~61MB). Use targeted search; do not dump it wholesale into context.
- Feature-engineering teammate session: `/home/parasite/Downloads/codex-session-logika.md` (~13MB).
- The earlier shared session `codex-session-01a15259-a648-7d63-bf1b-d334a6ff57e5.md` was explicitly declared unrelated by the user. Do not treat it as the source of the best recipe.

## 11. Historical inventory and honest experiment counts

Root: `/home/parasite/Project/Competition/logika-2026/a100/faris/non-dominated-research-20261021/`.

- `inventory/reconciliation.json`, `retained_records.json.gz`, `parameter_duplicates.json`, `metadata_dispositions.json`, `unresolved_sources.json`, `conflicts.json`, `REPORT.md`.
- Host metadata snapshots: `snapshots/pc2/`, `snapshots/pc3/`, `snapshots/node380/`. These are snapshots, not live node states.
- Independent metric checks: `validation/prediction_audit.jsonl`, `prediction_audit_status.json`, `teammate_audit.jsonl`, `teammate_audit_status.json`.
- Saved state reported **36,136 prediction files independently checked**, **4,096 fixed blends replayed**, **118 teammate run records verified**. These measure different kinds of evidence.
- The retained reconciliation snapshot has **43,868 recipe/assessment records**, merged from **66,158 source references**, **22,290 repeated references merged**, and **206 unresolved historical metadata sources**. It contains different protocols, fold evidence, reused references, failed/partial work and ensemble assessments. It is **not43,868 independent completed model designs** and not directly the user's colloquial “12k experiments.”
- The recent feature-only summary has **18 complete recipe/control comparisons**, not18 independent hypotheses, and is a much smaller scientifically matched subset of the broader archive.
- PC1 was unavailable to parts of that historical reconciliation. The new route is now verified, but this handoff did not silently redo its whole archive audit.

## 12. What the evidence supports and what remains unfinished

Supported operational facts:

- Baseline/checkpoint/CSV reproduction and many tied-KS/model-replay gates passed.
- V25's actual six-field correction change was reconstructed; its effect was separated from a whole-model redesign.
- Several temporal/measurement hypotheses were tested under fixed predictive settings, including negative/null controls.
- The chosen periodicity recipe has the highest pooled KS among the recent matched feature-only results, despite a slight AUC decline and inconsistent fold gains.
- The requested full-data submission was actually trained and delivered. All models and previous submissions remain preserved.

Not established:

- No verified0.54 pooled70kOOF result.
- No known Kaggle result for the current final CSV in this thread.
- No untouched private validation or guarantee that a local point improvement transfers to public/private leaderboards.
- Conditional bootstrap intervals do not erase historical/adaptive label reuse or account for all refit uncertainty.
- Better likelihood/AUC/measurement forecasts or beating a shuffled control does not prove an improvement over the stronger matched KS baseline.
- The prepared bureau-occasion default comparison is unexecuted; it is not a failed or successful default model.

If asked to continue research later, first read the exact current request, obtain any new compute/time scope from that request, inspect live paths/states and prevent duplicate jobs. Use a small explicit hypothesis with a matched baseline, train-only transforms, full aligned OOF and retained evidence. Do not manufacture another thousands-of-recipes sweep or repeat closed branches without a new evidence-backed mechanism. User has repeatedly requested deeper feature reasoning, honest completion and low token waste.

## 13. Read-only recovery and verification commands

```sh
# Local CSV integrity; no training or upload.
sha256sum /home/parasite/Project/Competition/logika-2026/submission.csv

# Final full-data run is terminal; inspect receipts.
ssh faris@labkc-comm1 'cat /mnt/home/logika-faris-exploration/submission-full-periodicity-20261021/STATE.json'
ssh faris@labkc-comm1 'cat /mnt/home/logika-faris-exploration/submission-full-periodicity-20261021/FINAL_STATUS.json'

# Observe a specific known unit without modifying it.
ssh faris@labkc-comm1 'systemctl --user show logika-full-periodicity-refit-20261021.service -p ActiveState -p MainPID'

# Node380, existing control session.
python3 /home/parasite/Project/Competition/logika-2026/a100/profile.py node380 term 'ls /home/taqiyudinmiftah/adit/Logika/insight/goal054_offset_baseline_20261019/v7/reports'
```

Do not re-run `controller.py`, `prepare.py`, `train.py`, `build_submission.py`, `launch`, `resume` or a replay that retrains merely to inspect a result. Many scripts deliberately reject existing output directories or expired deadlines. First distinguish inference-only replay, metric verification and new model fitting. Old PIDs can be reused; check command line and ownership before any campaign-only stop.

The local installer `a100/faris/submission-full-periodicity-20261021/install_verified.py` can collect a completed final run and replace the root CSV, preserving a backup; **it is a mutation**, not a read-only status command. The handoff did not rerun it because delivery is already verified.

## 14. Complete local experiment-directory index

All directory links below are local absolute paths. Remote locations identify the archive to inspect when the local mirror is partial. Existence checks and exact scope are in `EXPERIMENT_PATHS.json`, `PC2_LIVE_PATHS.json`, `ACCESS_VERIFICATION.json`, `NODE380_LIVE_PATHS.json`. A verified directory is not proof that every planned run finished.

__PATH_TABLE__

## 15. Handoff verification evidence

- [ACCESS_VERIFICATION.json](./a100/faris/handoff-20261023/ACCESS_VERIFICATION.json): new PC1 route/password login, PC3 hop and node380 access checks.
- [PC2_LIVE_PATHS.json](./a100/faris/handoff-20261023/PC2_LIVE_PATHS.json): dated primary archive existence checks; 47 study directories present at the standard root, plus verified older roots.
- [NODE380_LIVE_PATHS.json](./a100/faris/handoff-20261023/NODE380_LIVE_PATHS.json): current teammate master timestamp/hash/public receipt metadata and actual node380 study directories.
- `a100/faris/handoff-20261023/evidence/`: freshly collected terminal result/independent-audit files for risk representation and nested correction, plus the bureau-occasion measurement diagnostics. These explain where old local mirrors lag PC2.
- [EXPERIMENT_PATHS.json](./a100/faris/handoff-20261023/EXPERIMENT_PATHS.json) / [CSV](./a100/faris/handoff-20261023/EXPERIMENT_PATHS.csv): complete local directory index and primary remote mappings.

No credentials are embedded in the handoff or inventory. The new password is retained separately in the private access file described in section 2. Existing node profiles keep their existing credentials. Do not commit or display them.
'''

doc=doc.replace('__FEATURE_TABLE__',feature_table).replace('__PATH_TABLE__',path_table).replace('__FREE_GIB__',f"{live['disk_free_GiB']:.2f}").replace('__MASTER_TIME__',node['master_updated_UTC']).replace('__MASTER_SHA__',node['master_sha256'])
# Absolute local links stay useful when the document is opened from another folder.
doc=doc.replace('(./a100/',f'({PROJECT}/a100/')
cleanup_report=BASE/'model-cleanup-20261025/REPORT.json'
if cleanup_report.exists()and read(cleanup_report).get('success'):
 cleanup=read(cleanup_report)
 notice=f'''## Update — 25 October 2026: PC2 model deletion

The user subsequently requested deletion of **every model under `{PC2}`**.
That cleanup is verified complete: **{cleanup['deleted_model_files']:,} model/transform/checkpoint files**,
**{cleanup['deleted_model_links']:,} model aliases**, and **{cleanup['removed_archive_model_members']:,} model members inside delivery archives** removed.
Outputs, predictions, submissions, code, configuration records and datasets remain.
Readable embedded settings exports are on PC2 in `{PC2}/model-cleanup-20261025/parameters/`.
See {link(cleanup_report,'cleanup receipt')} and {link(cleanup_report.parent/'README.md','scope and verification methods')}.

**This update overrides every earlier claim below that PC2 models or checkpoints are retained.**
Old paths and model hashes are historical provenance, not evidence of currently available weights.
Running those pipelines now requires retraining or separately retained weights.
The local laptop models and other machines were outside this cleanup scope.
Historical integrity manifests are preserved; mixed archives have new cleanup hash receipts.

Current verified PC2 route: `ssh faris@100.68.29.126` (port 22), host `labkc-comm1`.
Port 2001 refused the connection. Reach Agung from PC2 as described below.
The old Python 3.11 installation outside the archive disappeared during cleanup;
the historical virtual-environment interpreter links are broken. The cleanup
used system Python and did not repair or delete that installation.

'''
 first,rest=doc.split('\n',1);doc=first+'\n\n'+notice+rest
 doc=doc.replace('ssh faris@labkc-comm1','ssh faris@100.68.29.126')
migration_report=BASE/'home-migration-20261025/REPORT.json'
if migration_report.exists()and read(migration_report).get('success'):
 migration=read(migration_report)
 notice=f'''## Update — 25 October 2026: new PC2 home and repaired dependencies

Fresh SSH logins now use **`/mnt/home/faris`** as both `$HOME` and the initial directory.
The Logika workspace hub is **`/mnt/home/faris/Logika`**; the experiment archive is
**`/mnt/home/faris/logika-faris-exploration`**. Its former `/mnt/home/logika-faris-exploration`
path remains a compatibility symlink. `/home/faris` is also a compatibility symlink
to the new home: **keep that link; do not recursively delete its target**.
The original home data remains backed up at `/home/faris.pre-migration-20261025`.
It was not deleted. The live SSHFS storage mount is under the new arena directory.

The Python path failure described in the earlier cleanup update is now repaired.
All **{len(migration['environment_checks'])} old/new environment invocation checks passed**.
The two license-plate services and 11 affected CVAT containers were migrated with
explicit user permission; their APIs returned HTTP 200. Existing CVAT images were
preserved, and database/Redis and other unaffected containers retained their identities.
No reboot or poweroff occurred. See {link(migration_report.parent/'REPORT.md','home migration report')}
and {link(migration_report,'live verification receipt')}.

'''
 first,rest=doc.split('\n',1);doc=first+'\n\n'+notice+rest
target=PROJECT/'HANDOFF_LOGIKA_20261023.md';target.write_text(doc)
print(json.dumps(dict(markdown=str(target),study_directories=len(index),feature_comparisons=len(feature['rows']),words=len(doc.split())),indent=2))
