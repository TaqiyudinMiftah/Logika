# Depth campaign — 17 October 2026

All execution has an absolute 06:00 WIB deadline (1792191600). Individual schedulers stop search earlier to preserve reporting time. Resume preserves each immutable deadline. The earlier 12-hour campaign remains independent.

The supplied 56,000 development rows use the same frozen three folds. Confirmation arrays are physically excluded from the new bank view. Historical and adaptive reuse of development labels is disclosed; these results are exploratory. No test predictions or submissions are generated.

## Hosts and queues

- PC1: `agung@100.89.211.119:~/faris/depth-20261017/campaign` (16 CPU workers) and `campaign-amd` (one RX6600LE OpenCL worker).
- PC2: `faris@labkc-comm1:~/faris/depth-20261017/campaign-parallel` (up to 12 CPU units, both RTX3060s).
- PC3: access through PC2 only; `/mnt/home/faris/faris/depth-20261017/campaign-parallel` (up to 12 CPU units, both RTX3060s).
- Node380: `/home/taqiyudinmiftah/faris/depth-20261017/campaign-fm2` (one exclusive A100 Kumo Large trial; context/estimator search and bounded episodic gradient adaptation).

PC2/3 startup `campaign/` archives are preserved after fixing GPU-queue blocking in a new sealed `campaign-parallel/`; never sum these as independent logical configurations without deduplication. The original adapters and saved artifacts remain untouched. All generated models/checkpoints are retained. Disk pressure blocks admissions rather than deleting weights.

## Explicit operations from the repository root

```sh
python a100/faris/depth-20261017/control.py status all
python a100/faris/depth-20261017/control.py report all
python a100/faris/depth-20261017/control.py stop all
python a100/faris/depth-20261017/control.py resume pc2
```

These commands make one requested operation; there is no AI monitor or recurring automation. A managed Jupyter provider can still terminate node380 independently.

Each remote campaign contains `state.json`, `heartbeat.json`, `owner.json`, `search.sqlite3`, immutable configuration/source/splits, `runs/*/attempt_*/{stdout.log,stderr.log,spec.json,models.json,validation.npz}`, retained fold models, and `reports/{report.md,summary.json,trials.csv,trials.jsonl}` after automatic finalization. PC controller logs are in `journalctl --user -u oneshot-depth-*`; node380 has `fm2-controller.log`. Status does not recompute metrics; reports recheck artifact hashes. Each runtime's `verify` command recomputes one job's metric without training.

Synthetic adapter tests cover 3-fold OOF completeness, model retention, independent tied KS, train-only imputation and inner cross-fit encoding. Runtime regression tests cover GPU queues not blocking CPU jobs, GPU exclusivity, failure/timeout retention, disk admission, tampering, and stop/resume deadline preservation. GPU tests distinguish simulated NVIDIA scheduling from real Kumo GPU backward/inference and real AMD OpenCL training.

Research, selected historical recipes, hypothesis links, applicability exclusions and EDA are in `evidence/`. Dependencies are pinned from each execution environment inside its sealed source snapshot. NVIDIA weights are the existing staged `nvidia/Kumo-Tabular@v1.0.1`, SDK commit `329a707bbcf9ca147aa45a63fef8b2029f4ee37b`. Episodic AdamW tuning is an engineering experiment, not a claim that `fit()` performs gradient updates.

The dedicated AMD queue overrides the base CPU campaign applicability note that excluded AMD from NVIDIA-only admission. Its adapter explicitly passes LightGBM OpenCL platform/device settings, and startup logs identify gfx1030. AMD allocation is one dedicated process; the toolkit NVIDIA GPU-allocation counter does not include AMD time. The later user instruction also explicitly authorizes node380 foundation models, superseding the earlier no-FM scope for this new extension.

FM revision 1 completed one full-CV baseline, then stopped after adaptation OOM and inherited notebook-backend failures. It remains archived as campaign-fm. Revision 2 (campaign-fm2) fixes Agg backend and uses training-only episodes of 128 context/32 query rows; it starts a new immutable bundle within the same absolute 06:00 cutoff. These revisions must not be pooled as identical recipes.
