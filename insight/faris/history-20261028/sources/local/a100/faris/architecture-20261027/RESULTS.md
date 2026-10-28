# Frozen-feature architecture campaign: completed without a supported gain

All 160 planned fits completed with zero failures or unresolved jobs. Features, rows and seeds remained fixed. The program selected its recipe before terminal outer scoring; the independent terminal audit passed. No full-data refit or Kaggle submission was made.

| Evaluation | Pooled OOF KS | Change versus baseline |
|---|---:|---:|
| Matched original baseline | 0.5306361042 | — |
| Frozen selected architecture blend | 0.5291772104 | -0.0014588939 |
| Best paired design, diagnostic only (resnet_master_bce) | 0.5249607251 | -0.0056753792 |

The selected blend improved 1/5 fold point estimates. Its paired class-stratified 2,000-replicate 95% interval for the KS change was [-0.0034979204, +0.0007581955]. Both fixed-recipe seed changes were negative: 2026 -0.0022356833, 7731 -0.0013914160. Target 0.54 was not reached. The interval also does not establish a definite population loss.

## Architectural and input-view evidence

Each row below averages the two fixed seeds before the unchanged correction head and maturity mixture. These family diagnostics were predeclared; the highest outer score was not used to replace the primary recipe. No paired design or single configuration exceeded the matched baseline.

| Design / objective | Original 90 | Frozen master 116 | Master minus original |
|---|---:|---:|---:|
| mlp_control / bce | 0.52044041 | 0.52316211 | +0.00272170 |
| resnet / bce | 0.52297582 | 0.52496073 | +0.00198491 |
| resnet / bce_pair | 0.52488615 | 0.52475654 | -0.00012961 |
| dcn_stacked / bce | 0.51990303 | 0.52279336 | +0.00289033 |
| dcn_parallel / bce | 0.52086185 | 0.52422667 | +0.00336482 |
| tabm / bce | 0.52280277 | 0.52238467 | -0.00041811 |
| tabm / bce_pair | 0.52396621 | 0.52317892 | -0.00078729 |
| ft_transformer / bce | 0.52341465 | 0.52219121 | -0.00122344 |

The larger frozen input set improved the point estimate in 4/8 matched design comparisons: MLP, BCE ResNet and both DCN layouts. It reduced it in the other four. Pairwise ranking improved the point estimate in 3/4 matched objective comparisons, but none reached the baseline. These are conditional development observations and do not establish that more features or pairwise training help universally.

The correction-only selection chose baseline-blend weights [0.05, 0.70, 1.00, 0.05, 0.05] across folds. Its terminal deterioration illustrates that correction-set selection did not reliably identify a beneficial combination here. We did not alter those choices after reading terminal results.

## Verification and limitations

The independent NumPy CDF/rank implementation reproduced all 52 KS/AUC summaries with maximum error 3.33e-16; correction-selection replay differed by at most 2.22e-16. All 260 downstream correction fits (52 variants × five folds) reproduced the saved predictions exactly, maximum absolute error 0.0. Baseline replay error was 1.11e-16. The independent 2,000-draw bootstrap reproduced the recorded interval.

The audit also verified all five immutable feature/role/baseline packets, all 160 prediction digests, exact 70,000-row outer coverage, configuration and source seals, context-only training boundaries, stopping/refit epochs, seed identities, clock integrity and absence of serialized fitted weights. Runtime failures were zero. Both nodes have no remaining campaign units after audit. Remote original outputs and the local mirror are checked in `REMOTE_RECEIPTS.json`.

These outer rows were already exposed development data. There is no untouched confirmation, and the bootstrap is conditional on this reused development protocol. The fixed correction head can favor its original baseline family. Two seeds and one declared parameter schedule per design do not exhaust these architectures. This result is about the tested recipes, not a proof that neural architectural modelling cannot improve Logika.

## Resources and preserved artifacts

The training/evaluation program ran from 00:11:58 to 00:47:57 UTC on 27 October (07:11:58–07:47:57 WIB), 35.98 minutes within the six-hour ceiling. Qualification, pilot and evaluation were included. Independent verification followed within the same original ceiling; the clock was never reset. The six hours was a ceiling, not a minimum runtime.

The 160 main fits comprised 129 CPU fits and 31 CUDA fits, only on nodes 2001 and 2003. Summed per-fit wall durations were 4.284 hours because fits ran in parallel. The longest fit took 478.6 seconds. Maximum recorded fit RSS was 1.668 GiB; peak allocated CUDA memory was 0.532 GiB. No node2002/A100 work or interference with unrelated workloads was used.

The archive preserves configs, seeds, training/preprocessing recipes, sources, environment identities, logs, compact predictions, frozen selection, terminal scores and audits. It contains no serialized fitted neural weights or KV caches. `audit/terminal_audit.py` can repeat the checks; `CONFIGS.json`, `CLOCK.json` and `GATE.json` identify the actual run. Do not use `launch.py --hours` again. No further training, refit or submission is queued. The prior incumbent is retained.
