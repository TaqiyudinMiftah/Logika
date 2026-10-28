# Feature-subspace MoE: completed, no supported gain

All260 predeclared expert fits completed:80 Kumo,90 LightGBM,90 CatBoost.
There were no failed or unresolved expert fits. Three superseded lab Kumo
attempts remain separately preserved; they were never mixed into the consistent
80-fit A100 cohort. Feature selection, nested anchor fits, calibration and gate
fits are additional pipeline fits, not extra independent replications.

| Comparison | Pooled local KS | Delta from matched anchor |
|---|---:|---:|
| Matched maturity-adjusted anchor |0.5306361042|0|
| Frozen primary: expanded Kumo, conditional gate |0.5295070116|-0.0011290926|
| Expanded Kumo, fixed blend |0.5293967574|-0.0012393469|
| Existing Kumo, conditional gate |0.5292563835|-0.0013797208|
| Best secondary point: wide LightGBM, conditional gate |0.5306791136|+0.0000430094|

The primary lost in both seeds; only2of5 fold deltas were positive. Its paired
conditional95percent interval for the KS difference was[-0.0023074610,+0.0008038809].
The registered acceptance rule failed and the0.54 target was not reached.
The best secondary's tiny point gain is below the registered0.0005 minimum,
with only2of5 positive folds and1of2 positive seeds. It cannot replace the frozen
primary after outer scoring and is not a supported improvement.

Matched routing improvement was+0.0001102543, interval[-0.0002269021,+0.0002820409].
Matched expansion improvement was+0.0002506282, interval[-0.0014230810,+0.0013128798].
Conditional routing versus the separately predeclared learned-global-weight gate
was+0.0001433160, interval[-0.0002255815,+0.0003037429]. None established a reliable
benefit. These findings concern this four-domain recipe and regularized small gate;
they do not supply a general bound on every possible feature expansion or MoE.

The2090-definition source-backed bank was selected within each50000-row context.
Each expanded Kumo expert received at most99 predictors, including20 shared core
features. Expanded domain unions had281–286 features across folds. The6000 meta
rows and14000 validation rows were excluded from expert training and supervised
feature selection. Anchor heads/splines and domain calibration supplied held-out
meta inputs to the gate. Five original folds and seeds2026/7731 were retained.
Model seeds are the base seed plus fold index. All70000 cases had prior development
exposure: no untouched statistical audit exists. The intervals condition on the
fitted predictions and omit retraining and historical research-selection uncertainty.

Independent artifact readback passed: all260 source/seed/row boundaries and
prediction hashes, every cohort's SciPy KS recomputation, matched baseline replay,
and the primary's2000-replicate conditional interval. A second local collection
check verified all260 compact outputs and the3 retired lab attempts. There are1262
private final artifacts totaling171349676bytes, with a private full hash manifest.
This establishes artifact consistency, not independent statistical confirmation.

The user-authorized12hour clock started28October11:28:37 UTC. The A100 was released
at14:46:27 UTC(21:46:27 WIB), and scientific evaluation finished14:47:15 UTC.
The paid runtime existed for about2hours15minutes. Rounded account readings show
about15.18 compute units consumed; the live meter reported6.77CU/hour while active
and0CU/hour with zero assignments afterward. The original total deadline remains
28October23:28:37 UTC. Remaining ceiling is not a reason to reopen this completed
comparison or keep a paid VM allocated.

No full-data refit or Kaggle submission was justified or performed. The retained
public incumbent remains the separately identified V25 artifact, recorded KS0.56822;
this completion did not perform a new Kaggle leaderboard lookup. Parameters,
source, seeds, feature identities, preprocessing recipes, environment identities,
compact predictions, setup/transfer diagnostics and hashes are saved. No new fitted
weights or KV caches were serialized. The pretrained public Kumo checkpoint hash
remained cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613.

Public evidence is in this run directory; full20-cohort results are in
evaluation/RESULTS.md and evaluation/RESULT.json. Private case-level artifacts are
under FINAL_EVIDENCE locally and the canonical lab home. Do not rerun launch.py,
compose.py or any completed unit in this sealed directory. Any subsequent study
must retain this negative result, identify its reused-development exposure, and
have its own explicit protocol and authorized budget before training.
