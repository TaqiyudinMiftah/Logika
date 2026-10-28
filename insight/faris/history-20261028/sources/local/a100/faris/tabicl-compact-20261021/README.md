# Compact TabICLv2 controlled comparison

One frozen eight-estimator backbone recipe, five original outer folds, each with50,000 context labels,6,000 correction-head labels and14,000 evaluation rows. The same repaired compact90 input view, V25 correction18 features and independently replayed spline77 model are used. Primary: TabICL versus repaired Kumo90. Secondary: fixed80% Kumo /20% TabICL. No weight search or partial-fold scoring. No competition submission is generated.

PC2 service: `logika-tabicl-compact-repr-20261021.service`. Two independent GPU slots execute allfivefolds. The original deadline is2026-10-21 19:31:11 WIB with athree-hour finalization reserve. All fitted transformers, checkpoints and full-query replay evidence are retained remotely under `/mnt/home/logika-faris-exploration/tabicl-compact-20261021`.

The initial full KV cache exhausted12-GiB GPU memory; its source, logs and partial attempts are preserved under `incidents/kv-cache-OOM/`. Representation caching keeps the fullcontext and all eight estimators. The cache modes agreed to2.68e-7 in full precision on the training-only engineering probe; their AMP outputs differed by up to0.000787. The representation-cache saved-model replay was exact. Those checks do not substitute for complete-fold performance evaluation.

Read live progress from the repository root:

```sh
ssh faris@labkc-comm1 'cat /mnt/home/logika-faris-exploration/tabicl-compact-20261021/STATUS.json'
```

The terminal `result.json` is produced only after allfivefold artifacts and prediction replays pass. Historical target reuse remains disclosed. Earlier39TabICL experiments used different features and a56k/threefold protocol; their metrics are context only, not direct comparison scores.
