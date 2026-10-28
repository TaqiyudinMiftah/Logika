# A100 full-cache TabICL verification retry

The original eight-estimator TabICLv2 compact90 recipe exceeded the RTX3060 full-KV-cache memory limit. This retries that exact backbone configuration on node380's A100 after our Kumo adaptation finished. It is a verification retry, not a new novel architecture or an additional feature search. PC2's representation-cache run continues independently; folds from different execution modes will not be mixed.

Every outer fold uses50,000 context rows,6,000 correction rows and14,000 validation rows under the original frozen fivefold protocol. Correction18 and spline77 pipelines, weights75:25, seed2026+fold and the fixed80:20 Kumo/TabICL secondary blend are unchanged. No evaluation labels enter backbone or correction fits.

Source and pretrained checkpoint hashes were checked before launch. The pinned environment uses TabICL2.2.0 and einops0.8.1 with the existing matching core dependency bundle. The actual A100 training-only fit/save/reload preflight passed with exact replay. Main execution is one GPU job at a time, with a36-GiB allocator cap and at least40,000MiB free required before each fold. No unrelated process is modified.

The inherited deadline remains2026-10-21 19:31:11 WIB, with athree-hour finalization reserve. The detached runner and every owned child enforce timeouts. Model training never prunes from one-fold performance. All fitted models and checkpoints remain on node380 under `/home/taqiyudinmiftah/faris/tabicl-compact-a100-20261021/`.

```sh
python3 a100/profile.py node380 term 'cat /home/taqiyudinmiftah/faris/tabicl-compact-a100-20261021/STATUS.json'
```

The terminal `result.json` requires allfivefolds, complete70k OOF coverage, artifact hashes and full fitted-model prediction replays. It contains independently checked point metrics; paired uncertainty is a later PC2 calculation. It is not automatic promotion, full-data refitting or a competition submission.
