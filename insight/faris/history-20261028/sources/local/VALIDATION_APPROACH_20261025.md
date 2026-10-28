# Logika: current result and next decision

**The 0.54 local KS target is still unmet.** Five more fixed hypotheses completed in the latest continuation. None earned a submission. The highest observed point is **0.531505932**, but its tiny gain failed the prespecified checks; the selected local incumbent remains **0.531403805**.

| Decision | Current evidence |
|---|---|
| Best observed public submission | **V25, 0.56822**; exact uploaded bytes downloaded and verified. |
| Next closest public candidate | B, **0.56818**. |
| Higher local score transferred to Kaggle | A scored **0.531403805 locally** and **0.56656 publicly**; the higher local point did not win. |
| New submissions after reset | **0**; authenticated API still reports **3 available** at the recorded check. |
| Latest near-gain | H016 **0.531505932**, only +0.000102127 over A; loses on repeat seed, wins2/5folds, adjusted interval crosses zero. |
| 0.54 reached | **No.** |

The new tests covered final-mixture correction, balanced correction loss, a shared-application bureau measurement model, balanced Kumo contexts, and direct-KS spline training. All used the original five-fold pooled evaluation. A separate train/test membership classifier scored AUC **0.50172**, providing no useful evidence for domain weighting under that diagnostic.

The earlier H010 experiment is complete: five contexts per outer fold scored **0.528190778** with ordinary correction and **0.528059537** with direct-KS correction. Its matched one-context control scored **0.530290544**. H011's training percentile maps also failed. None led to a new full-data refit or upload.

All70,000 labels have prior development exposure. These comparisons and bootstrap intervals are conditional research evidence, not independent confirmation. Repayment snapshot timing remains unresolved. No new raw fields, hidden labels or row-order reconstruction were introduced.

Use the [verified V25 CSV](/home/parasite/Project/Competition/logika-2026/submission_best_public_V25_056822_verified.csv) as the preserved best observed public artifact. See the [continuation report](/home/parasite/Project/Competition/logika-2026/a100/faris/transfer-20261025/CONTINUATION_REPORT.md) for the exact results, failures, replay audits and submission receipts, or the [resumable state](/home/parasite/Project/Competition/logika-2026/a100/faris/transfer-20261025/research-state.json) for machine-readable status.
