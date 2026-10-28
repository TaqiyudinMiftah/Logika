# Fold-local prior repair

Rebuild the inherited v3/v4/v5 and latent empirical-Bayes statistics using only the frozen 50,000 context applicants in each outer fold. Correction and validation applicants receive transforms from that context. Real target labels are unavailable to this feature builder.

All inherited source snapshots remain in vendor/. Outputs and intermediate files remain inside this study. Historical models, predictions and submissions are preserved. Raw DPD snapshot uncertainty is unchanged; this repair addresses cohort-fitting scope only.

Fold0 preprocessing is running on PC2 as `logika-fold-priors-f0-attempt2-20261021.service`. Attempt1 failed before completion because bureau_latest.parquet was missing; its partial files were preserved and canonical provider-report construction was added. No fitted competition model or completed pipeline came from that attempt.

Model training remains unadmitted. After preprocessing verification, the prepared Kumo input-placement comparison must use repaired inputs for all conditions and refit the spline branch; it cannot reuse the old pooled-statistic spline predictions.
