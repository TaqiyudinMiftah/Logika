# Antrean10 eksperimen Kumo frozen

Total10; selesai2; tersisa8. BaselineV25, target pooledKS>=0.531. Kumo100% frozen.

Otomatis berjalan bila RAM dan VRAM mencukupi. Pemantau mengecek setiap10detik; tidak memuat model/data sambil menunggu. Hasil, checkpoint, KS/AUC, CI, error analysis, risiko overfit dan submissionCSV disimpan. Push dan pemeriksaan main sebelum lanjut. Dua retry khusus SIGKILL setelah menunggu resource; kesalahan kode atau perubahan peer membutuhkan pemeriksaan, bukan pengulangan buta.

| ID | Metode | Status |
|---|---|---|
| V169 | Kumo Large learns V25 exposure and conditional-liquidity counts directly | COMPLETE |
| V170 | Frozen Kumo with spending velocity and liquidity pressure | COMPLETE |
| V171 | Frozen Kumo with compact spectral band representation | WAITING_FOR_RESOURCES |
| V172 | Coverage-aware spectral model with V25 fallback for unobserved repayment allocation | QUEUED |
| V173 | Stratified context bagging: two40k frozen Kumo fits from identical50k eligible pool | QUEUED |
| V174 | Empirical-Bayes allocation propensity and uncertainty inside frozen Kumo | QUEUED |
| V175 | Selective latent uncertainty and income-contradiction inputs | QUEUED |
| V176 | Compact item-value basket composition, without unverified order denominators | QUEUED |
| V177 | Conservative fixed90:10 V25 and V169 score blend | QUEUED |
| V178 | Conservative global-local frozen context experts by history coverage | QUEUED |