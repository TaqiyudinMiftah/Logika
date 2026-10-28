# Riset V25: target pooled OOF KS0.531

Baseline public tetap **V25:0.56822**, verified terhadap hash CSV. Pooled OOF V25 **0.5302846600**, AUC **0.8470292789**. Target pooled0.531 tercapai pada fold pengembangan; generalisasi belum terbukti. Tidak ada submission otomatis ke Kaggle.

**242 eksperimen aktif tercatat**, termasuk122 eksperimen modeling baru. Semuanya memiliki lima fold, OOF, checkpoint dan submission di workspace node480. Hasil terbaik baru **V221: KS0.5316293525, AUC0.8471248959**, deltaKS vsV25 **+0.0013446925**, dengan3/5 fold menang. Paired95% deltaKS CI **[-0.00109414,+0.00334117]**. Kandidat tidak otomatis menggantikan incumbent public.

## Hasil dan arah penelitian

- V120:V131 menguji regularisasi, kekuatan koreksi dan probability/logit blend. V127 menurunkan L2 dari0.003 ke0.001 dengan strength0.5 dan blend75:25 tetap.
- V132:V143 menguji pasangan fitur dan embedding interaksi rank4/8 pada18 input V25 yang sama. Semua lebih rendah daripada V25; memperbesar representasi pada6k correction rows belum membantu.
- V144:V151 menambahkan dua histogram-GBDT, dilatih56k outer-training dengan inner validation. Input116 adalah union informasi yang telah dipakai pipeline V25, tanpa fitur histori baru. Expert standalone KS0.52154/0.52115; fixed blends terbaik0.530528, belum melebihi V127.
- V152:V159 menguji pairwise ranking dan smooth-CDF loss yang ditambatkan ke V25. Semua lebih rendah daripada V25. Eksperimen terakhir V159 KS0.5301448252, AUC0.8470188808.

Setiap eksperimen memiliki `ERROR_ANALYSIS.md/json`, `ERROR_SEGMENTS.csv`, confidence interval dan metrik per-fold di `results/`. Analisis menggunakan prediksi outer-held-out, mencatat perbaikan maupun kemunduran top-risk capture dan segmen. Case-level OOF/target, raw data dan checkpoint besar tetap di node480, tidak dipublikasikan.

## Reproduksi

Workspace runtime: `/home/fariselhakim/adit/Logika/insight/goal054_offset_baseline_20261019`.
Lingkungan: `/home/fariselhakim/adit/.venv-logika-offset-20261027/bin/python`.
Kode ada di `v7/scripts/`; formula/metode tercatat di `EXPERIMENTS.csv`, konfigurasi dan fingerprint di `EXPERIMENT_REGISTRY.json`. Jalur `data/`, `artifacts/`, `v7/extension/data/` membutuhkan cache privat yang sudah diaudit, bukan diunduh dari GitHub. Jangan menjalankan ulang batch sebelum memeriksa protokol sinkronisasi.

Kumo Large100% frozen,90 input,50k context,8 estimators. Koreksi V25 memakai18 input dan6k rows; original spline77; lima outer fold seed31027 identik. Target/model selection memakai fold pengembangan berulang; CI bootstrap bersyarat pada fitted scores, tidak mengoreksi pencarian atau retraining. Ketidakpastian snapshot DPD/global references lama tetap diwarisi.

## Kolaborasi

Baca `SYNC_PROTOCOL.md` sebelum setiap eksperimen. Fetch main terbaru, periksa hasil teman, bandingkan signature, jalankan satu eksperimen, catat hasil/error analysis/submission hash, lalu push fast-forward. Jangan menimpa hasil teman atau memakai force push.


Latest completed: **V241**, pooledKS **0.5302003018**, AUC **0.8469968971**. See its results directory.


**Public feedback: V221 = 0.56614, exact CSV verified. V25 remains the public incumbent at0.56822. Highest pooled OOF has not translated into a public improvement.** Read the [data and validation-noise audit](v7/audit_noise/reports/VALIDATION_NOISE_AUDIT.md), including coverage subgroups, shared spectral missing indicators, fold/threshold sensitivity and selection uncertainty.
