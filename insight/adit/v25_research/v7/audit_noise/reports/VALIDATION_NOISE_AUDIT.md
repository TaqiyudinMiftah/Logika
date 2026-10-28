# Audit data dan noise validasi: V25 vs V221

Kesimpulan: V221 belum menunjukkan peningkatan ranking yang cukup stabil untuk menggantikan V25. Hasil paling sesuai dengan gain validasi yang kecil, sensitif terhadap threshold/pooling dan pemilihan kandidat berulang. Overfitting terhadap proses seleksi merupakan risiko kuat; penyebab tunggal penurunan public tidak dapat diidentifikasi tanpa label dan keanggotaan subset public.

Audit ini membaca data aktual: 70.000 train/9.507 event, 30.000 test, selected application/customer dari CSV mentah, matriks input, histori coverage per aplikasi, spectral cache, OOF dan checkpoint lama. Tidak melatih model baru, mengubah CSV baseline atau mengirim submission. Skor public V2210.56614 terverifikasi lewat Kaggle API dan exact CSV SHA256 (ref57720168).

## 1. Selisih validasi lebih kecil daripada kesan dari peringkat eksperimen

| Model | Pooled KS | Mean fold KS | AUC | Public KS |
|---|---:|---:|---:|---:|
| V25 | 0.53028466 | 0.53332982 | 0.84702928 | 0.56822 |
| V185 | 0.53115166 | 0.53356562 | 0.84710878 | 0.56699 |
| V221 | 0.53162935 | 0.53348679 | 0.84712490 | 0.56614 |

V221: delta pooledKS +0.00134469; delta mean-foldKS +0.00015697; delta AUC +0.00009562. Menang3/5fold. Dua fold turun -0.002651 dan -0.003103. Paired bootstrap95% deltaKS [-0.00109414,+0.00334117] masih memuat nol. Bootstrap mengondisikan fitted predictions; tidak mengoreksi ratusan percobaan adaptif atau retraining.

Setelah setiap fold diubah ke percentile ranking tanpa label, pooledKS V25=0.53015818 dan V221=0.53052961, selisih +0.00037143. Ini diagnostik sensitivitas penggabungan skala skor, bukan dekomposisi kausal maupun metode submission yang sudah divalidasi. Setiap ROC/KS dalam fold tetap sama di bawah transformasi monoton ini.

## 2. Puncak KS berpindah; kualitas tidak naik di semua rentang risiko

Puncak V25 ada pada top32.3686% populasi; V221 pada top31.0743%. Pada fraksi32.3686% yang sama, V25 menangkap7.434 default dan V2217.417: berkurang17. KS pada fraksi tersebut turun dari0.53028466 ke0.52821548. Pada top10%, V221 menangkap6 default lebih sedikit. Jadi maksimum KS yang lebih tinggi tidak berarti seluruh ranking lebih baik. AUC hanya naik sekitar0.000096; logloss/Brier tidak membaik terhadapV25.

## 3. Komposisi data aktual tidak mendukung dugaan drift besar yang sederhana

Train dan test sama-sama1Jan-28Des2025, masing-masing70k/30k,100k customer unik tanpa overlap. Keduanya mempunyai30% tanpa account history. Tanpa repayment history sekitar33.84%/33.85%. Tanpa allocation teramati33.9029%/33.9400%. Median declared income Rp7.784.706/Rp7.749.696. Tidak ada bukti bahwa test semata-mata berasal dari periode setelah train.

Diuji97 fitur numerik baseline/spectral; maksimum PSI=0.00115382. Ini pelemahan hipotesis pergeseran marginal besar, bukan bukti tidak ada multivariate/concept drift. Label test dan anggota subset public tidak tersedia. Histori snapshot DPD/global references lama tetap memiliki ketidakpastian provenance yang diwarisi.

## 4. Pola domain nyata: aktivitas dan kelengkapan histori

| Pesanan completed90d | Train N | Event | Default rate | Test share | Delta KS V221-V25 |
|---|---:|---:|---:|---:|---:|
| zero | 3428 | 106 | 3.092% | 4.907% | -0.006918 |
| 1to3 | 18715 | 1163 | 6.214% | 26.727% | -0.002998 |
| 4to10 | 36368 | 5141 | 14.136% | 51.680% | +0.003068 |
| over10 | 11489 | 3097 | 26.956% | 16.687% | -0.000115 |

Aktivitas e-commerce mempunyai hubungan risiko yang jelas, tetapi tidak menyatakan konsumsi menyebabkan default. Kelompok0-3order berisiko lebih rendah dan justru mengalami kemunduran KS/AUC padaV221. Tanpa allocation:23.732 nasabah,2.895event, default12.20%, deltaKS-0.003086; dengan allocation:46.268 nasabah,6.612event, default14.29%, deltaKS+0.001164. Tanpa account:21.000 nasabah,2.461event, default11.72%, deltaKS-0.001259. Komposisi kelompok-kelompok ini hampir sama di test; kesalahan lokal tetap relevan meski pooledKS naik.

## 5. Temuan representasi: coverage lebih kuat daripada spectral value secara univariat

Semua10kolom spectral-value mempunyai mask missing yang persis sama. Missing:21.736 nasabah,1.693event, default7.7889%. Finite:48.264 nasabah,7.814event, default16.1901%. Di antara48.264 nasabah finite, AUC masing-masing spectral value hanya0.4856-0.5072 dan KS0.00613-0.02490. Kelengkapan histori membawa sinyal kuat; bukti univariat atas pola frekuensi individual lemah. Efek conditional/interaksi tetap mungkin dan belum ditolak oleh audit ini.

Generator mengharuskan minimal8completed orders dalam180hari penuh dan first recorded event sebelum jendela. First event adalah proxy cakupan, bukan jaminan observasi lengkap. Missing spectral karenanya mencampurkan aktivitas, umur histori, dan ketersediaan informasi.

Preprocessor membuat10missing indicator yang identik. Dengan ridge dan total koefisien B untuk m kolom identik, minimum penalti pada subspace ini adalah lambda*B^2/(2m), dibanding lambda*B^2/2 untuk satu kolom. Jadi menambahkan spectral juga mengubah kekuatan penalti coverage; perbandingan ini tidak hanya menambahkan informasi frekuensi. Identitas tersebut bukan ukuran seluruh effective regularization model.

CheckpointV167 mengonfirmasi10koefisien missing identik dalam setiap fold. Jumlahnya berubah antar-fold: -0.085577, -0.127893, +0.044929, +0.012956, +0.091104. Komponen missing berubah tanda; efek neto prediksi tetap bergantung pada seluruh basis dan kovariat. Ini menunjukkan parameter komponen tersebut belum stabil, bukan bukti kausal bahwa10kolom ini sendiri menyebabkan penurunan leaderboard.

## 6. Frozen Kumo tetap memungkinkan selection overfitting

Kumo100%frozen tidak menghilangkan overfitting pada correction heads, pilihan fitur, context sampling, threshold evaluasi dan kombinasi model. V221 mewarisiV167(145koefisien/head vs90V25), local-contextV178, calibrationV189 melaluiV193, serta tiga bootstrap headsV219. Field zero new fusion parameters benar untuk langkah blending terakhir, tetapi bukan ukuran kompleksitas seluruh ensemble. Head gapV221 yang disalin dariV25(0.011108) juga bukan gap ensemble baru; sourceV167 mempunyai gap0.016101.

Uji retrospektif40pembagian setengah data, stratifikasi fold/kelas, terhadap11 kandidat dengan OOF lokal tersedia: memilih pemenang di setengah pertama memberi rata-rata deltaKS+0.001304, tetapi pada komplemennya+0.000373. Selisih+0.000931. Ini hanya sensitivitas atas prediksi tetap dan kandidat yang sudah pernah dipilih dengan semua label; bukan nestedCV dan bukan estimasi generalisasi baru. Daftar kandidat yang tidak memiliki OOF lokal dicatat eksplisit, tidak diimputasi.

Risiko seleksi berulang ini konsisten dengan pembahasan [Cawley & Talbot(2010)](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf); publikasi tersebut tidak membuktikan penyebab khusus padaLOGIKA.

## 7. Keputusan dan audit lanjutan yang tepat

PertahankanV25 sebagai public incumbent. V221 gagal memperbaiki public untuk exactCSV-nya; numerical-best OOF bukan rekomendasi submission. Riwayat hasil selesai tidak diubah; public feedback dilampirkan lewat ledger berdasarkan SHA.

Sebelum perluasan fitur lain, ablation terkontrol yang membedakan mekanisme: V25+SATU spectral-eligibility/missing flag; V25+nilai spectral dengan shared missing flag tunggal dan penalti eksplisit; negative control permutasi spectral finite di dalam strata aktivitas/coverage. Fit/seleksi hanya inner-training dan kunci konfigurasi sebelum outer. Ini rancangan berikutnya, belum dijalankan sebagai hasil eksperimen.

Random customer-disjoint CV masih relevan karena test berada pada periode yang sama dan customer unik. Tambahkan temporal/seed stress tests sebagai pemeriksaan stabilitas, bukan otomatis menganggap temporal split meniru test. Semua label pengembangan sudah terekspos selama penelitian; pembagian ulang tidak menciptakan holdout yang benar-benar belum tersentuh. Peningkatan kecil perlu konsistensi mean-foldKS, AUC, top-risk capture, coverage groups dan uncertainty, bukan maksimum pooledKS saja.

AntreanV240-V242 tetap tersimpan; tidak ada eksperimen baru dimulai selama audit ini.

## Reproduksi dan bukti

Kode: audit_validation_noise.py, audit_data_segments.py, audit_spectral_coefficients.py dan write_noise_audit_report.py di v7/scripts/. Metadata raw dan label/OOF privat tetap di workspace. CSV agregat, hash sumber, formula, cohort results, confidence intervals dan keterbatasan ada pada laporan JSON/CSV di folder ini. Koefisien checkpoint dibaca dengan lingkungan pinned node480; analisis lainnya berjalan lokal agar tidak menambah beban RAM node.

[Kurva KS pada fraksi populasi yang sama](KS_RISK_PROFILE.svg). Ini perbandingan labeledOOF; marginal histogram submission tidak dapat mengungkap hidden class-conditional KS.
