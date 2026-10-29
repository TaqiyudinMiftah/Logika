# Credit post-hoc: tinjauan repo, data, dan residual V25

Tanggal: 29 Oktober 2026. Repo diperiksa sampai main `47fed7d5e670d0c387e42fa9c0415fce1d176462` melalui sumber GitHub. Ini analisis, bukan eksperimen model baru. Tidak ada fitting correction baru, modifikasi submission, atau pengiriman Kaggle.

## Makna istilah dan batas kepastian

Pencarian nama persis `credit posthoc`, `credit_posthoc`, dan `post_hoc` pada repo serta pencarian web tidak mengidentifikasi suatu implementasi/paper unik yang dapat dipastikan sebagai metode teman pengguna. Jangan mengklaim sudah mereproduksi metode tersebut atau menganggapnya nama algoritme standar.

Interpretasi yang relevan bagi dataset ini adalah **credit-aware post-processing / conditional risk correction**: model utama menghasilkan skor, kemudian informasi kredit mengoreksi kesalahan yang masih tersisa. Rumus konseptual `p_new = sigmoid(logit(p_base) + lambda*g(credit_features, p_base))`. Karena g berbeda per nasabah, urutan risiko bisa berubah. Ini berbeda dari satu transformasi monoton global terhadap semua skor.

Kalibrasi global yang strictly increasing mempertahankan ranking, KS, dan AUC (selama tidak ada pembulatan/ties baru). Isotonic dapat membuat ties. Kalibrasi per segmen atau kalibrasi sebelum blending dapat mengubah ranking gabungan. Tidak ada jaminan KS membaik. Materi resmi: [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html).

Keluarga literatur paling dekat untuk koreksi berbasis subgroup/residual adalah [Multiaccuracy](https://www.aies-conference.com/2019/wp-content/papers/main/AIES-19_paper_44.pdf), [Multicalibration](https://proceedings.mlr.press/v80/hebert-johnson18a.html), dan implementasi [mcboost](https://mlr-org.github.io/mcboost/). Ini rujukan yang relevan, bukan identifikasi metode teman. Penelitian [Hansen dkk.](https://arxiv.org/abs/2406.06487) juga menemukan manfaat post-processing tergantung model dan kondisi awal; tidak otomatis meningkatkan model yang sudah terkalibrasi.

## Problem dan data yang sudah diperiksa

Data dictionary v2 dibaca lengkap. Target adalah mencapai DPD90 dalam12bulan kalender. Jumlah aktual:70.000train,9.507event(13,5814%),30.000test;70.000customertrain unik dan tidak tumpang tindih dengan customer test. Pilihan tenor pinjaman target12bulan. Audit data terdahulu menunjukkan train/test mencakup periode2025yang sama; validasi customer-disjoint stratified tidak otomatis salah hanya karena kasusnya kredit.

Setiap fold yang dicek:50.000context Kumo,6.000correction,14.000outervalidation, saling terpisah. Kumo90input/frozen, original spline77input; V25correction18input. Mean/SD price laten berasal dari pengukuran harga kategori relatif; bureau laten menggabungkan pengukuran provider; repayment propensity dan income/burden menyumbang bukti kemampuan/perilaku bayar. Enam fitur tambahanV25 merepresentasikan jumlah bulan kas negatif/nonnegatif, bulan rekening, order90hari, durasi histori e-commerce, dan allocationcount.

Sumber: [original correction code](https://github.com/TaqiyudinMiftah/Logika/blob/47fed7d5e670d0c387e42fa9c0415fce1d176462/scripts/research_context_correction054.py). V25 adalah pengembangan prosedur ini: `p_corrected=sigmoid(logit(p_Kumo)+0.5*X*beta)`, kemudian `p_final=0.75*p_corrected+0.25*p_spline`. Beta difit dengan BCE+ridge0.003 pada6.000baris. Tidak ada backbone epoch tambahan.

Cutoff aplikasi tetap berlaku: historical decisions sebelum cutoff, account hanya bulan selesai, bureau memakai available_datetime, e-commerce ordertotal dihitung satu kali perorder dan cancelled bukan spending. DPD/rawpaid snapshot dan global reference features mewarisi ketidakpastian baseline; audit ini tidak menghapusnya atau memberikan sertifikasi point-in-time baru.

## Pengukuran ulang dari prediksi held-out yang tersedia

Ini pengukuran ulang cache, tidak melatih model. Prediksi dan label diselaraskan berdasarkanapplication_id; semua fold diperiksa.

| Prosedur pada cache/fold sama | KS | AUC |
|---|---:|---:|
| Frozen Kumo sebelum koreksi |0.5272829071|0.8471128164|
|75%rawKumo+25%originalspline|0.5284874448|0.8473225044|
|Original12-input credit offset+spline|0.5292994659|0.8472026884|
|V25:18-input credit/coverage offset+spline|0.5302846600|0.8470292789|

V25 dibanding campuran tanpa correction: deltaKS+0.0017972153, deltaAUC-0.0002932255. Ini contoh nyata bahwa KS naik belum tentu seluruh ranking membaik. PublicV25tetap0.56822 berdasarkan receipt sebelumnya, bukan hasil public baru pada audit ini.

Uji remapping `sigmoid(0.7*logit(p_V25)+0.2)` menghasilkan KS dan AUC persis sama denganV25. Ini mendemonstrasikan bahwa perubahan skala skor saja bukan penjelasan untuk lonjakan KS.

## Apakah masih ada residual kredit yang sederhana?

Perbandingan kuartil atas vs bawah dilakukan dalam fold dan20bin skorV25 yang sama. Bobot per sel adalah minimum jumlah kedua kelompok agar perbedaan komposisi rentang skor berkurang. Statistik yang diukur: selisih residual rata-rata `y-p`. Ini deskriptif, bukan penyesuaian kausal atau validasi model baru. Tujuh kontras yang ditetapkan sebelum pembacaan hasil semuanya mempunyai interval95%melintasi0; koreksi Bonferroni ketujuhnya tidak menemukan hasil signifikan. Koreksi tersebut tidak mencakup seluruh riwayat eksperimen.

| Perbandingan high-minus-low | Selisih residual (percentage point) |95%interval(pp)|
|---|---:|---:|
|Bureau posterior|+0.116|[-0.617,+0.849]|
|Late30posterior|+0.533|[-0.446,+1.512]|
|Installment/observed income|+0.095|[-0.662,+0.851]|
|Existingtotalburden|+0.543|[-0.236,+1.321]|
|Priceposterior|-0.119|[-1.158,+0.919]|
|Bureauuncertainty|-0.257|[-0.968,+0.454]|
|Declaredincome gap|-0.018|[-0.780,+0.744]|

Contoh silang yang patut menjadi hipotesis, bukan aturan koreksi siap pakai:

* Bureauposterior>700.8303 sekaliguslate30posterior>0.2506393:2.537nasabah/451event, default17.7769% vs meanprediction16.9195%. Residual+0.8574pp, interval95%[-0.4965,+2.2113]pp. Perbedaan belum meyakinkan. Dalam skor0.1–0.2,651nasabah/94event mempunyai default dan meanprediction sama-sama sekitar14.4393%.
* Totalburden>1.20608 sekaligus allocationcount<=3:558nasabah/132event, default23.6559% vs prediksi21.5114%. Residual+2.1445pp, interval[-0.9019,+5.1909]pp. Jumlah cukup kecil dan ketidakpastian besar; jangan langsung menaikkan risiko kelompok ini.
* Tidak ada allocation teramati:23.732nasabah/2.895event, default12.1987% vs prediksi12.0461%. Residual+0.1527pp, interval[-0.2204,+0.5257]pp. Missing history tidak boleh otomatis dianggap buruk atau baik.

Implikasi: banyak sinyal kredit marginal telah tercermin padaV25. Tidak menemukan residual sederhana bukan bukti tidak ada interaksi/urutan temporal yang bermanfaat. Namun, tidak ada dasar dari pemeriksaan ini untuk menambahkan shift besar berdasarkan DPDatau bureau saja.

Repo juga mencatat uji nested credit-agreement(bureau×late30) sebelumnya: KS0.52862782 vs control0.52929947; tiga interaksi dan selector tidak memberi jointKS/AUCgain. [Laporan](https://github.com/TaqiyudinMiftah/Logika/blob/47fed7d5e670d0c387e42fa9c0415fce1d176462/insight/faris/history-20261028/sources/local/a100/faris/teammate-approach-20261020/REPORT.md). Jadi interaksi sederhana itu tidak layak dipromosikan sebagai penemuan baru tanpa alasan desain yang berbeda.

## Desain follow-up yang relevan jika dilakukan

Tentukan sedikit segmen sebelum evaluasi: strength-of-evidence repayment, contradiction bureau/repayment, dan burden/liquidity. Fit residual auditor yang sangat kecil/regularized menggunakan prediksi inner-held-out; nol koreksi adalah opsi dasar. Batasi perubahan logit dan gunakan shrinkage menurut jumlah observasi. Pilih kompleksitas/strength hanya dalam innertraining, kemudian laporkan hasil outer identik, KS/AUC, antar-fold, bootstrap, dan kelompok yang dirugikan.

Kendala spesifik pipeline: original spline sudah dilatih menggunakan6.000baris correction. Karena itu skor finalV25 pada baris tersebut tidak boleh diperlakukan sebagai prediksi jujur untuk melatih meta-correction tambahan. Pilihan aman adalah tetap mengoreksi Kumo sebelum blend, atau membangun inner-OOF untuk seluruh pipeline(Kumo, correction, spline) sebelum melatih post-hoc pada skor final. Memakai seluruh70.000OOFuntuk fitting lalu mengukur pada70.000yang sama akan menghasilkan evaluasi bocor.

Kesimpulan: istilah paling masuk akal dari data adalah **conditional credit-risk reranking**, bukan trik global probability calibration. Kerangka dasarnya sudah ada padaV25. Metode teman, besaran gain, dan baseline asalnya tetap belum terverifikasi; jangan memaksakan identifikasi. Bukti saat ini mendukung audit residual terkontrol, bukan klaim bahwa satu koreksi kredit baru akan otomatis memberikan lonjakan.

## Artefak

`CREDIT_POSTHOC_AUDIT.json`, `MATCHED_SCORE_CONTRASTS.csv`, `CREDIT_RESIDUAL_SEGMENTS.csv`; executable `v7/scripts/audit_credit_posthoc.py`. Tidak ada informasi label individual dalam laporan publik ini. Case-level input/OOF tetap privat. Seluruh hasil bersifat conditional pada fittedscores dan fold yang berulang dipakai; interval tidak memperhitungkan retraining dan selection history.
