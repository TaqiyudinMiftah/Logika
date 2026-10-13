# Panduan review notebook LOGIKA

Mulai dari **01** untuk mempelajari alur paling sederhana. Baca **02–06** untuk melihat perubahan fitur, validasi, dan model. Jika ingin mereview kandidat terbaru, buka **06**; submission default tetap **05**. Setiap notebook dapat dijalankan sendiri dari dalam proyek: kode EDA, agregasi fitur, preprocessing, training, prediksi test, dan ekspor CSV berada di sel notebook.

| Notebook | Pipeline yang direview | Validasi / model final | Public historis |
|---|---|---|---:|
| [01 — Baseline XGBoost](01_baseline_xgboost.ipynb) | Profil dan agregasi seluruh riwayat | Holdout 25%; model final dilatih pada seluruh train | 0,52623 |
| [02 — Tuning KS versi 2](02_tuning_ks_v2.ipynb) | Fitur extended dan perbandingan kandidat pada data tuning | Core 42.000 / tuning 10.500 / confirmation 17.500; final LightGBM 486 pohon | Belum tercatat |
| [03 — Ensemble KS](03_ensemble_ks.ipynb) | Pemilihan bobot LightGBM dan CatBoost | Tiga fold; 25% LightGBM + 75% CatBoost | 0,54710 |
| [04 — Submission versi 3](04_submission_v3.ipynb) | Harga per kategori, shrinkage, arus kas, dan class weighting | Dua model baru sepuluh fold, digabungkan dengan versi KS | 0,55845 |
| [05 — Submission versi 4](05_submission_v4.ipynb) | Harga dinormalisasi, cicilan matang, dan model spline | Versi 3 + CatBoost baru + spline, masing-masing sepertiga | Belum tercatat |
| [06 — Kandidat versi 5](06_submission_v5.ipynb) | Harga semua basket dan TabM, early stopping di dalam training fold | 75% versi 4 + 25% TabM; KS OOF 0,52446 | Belum tercatat |

Skor public di atas berasal dari file historis. Angka validasi di notebook adalah KS lokal, bukan skor leaderboard. `submission_v2.csv` adalah alternatif yang pernah dibuat; skor 0,54710 berasal dari `submission_ks.csv`.

## Cara menjalankan

1. Instal paket dari direktori proyek: `python -m pip install -r requirements-optimization.txt -r requirements-notebook.txt`.
2. Unduh data sekali: `python scripts/download_data.py`. Kredensial Kaggle tetap di file lokal yang dikecualikan dari Git; jangan menyalin key ke sel notebook.
3. Buka notebook melalui Jupyter atau VS Code, pilih kernel Python yang memiliki paket tersebut, lalu **Restart & Run All**.
4. Ambil `submission.csv` pada folder output yang ditampilkan sel terakhir.

Untuk **06**, gunakan Python 3.10 dan tambahkan `requirements-v5.txt`. Lingkungan referensi memakai PyTorch 1.13.1+cu116 dan GPU A100. CPU dapat dipakai, dengan durasi dan kemungkinan pembulatan berbeda. Skor validasi versi 5 naik dibanding versi 4, tetapi hanya lima fold membaik dari syarat enam; file kandidat terpisah belum mengganti `submission.csv` di root.

| Notebook | Lokasi output dari direktori proyek |
|---|---|
| 01 | `artifacts/notebook_runs/01_baseline/submission.csv` |
| 02 | `artifacts/notebook_runs/02_v2/submission.csv` |
| 03 | `artifacts/notebook_runs/03_ks/submission.csv` |
| 04 | `artifacts/notebook_runs/04_v3/submission.csv` |
| 05 | `artifacts/notebook_runs/05_v4/submission.csv` |
| 06 | `artifacts/notebook_runs/06_v5/submission.csv` |

Model historis, data mentah, fitur parquet, dan prediksi OOF per baris tidak disertakan di GitHub. Pada clone baru, sel agregasi membangun fitur yang belum ada, dan `.fit()` melatih model yang belum ada. Versi 3 dan 4 memerlukan lebih banyak pelatihan karena juga membangun komponen ensemble sebelumnya.

## Cara membaca kode

- Bagian **EDA** menunjukkan ukuran data, missing value, ketidakseimbangan target, pola bulanan, dan perbedaan kelompok pekerjaan.
- Bagian **fitur** dipecah per sumber data. Fokus pada cutoff, `groupby`, denominator rasio, dan join. Semua kode agregasi terlihat di sel, termasuk ketika cache membuatnya dilewati.
- Bagian **preprocessing** menjelaskan bagaimana kategori, missing numerik, dan class weighting ditangani.
- Bagian **validasi** membedakan tuning, confirmation, OOF, dan model final. Kandidat serta grid bobot historis juga ditampilkan.
- Bagian **prediksi test dan ekspor** menghitung ulang probabilitas dari model, menggabungkan anggota ensemble, memeriksa ID/probabilitas, lalu menulis CSV.

Gunakan panel **Table of Contents / Outline** untuk melompat antarbagian. Bagian agregasi cukup panjang karena semua kode tersedia langsung untuk direview; jika sudah memahami fitur dasar pada 01, Anda dapat fokus pada tambahan fitur dan model di versi selanjutnya.

## Mengendalikan cache dan eksperimen

| Flag di sel persiapan | Jika `True` |
|---|---|
| `FORCE_REBUILD_FEATURES` | Membaca ulang riwayat dan membangun kembali fitur |
| `FORCE_RETRAIN_MODELS` | Melatih ulang model, termasuk komponen ensemble sebelumnya; OOF dihitung dari model tersebut |
| `FORCE_RECOMPUTE_VALIDATION` | Menghitung ulang prediksi validasi dari model fold yang sesuai |
| `RUN_OPTIONAL_SEARCH` | Mengulang pencarian kandidat/jumlah pohon yang lebih mahal |

Default semua flag adalah `False`. Jika cache tersedia, notebook menggunakan model/OOF historis dan **tetap menghitung ulang prediksi test dari model**. Jika Anda mengubah parameter, daftar fitur, preprocessing, atau data mentah, aktifkan `FORCE_RETRAIN_MODELS`; jika mengubah agregasi, aktifkan pula `FORCE_REBUILD_FEATURES`. Beberapa ketidaksesuaian parameter cache diperiksa dan menghentikan eksekusi dengan pesan untuk melatih ulang.

Parameter dan bobot final historis ditulis eksplisit. Menjalankan pencarian opsional menampilkan hasil pencarian; untuk memakai pilihan baru, ubah konfigurasi final pada sel model. Ini membuat perbedaan antara reproduksi dan eksperimen mudah diperiksa.

Model validasi tidak berlatih pada baris validasinya. Namun fitur compact, jumlah pohon, dan bobot pernah dipilih dari label training yang sama, sehingga hasil CV dapat mengandung optimisme seleksi. Cache OOF tidak berasal dari model yang dilatih pada seluruh label.

## Pemeriksaan yang telah dilakukan

Kelima notebook dijalankan sampai ekspor dengan model historis yang tersedia. Semua CSV memiliki 30.000 ID sesuai sample, probabilitas valid, serta prediksi dan SHA-256 yang sama dengan checkpoint masing-masing. Sel fitur juga dijalankan ulang dari data mentah di direktori terpisah: fitur dasar, extended, versi 3, versi 4, dan tabel biro terakhir identik dengan tabel eksperimen asli. Rincian verifikasi tersedia pada [verification.json](verification.json).

Notebook EDA lama di root tetap merupakan pembaca laporan. Enam notebook dalam folder ini adalah pipeline yang dapat membuat submission. Notebook 06 juga dijalankan sampai selesai: CSV identik dengan `submission_v5.csv`, dan prediksi OOF/test direproduksi persis dari sepuluh model TabM tersimpan. Detail pemeriksaan versi 5 tersedia di `artifacts/optimization/v5/`.

Untuk meregenerasi notebook dari kode sumber: `python scripts/build_review_notebooks.py`. Untuk sekaligus mengeksekusinya: `python scripts/build_review_notebooks.py --execute`.

Untuk notebook 06: `python scripts/build_review_notebook_v5.py --execute`. Generator 06 memakai sel pipeline notebook 05 sebagai komponen versi 4, lalu menambahkan kode fitur dan TabM secara langsung. Jalankan generator 01–05 terlebih dahulu jika ingin meregenerasi seluruh notebook dari awal.
