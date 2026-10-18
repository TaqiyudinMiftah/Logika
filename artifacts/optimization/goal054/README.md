# Riset menuju KS OOF 0,54

Goal dimulai 18 Oktober 2026. Metrik utama adalah KS gabungan seluruh 70.000
prediksi out-of-fold, memakai StratifiedKFold lima fold, shuffle=True, seed 31027.
Rata-rata KS fold, seluruh KS per fold, AUC, dan delta terhadap baseline pada fold
identik juga dilaporkan. Memilih satu fold yang melewati 0,54 tidak memenuhi goal.

Baseline adalah campuran 75% Kumo Large + 25% spline: pooled KS **0,528510**,
rata-rata KS fold **0,530955**. Pengguna melaporkan public **0,56393** untuk
`submission_kumo_spline_exploratory.csv`; file dengan checksum
`3de98287976a7137ad63c946e9932b6ac3ba3a7ee6df607d74cde1aa6a8fc417`
dipertahankan. Lihat [pencatatan public](../kumo_hybrid/public_result_056393.json).

## Jalur awal

- **TabPFN 3.5 / 3.5-Fast:** 8 anggota, 50.000 konteks dari outer-training,
  fitur V5 90 kolom atau fitur denoised yang sudah didokumentasikan. Revisi bobot
  diikat ke commit Hugging Face. Inferensi dan data tetap lokal.
- **Explainable Boosting Machine:** model aditif dengan 12 interaksi yang dipilih
  dari outer-training, 8 bag internal, smoothing dan early stopping internal.
  Varian loss biasa dan seimbang diuji. Bobot seimbang dihitung per outer-training;
  prior populasi dikembalikan sebelum probabilitas dicampur.

Bobot tambahan model baru terhadap baseline ditetapkan 25%, 50%, 75%, serta model
tunggal. Rencana dan parameter disimpan sebelum hasil lengkap tersedia:
[rencana utama](research_plan.json), [EBM](ebm_plan.json).

## Reproduksi

Siapkan fitur V5 dan reverse menggunakan notebook 09/10. Gunakan lingkungan Python
3.11 terpisah dan `requirements-goal054.txt`.

```bash
python scripts/run_research054.py
python scripts/research_ebm054.py
python scripts/research_goal054.py --compare-only
python scripts/research_ebm054.py --compare-only
```

Mode cache TabPFN disimpan di CPU di antara query untuk mengurangi pengulangan
perhitungan konteks. Tidak ada kuantisasi cache. File SafeTensors dimaterialisasi
dengan suffix yang benar karena TabPFN 9.2 mendeteksi format setelah meresolusi
symlink Hub. [Catatan teknis](runtime_adjustment.json) dan preflight mendokumentasikan
perubahan eksekusi. Percobaan awal dihentikan sebelum OOF lengkap dibuat dan tidak
dipakai untuk seleksi.

## Batas interpretasi

Label training, fitur dan lima split ini pernah digunakan pada eksperimen lain.
Pemilihan model berulang mengandung optimisme seleksi; skor ini bukan hasil holdout
yang belum pernah disentuh. Transformasi yang belajar dari label, interaksi,
early stopping, dan model baru dipasang hanya pada outer-training. ID tidak menjadi
fitur, dan label test tidak tersedia. Semua file dengan prediksi berubah mempunyai
skor public yang belum diketahui. Target belum dinyatakan tercapai sampai seluruh
lima fold dan audit selesai.
