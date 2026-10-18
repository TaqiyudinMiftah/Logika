# Riset KS OOF 0,54

**Target belum tercapai pada eksperimen lima fold yang selesai.**

Lima fold stratified seed 31027, 70.000 OOF. Public baseline 0,56393 menurut pengguna; public semua kandidat baru belum diketahui.

| Kandidat | Bobot baru | KS pooled | Rata-rata KS fold | Δ pooled | Fold membaik |
|---|---:|---:|---:|---:|---:|
| public_056393_recipe | 0 | 0.528510 | 0.530955 | +0.000000 | 0/5 |

Model yang belum menyelesaikan lima fold tidak diberi skor pooled: ebm_denoised_unweighted, tabpfn35_fast_v5_cache_e8_ctx50000, tabpfn35_fast_v5_e8_ctx50000

Model and feature selection reuse training labels and folds. This is not an untouched holdout; changed submissions have unknown public score.

Audit menghitung KS dengan SciPy dan sklearn, memeriksa semua ID outer-training/validation, konteks pretrained, fingerprint fitur dan checksum prediksi.
