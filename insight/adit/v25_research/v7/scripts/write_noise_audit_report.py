"""Render the completed, aggregated validation/data audit for review."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
V=ROOT/'v7'
OUT=V/'audit_noise/reports'


def main():
    s=json.loads((OUT/'VALIDATION_NOISE_AUDIT.json').read_text())
    a=json.loads((OUT/'DATA_SEGMENT_AUDIT.json').read_text())
    missing=json.loads((OUT/'SPECTRAL_MISSINGNESS_AUDIT.json').read_text())
    coefficients=json.loads((OUT/'SPECTRAL_DUPLICATE_COEFFICIENT_AUDIT.json').read_text())
    models={m['model']:m for m in s['models']}
    d=pd.read_csv(OUT/'FEATURE_DRIFT.csv')
    b,p=models['V25'],models['V221']
    q=s['selection_sensitivity']
    ci=s['V221_vs_V25_CI']['delta_KS']
    lines=[
        '# Audit data dan noise validasi: V25 vs V221',
        '',
        'Kesimpulan: V221 belum menunjukkan peningkatan ranking yang cukup stabil untuk menggantikan V25. Hasil paling sesuai dengan gain validasi yang kecil, sensitif terhadap threshold/pooling dan pemilihan kandidat berulang. Overfitting terhadap proses seleksi merupakan risiko kuat; penyebab tunggal penurunan public tidak dapat diidentifikasi tanpa label dan keanggotaan subset public.',
        '',
        'Audit ini membaca data aktual: 70.000 train/9.507 event, 30.000 test, selected application/customer dari CSV mentah, matriks input, histori coverage per aplikasi, spectral cache, OOF dan checkpoint lama. Tidak melatih model baru, mengubah CSV baseline atau mengirim submission. Skor public V2210.56614 terverifikasi lewat Kaggle API dan exact CSV SHA256 (ref57720168).',
        '',
        '## 1. Selisih validasi lebih kecil daripada kesan dari peringkat eksperimen',
        '',
        '| Model | Pooled KS | Mean fold KS | AUC | Public KS |',
        '|---|---:|---:|---:|---:|',
        f"| V25 | {b['KS']:.8f} | {b['mean_fold_KS']:.8f} | {b['AUC']:.8f} | 0.56822 |",
        f"| V185 | {models['V185']['KS']:.8f} | {models['V185']['mean_fold_KS']:.8f} | {models['V185']['AUC']:.8f} | 0.56699 |",
        f"| V221 | {p['KS']:.8f} | {p['mean_fold_KS']:.8f} | {p['AUC']:.8f} | 0.56614 |",
        '',
        f"V221: delta pooledKS {p['delta_pooled_KS']:+.8f}; delta mean-foldKS {p['delta_mean_fold_KS']:+.8f}; delta AUC {p['AUC']-b['AUC']:+.8f}. Menang3/5fold. Dua fold turun -0.002651 dan -0.003103. Paired bootstrap95% deltaKS [{ci[0]:+.8f},{ci[1]:+.8f}] masih memuat nol. Bootstrap mengondisikan fitted predictions; tidak mengoreksi ratusan percobaan adaptif atau retraining.",
        '',
        f"Setelah setiap fold diubah ke percentile ranking tanpa label, pooledKS V25={b['rank_normalized_pooled_KS']:.8f} dan V221={p['rank_normalized_pooled_KS']:.8f}, selisih {p['rank_normalized_pooled_KS']-b['rank_normalized_pooled_KS']:+.8f}. Ini diagnostik sensitivitas penggabungan skala skor, bukan dekomposisi kausal maupun metode submission yang sudah divalidasi. Setiap ROC/KS dalam fold tetap sama di bawah transformasi monoton ini.",
        '',
        '## 2. Puncak KS berpindah; kualitas tidak naik di semua rentang risiko',
        '',
        f"Puncak V25 ada pada top{100*b['peak_top_fraction']:.4f}% populasi; V221 pada top{100*p['peak_top_fraction']:.4f}%. Pada fraksi32.3686% yang sama, V25 menangkap7.434 default dan V2217.417: berkurang17. KS pada fraksi tersebut turun dari0.53028466 ke0.52821548. Pada top10%, V221 menangkap6 default lebih sedikit. Jadi maksimum KS yang lebih tinggi tidak berarti seluruh ranking lebih baik. AUC hanya naik sekitar0.000096; logloss/Brier tidak membaik terhadapV25.",
        '',
        '## 3. Komposisi data aktual tidak mendukung dugaan drift besar yang sederhana',
        '',
        'Train dan test sama-sama1Jan-28Des2025, masing-masing70k/30k,100k customer unik tanpa overlap. Keduanya mempunyai30% tanpa account history. Tanpa repayment history sekitar33.84%/33.85%. Tanpa allocation teramati33.9029%/33.9400%. Median declared income Rp7.784.706/Rp7.749.696. Tidak ada bukti bahwa test semata-mata berasal dari periode setelah train.',
        '',
        f"Diuji{len(d)} fitur numerik baseline/spectral; maksimum PSI={s['maximum_feature_PSI']:.8f}. Ini pelemahan hipotesis pergeseran marginal besar, bukan bukti tidak ada multivariate/concept drift. Label test dan anggota subset public tidak tersedia. Histori snapshot DPD/global references lama tetap memiliki ketidakpastian provenance yang diwarisi.",
        '',
        '## 4. Pola domain nyata: aktivitas dan kelengkapan histori',
        '',
        '| Pesanan completed90d | Train N | Event | Default rate | Test share | Delta KS V221-V25 |',
        '|---|---:|---:|---:|---:|---:|'
    ]
    for row in a['segments']:
        if row['segment']=='orders90':
            lines.append(f"| {row['value']} | {row['n']} | {row['events']} | {100*row['target_rate']:.3f}% | {100*row['test_share']:.3f}% | {row['delta_KS']:+.6f} |")
    lines += [
        '',
        'Aktivitas e-commerce mempunyai hubungan risiko yang jelas, tetapi tidak menyatakan konsumsi menyebabkan default. Kelompok0-3order berisiko lebih rendah dan justru mengalami kemunduran KS/AUC padaV221. Tanpa allocation:23.732 nasabah,2.895event, default12.20%, deltaKS-0.003086; dengan allocation:46.268 nasabah,6.612event, default14.29%, deltaKS+0.001164. Tanpa account:21.000 nasabah,2.461event, default11.72%, deltaKS-0.001259. Komposisi kelompok-kelompok ini hampir sama di test; kesalahan lokal tetap relevan meski pooledKS naik.',
        '',
        '## 5. Temuan representasi: coverage lebih kuat daripada spectral value secara univariat',
        '',
        'Semua10kolom spectral-value mempunyai mask missing yang persis sama. Missing:21.736 nasabah,1.693event, default7.7889%. Finite:48.264 nasabah,7.814event, default16.1901%. Di antara48.264 nasabah finite, AUC masing-masing spectral value hanya0.4856-0.5072 dan KS0.00613-0.02490. Kelengkapan histori membawa sinyal kuat; bukti univariat atas pola frekuensi individual lemah. Efek conditional/interaksi tetap mungkin dan belum ditolak oleh audit ini.',
        '',
        'Generator mengharuskan minimal8completed orders dalam180hari penuh dan first recorded event sebelum jendela. First event adalah proxy cakupan, bukan jaminan observasi lengkap. Missing spectral karenanya mencampurkan aktivitas, umur histori, dan ketersediaan informasi.',
        '',
        'Preprocessor membuat10missing indicator yang identik. Dengan ridge dan total koefisien B untuk m kolom identik, minimum penalti pada subspace ini adalah lambda*B^2/(2m), dibanding lambda*B^2/2 untuk satu kolom. Jadi menambahkan spectral juga mengubah kekuatan penalti coverage; perbandingan ini tidak hanya menambahkan informasi frekuensi. Identitas tersebut bukan ukuran seluruh effective regularization model.',
        '',
        'CheckpointV167 mengonfirmasi10koefisien missing identik dalam setiap fold. Jumlahnya berubah antar-fold: '+', '.join(f"{x['coefficient_sum']:+.6f}" for x in coefficients)+'. Komponen missing berubah tanda; efek neto prediksi tetap bergantung pada seluruh basis dan kovariat. Ini menunjukkan parameter komponen tersebut belum stabil, bukan bukti kausal bahwa10kolom ini sendiri menyebabkan penurunan leaderboard.',
        '',
        '## 6. Frozen Kumo tetap memungkinkan selection overfitting',
        '',
        'Kumo100%frozen tidak menghilangkan overfitting pada correction heads, pilihan fitur, context sampling, threshold evaluasi dan kombinasi model. V221 mewarisiV167(145koefisien/head vs90V25), local-contextV178, calibrationV189 melaluiV193, serta tiga bootstrap headsV219. Field zero new fusion parameters benar untuk langkah blending terakhir, tetapi bukan ukuran kompleksitas seluruh ensemble. Head gapV221 yang disalin dariV25(0.011108) juga bukan gap ensemble baru; sourceV167 mempunyai gap0.016101.',
        '',
        f"Uji retrospektif40pembagian setengah data, stratifikasi fold/kelas, terhadap{q['candidate_count']} kandidat dengan OOF lokal tersedia: memilih pemenang di setengah pertama memberi rata-rata deltaKS{q['average_selection_delta']:+.6f}, tetapi pada komplemennya{q['average_complement_delta']:+.6f}. Selisih{q['average_selection_minus_complement']:+.6f}. Ini hanya sensitivitas atas prediksi tetap dan kandidat yang sudah pernah dipilih dengan semua label; bukan nestedCV dan bukan estimasi generalisasi baru. Daftar kandidat yang tidak memiliki OOF lokal dicatat eksplisit, tidak diimputasi.",
        '',
        'Risiko seleksi berulang ini konsisten dengan pembahasan [Cawley & Talbot(2010)](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf); publikasi tersebut tidak membuktikan penyebab khusus padaLOGIKA.',
        '',
        '## 7. Keputusan dan audit lanjutan yang tepat',
        '',
        'PertahankanV25 sebagai public incumbent. V221 gagal memperbaiki public untuk exactCSV-nya; numerical-best OOF bukan rekomendasi submission. Riwayat hasil selesai tidak diubah; public feedback dilampirkan lewat ledger berdasarkan SHA.',
        '',
        'Sebelum perluasan fitur lain, ablation terkontrol yang membedakan mekanisme: V25+SATU spectral-eligibility/missing flag; V25+nilai spectral dengan shared missing flag tunggal dan penalti eksplisit; negative control permutasi spectral finite di dalam strata aktivitas/coverage. Fit/seleksi hanya inner-training dan kunci konfigurasi sebelum outer. Ini rancangan berikutnya, belum dijalankan sebagai hasil eksperimen.',
        '',
        'Random customer-disjoint CV masih relevan karena test berada pada periode yang sama dan customer unik. Tambahkan temporal/seed stress tests sebagai pemeriksaan stabilitas, bukan otomatis menganggap temporal split meniru test. Semua label pengembangan sudah terekspos selama penelitian; pembagian ulang tidak menciptakan holdout yang benar-benar belum tersentuh. Peningkatan kecil perlu konsistensi mean-foldKS, AUC, top-risk capture, coverage groups dan uncertainty, bukan maksimum pooledKS saja.',
        '',
        'AntreanV240-V242 tetap tersimpan; tidak ada eksperimen baru dimulai selama audit ini.',
        '',
        '## Reproduksi dan bukti',
        '',
        'Kode: audit_validation_noise.py, audit_data_segments.py, audit_spectral_coefficients.py dan write_noise_audit_report.py di v7/scripts/. Metadata raw dan label/OOF privat tetap di workspace. CSV agregat, hash sumber, formula, cohort results, confidence intervals dan keterbatasan ada pada laporan JSON/CSV di folder ini. Koefisien checkpoint dibaca dengan lingkungan pinned node480; analisis lainnya berjalan lokal agar tidak menambah beban RAM node.',
        '',
        '[Kurva KS pada fraksi populasi yang sama](KS_RISK_PROFILE.svg). Ini perbandingan labeledOOF; marginal histogram submission tidak dapat mengungkap hidden class-conditional KS.'
    ]
    (OUT/'VALIDATION_NOISE_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curves=[]
    for ident in ['V25','V221']:
        frame=pd.read_csv(V/'runs'/ident/'OOF.csv',float_precision='round_trip')
        y=frame.target.to_numpy(int);score=frame.prediction.to_numpy();order=np.argsort(-score,kind='stable')
        fraction=np.arange(1,len(y)+1)/len(y)
        curve=np.cumsum(y[order])/y.sum()-np.cumsum(1-y[order])/(len(y)-y.sum())
        curves.append(curve)
    fig,axs=plt.subplots(2,1,figsize=(9,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    keep=(fraction>=.2)&(fraction<=.45)
    for ident,curve in zip(['V25','V221'],curves):axs[0].plot(100*fraction[keep],curve[keep],label=ident,lw=1.5)
    axs[0].set(ylabel='TPR - FPR',title='OOF KS profile: compare the same selected population fraction')
    axs[0].legend();axs[0].grid(alpha=.2)
    axs[1].plot(100*fraction[keep],(curves[1]-curves[0])[keep],color='#a33b3b')
    axs[1].axhline(0,color='black',lw=.7);axs[1].set(xlabel='Highest-risk population selected (%)',ylabel='V221 - V25');axs[1].grid(alpha=.2)
    for ax in axs:
        ax.axvline(100*b['peak_top_fraction'],ls=':',color='#666',lw=1)
        ax.axvline(100*p['peak_top_fraction'],ls='--',color='#666',lw=1)
    fig.tight_layout();fig.savefig(OUT/'KS_RISK_PROFILE.svg');plt.close(fig)
    print('AUDIT_REPORT_WRITTEN',OUT/'VALIDATION_NOISE_AUDIT.md')


if __name__=='__main__':main()
