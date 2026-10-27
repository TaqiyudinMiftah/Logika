# Protokol eksperimen bersama

1. Sebelum SETIAP eksperimen, baca HEAD terbaru main dan perubahan sejak receipt terakhir. Jika ada commit teman, baca konfigurasi, definisi fitur, folds, hasil, AUC dan error analysis terlebih dahulu.
2. Bandingkan signature konfigurasi+fitur+data/fold/backbone dalam EXPERIMENT_REGISTRY.json. Nama eksperimen berbeda belum membuktikan metode berbeda. Tandai duplikat dan skip; replication/seed test harus diberi alasan eksplisit.
3. Jalankan satu eksperimen pada fold yang sama. Preprocessing, early stopping, ranking pairs dan learned representation tidak boleh memakai outer-validation labels.
4. Simpan hasil termasuk negatif, OOF, submission, kode/checkpoint/formula, matched CI dan analisis error. Data/OOF berlabel dan checkpoint besar tetap lokal/node; GitHub mendapat kode, laporan agregat, metadata dan hash.
5. Fetch HEAD sekali lagi sebelum push. Buat commit di atas HEAD terbaru dan update main dengan expected SHA serta force=false. Bila teman keburu push, baca perubahannya dan bangun commit baru di atas head baru; jangan force.
6. Setelah push verifikasi commit dan file, tulis receipt pada node, lalu evaluasi eksperimen berikutnya.

Limit RAM16GiB dibagi semua akun pada kontainer. Periksa headroom dan peak proses sebelum training. Jangan memulai model berat ketika ruang terlalu kecil. Jangan mengakhiri proses pengguna lain. V25 tetap incumbent public sampai ada bukti baru yang jelas.
