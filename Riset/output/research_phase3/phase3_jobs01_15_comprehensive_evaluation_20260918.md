# Evaluasi Komprehensif Eksperimen Formal Phase 3 Job 1--15

**Tanggal evaluasi:** 18 September 2026  
**Cakupan:** 15 job formal, lima backbone, seed 42/123/2026  
**Metrik primer:** AP `trash_plastic` COCO bbox IoU 0,50:0,95  
**Status:** Evaluasi training formal lengkap; keputusan akurasi tersedia, pembekuan final menunggu benchmark inference dan audit provenance  
**Official test:** Tidak digunakan

## 1. Ringkasan Eksekutif

Seluruh 15 proses berakhir dengan `exit_code=0` dan menghasilkan 30 epoch serta
10 evaluasi berkala. Namun, keberhasilan proses tidak sama dengan validitas
ilmiah: Job 3, ConvNeXt-Tiny seed 42, menghasilkan loss `NaN` pada seluruh epoch
dan metrik nol. Sesuai protokol, Job 3 diklasifikasikan sebagai
`numerical_failure`, dipertahankan dalam laporan, dan dikeluarkan dari agregasi.
Dengan demikian terdapat 14 run finite: ResNet50, EfficientNet-B0, Swin-T, dan
PVTv2-B1 masing-masing tiga seed valid; ConvNeXt-Tiny hanya dua seed valid.

Berdasarkan metrik primer dan aturan keputusan yang dibekukan:

- **CNN yang direkomendasikan adalah EfficientNet-B0.** Mean AP target 7,78
  berjarak 0,51 poin dari ResNet50 (7,27), masih dalam toleransi ekuivalensi
  1 poin. Tie-break stabilitas memilih EfficientNet-B0 karena SD 0,24 jauh lebih
  kecil daripada 1,46. EfficientNet-B0 juga memakai peak reserved VRAM training
  lebih rendah dan memiliki AP75 target lebih tinggi, walaupun ResNet50 lebih
  baik pada overall mAP dan FN kritis.
- **Transformer yang direkomendasikan adalah PVTv2-B1.** Mean AP target 8,54
  melampaui Swin-T 7,09 sebesar 1,45 poin, melewati toleransi 1 poin sehingga
  keputusan sudah ditentukan oleh metrik primer. PVTv2-B1 juga lebih stabil dan
  menggunakan VRAM training lebih rendah. Swin-T tetap lebih tinggi pada overall
  mAP, sehingga kemenangan PVTv2-B1 bersifat target-class-specific.
- **ConvNeXt-Tiny tidak layak dipilih pada matriks ini.** Selain satu kegagalan
  numerik, mean dua run validnya hanya 4,49 AP target dan berada di bawah dua
  kandidat CNN lainnya.

Keputusan di atas merupakan keputusan **akurasi dan stabilitas training formal**.
Pembekuan pasangan final untuk tahap DAFNet belum sebaiknya dilakukan sebelum
benchmark inference formal, verifikasi reload/hash checkpoint terpilih, dan
audit empat versi source tree selesai.

## 2. Dasar Evaluasi

Setiap backbone diuji pada seed 42, 123, dan 2026 dengan 100% `train_internal`,
30 epoch, AMP FP16, batch size dan gradient accumulation yang sama, manifest
beku yang sama, serta evaluator yang sama. Checkpoint per run dipilih berdasarkan
AP `trash_plastic` tertinggi pada `val_internal`. Metrik dalam tabel ditampilkan
pada skala 0--100; nilai artefak asli disimpan pada skala 0--1.

FN kritis adalah ground truth `trash_plastic` medium/besar yang tidak terpasangkan
pada IoU minimal 0,50 dan score minimal 0,05. FN kritis tidak menggantikan AP
karena hanya memakai satu pasangan threshold.

## 3. Audit Validitas dan Kelengkapan

| Pemeriksaan | Hasil |
|---|---|
| Jumlah proses selesai | 15/15 |
| Run finite dan dapat diagregasi | 14/15 |
| Kegagalan ilmiah | Job 3, `numerical_failure` |
| Anggaran epoch | 30 epoch pada seluruh job |
| Evaluasi berkala | 10 evaluasi per job |
| Checkpoint best dan final | Tersedia pada seluruh direktori run |
| Perangkat | NVIDIA GeForce RTX 3060 Laptop GPU |
| Manifest/protokol | Dilaporkan konsisten pada artefak/preflight |
| Official test | Tidak diakses |
| Benchmark inference formal | Belum tersedia; direktori `benchmarks` kosong |
| Resume | Job 9 dilanjutkan setelah epoch 11 dengan training state |
| Source tree | Empat hash berbeda; perlu attestation final |

Status `completed` Job 3 hanya berarti proses selesai tanpa exit error; run tetap
tidak valid karena syarat finite loss dan finite metrics tidak terpenuhi. Job 9
dapat dipakai secara bersyarat karena optimizer, scheduler, scaler, epoch, dan
RNG state direkam saat resume, tetapi kesetaraan bitwise terhadap run tanpa
interupsi belum diuji.

## 4. Hasil Setiap Job

| Job | Backbone | Seed | Status ilmiah | Epoch terbaik | AP target | AP50 | AP75 | Overall mAP | FN kritis /270 |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ResNet50 | 42 | Valid | 9 | 7,94 | 18,43 | 4,60 | 8,32 | 74 |
| 2 | EfficientNet-B0 | 42 | Valid | 27 | 7,90 | 16,33 | 6,16 | 6,71 | 98 |
| 3 | ConvNeXt-Tiny | 42 | **Gagal numerik** | - | - | - | - | - | 270 |
| 4 | Swin-T | 42 | Valid | 9 | 9,24 | 23,42 | 5,12 | 7,69 | 86 |
| 5 | PVTv2-B1 | 42 | Valid | 9 | 8,29 | 19,02 | 5,11 | 7,23 | 89 |
| 6 | EfficientNet-B0 | 123 | Valid | 12 | 7,50 | 18,46 | 4,08 | 7,19 | 91 |
| 7 | ConvNeXt-Tiny | 123 | Valid | 12 | 3,33 | 10,68 | 1,34 | 5,48 | 84 |
| 8 | Swin-T | 123 | Valid | 18 | 6,56 | 17,36 | 2,94 | 9,20 | 100 |
| 9 | PVTv2-B1 | 123 | Valid; resume | 15 | 8,81 | 24,39 | 4,09 | 8,44 | 88 |
| 10 | ResNet50 | 123 | Valid | 12 | 5,59 | 14,69 | 3,22 | 8,73 | 100 |
| 11 | ConvNeXt-Tiny | 2026 | Valid | 12 | 5,65 | 14,99 | 4,08 | 5,98 | 90 |
| 12 | Swin-T | 2026 | Valid | 30 | 5,46 | 13,48 | 3,21 | 10,12 | 100 |
| 13 | PVTv2-B1 | 2026 | Valid | 18 | 8,51 | 20,48 | 4,18 | 8,65 | 105 |
| 14 | ResNet50 | 2026 | Valid | 27 | 8,28 | 19,12 | 6,09 | 9,68 | 72 |
| 15 | EfficientNet-B0 | 2026 | Valid | 24 | 7,93 | 18,51 | 5,56 | 7,11 | 87 |

## 5. Agregasi Antar-Seed

Run gagal tidak diberi skor nol dan tidak dimasukkan ke mean. SD adalah simpangan
baku sampel.

| Backbone | n valid | AP target mean ± SD | Rentang AP | AP50 mean | AP75 mean | AR100 mean | Overall mAP mean | FN kritis mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | 3 | 7,27 ± 1,46 | 5,59--8,28 | 17,41 | 4,63 | 34,96 | **8,91** | **82,0** |
| EfficientNet-B0 | 3 | **7,78 ± 0,24** | 7,50--7,93 | 17,77 | **5,27** | 33,75 | 7,00 | 92,0 |
| ConvNeXt-Tiny | 2 | 4,49 ± 1,64 | 3,33--5,65 | 12,84 | 2,71 | 31,36 | 5,73 | 87,0 |
| Swin-T | 3 | 7,09 ± 1,94 | 5,46--9,24 | 18,09 | 3,76 | 32,93 | **9,00** | 95,3 |
| PVTv2-B1 | 3 | **8,54 ± 0,26** | 8,29--8,81 | **21,30** | **4,46** | 32,46 | 8,11 | **94,0** |

### Ranking metrik primer

1. PVTv2-B1: 8,54 AP.
2. EfficientNet-B0: 7,78 AP.
3. ResNet50: 7,27 AP.
4. Swin-T: 7,09 AP.
5. ConvNeXt-Tiny: 4,49 AP dari dua seed valid, disertai satu kegagalan.

Ranking global ini hanya deskriptif. Keputusan resmi tetap dilakukan terpisah
untuk keluarga CNN dan Transformer.

## 6. Keputusan Keluarga CNN

### 6.1 EfficientNet-B0 versus ResNet50

Selisih mean AP target EfficientNet-B0 dan ResNet50 adalah 0,51 poin, lebih kecil
dari toleransi ekuivalensi 1 poin. Karena itu digunakan tie-break yang telah
ditetapkan sebelum hasil terlihat:

1. mean AP target: berada dalam zona ekuivalen;
2. SD AP: EfficientNet-B0 0,24 versus ResNet50 1,46;
3. FN kritis: ResNet50 lebih baik, 82 versus 92;
4. AP75: EfficientNet-B0 lebih baik, 5,27 versus 4,63;
5. peak reserved VRAM training: EfficientNet-B0 lebih rendah.

Tie-break berhenti pada stabilitas, sehingga **EfficientNet-B0 menjadi rekomendasi
CNN**. Hasil ini bukan kemenangan mutlak. ResNet50 menang pada dua dari tiga
seed, overall mAP lebih tinggi 1,91 poin, dan rata-rata FN kritis lebih rendah
10 objek. Mean EfficientNet-B0 menjadi lebih tinggi karena penurunan ResNet50
pada seed 123 cukup besar. Artikel harus melaporkan trade-off ini agar pilihan
berbasis target class tidak disalahartikan sebagai superioritas universal.

### 6.2 Posisi ConvNeXt-Tiny

ConvNeXt-Tiny tidak kompetitif pada metrik primer dan gagal numerik pada seed 42.
Pada dua seed finite, ResNet50 dan EfficientNet-B0 mengunggulinya pada kedua
perbandingan berpasangan. FN kritis ConvNeXt tidak terlalu tinggi, tetapi AP-nya
rendah; hal ini menguatkan bahwa metrik satu threshold tidak cukup untuk
pemilihan. ConvNeXt-Tiny tidak direkomendasikan untuk cabang CNN utama tanpa
studi diagnosis dan amendemen baru yang terpisah dari matriks formal ini.

## 7. Keputusan Keluarga Transformer

PVTv2-B1 menghasilkan mean AP target 8,54, sedangkan Swin-T 7,09. Selisih 1,45
poin melampaui toleransi ekuivalensi 1 poin, sehingga metrik primer langsung
memilih **PVTv2-B1 sebagai rekomendasi Transformer**. PVT menang pada seed 123
dan 2026; Swin menang pada seed 42.

PVTv2-B1 juga jauh lebih stabil (SD 0,26 versus 1,94), memiliki AP50/AP75 target
lebih tinggi, FN kritis sedikit lebih rendah, dan memakai VRAM training lebih
rendah. Sebaliknya, Swin-T memiliki overall mAP lebih tinggi (9,00 versus 8,11),
terutama pada seed 2026. Dengan demikian, PVT unggul untuk tujuan primer
`trash_plastic`, tetapi Swin menunjukkan representasi multikelas yang lebih baik.

## 8. Stabilitas dan Pengaruh Seed

Ranking per seed berubah nyata:

- seed 42: Swin-T > PVTv2-B1 > ResNet50 > EfficientNet-B0; ConvNeXt gagal;
- seed 123: PVTv2-B1 > EfficientNet-B0 > Swin-T > ResNet50 > ConvNeXt-Tiny;
- seed 2026: PVTv2-B1 > ResNet50 > EfficientNet-B0 > ConvNeXt-Tiny > Swin-T.

Perubahan tersebut membenarkan penggunaan multiseed. EfficientNet-B0 dan
PVTv2-B1 menunjukkan rentang AP paling sempit. Swin-T paling sensitif terhadap
seed, dari 9,24 menjadi 5,46. ResNet50 juga berubah cukup besar, dari 5,59 hingga
8,28. Tiga seed cukup untuk keputusan deskriptif sesuai protokol, tetapi tidak
cukup untuk klaim signifikansi statistik.

Kemenangan berpasangan adalah ResNet50 2--1 atas EfficientNet-B0, tetapi mean dan
stabilitas memilih EfficientNet-B0 sesuai aturan yang dibekukan. PVTv2-B1 menang
2--1 atas Swin-T dan juga unggul pada mean.

## 9. Target Class versus Performa Multikelas

Hasil memperlihatkan konflik objektif yang konsisten:

- EfficientNet-B0 memimpin CNN pada AP plastik, tetapi ResNet50 memimpin overall
  mAP dan FN kritis.
- PVTv2-B1 memimpin Transformer pada AP plastik, tetapi Swin-T memimpin overall
  mAP.
- Swin-T seed 2026 mencapai overall mAP tertinggi di antara seluruh run (10,12),
  tetapi AP plastiknya hanya 5,46.

Karena tujuan penelitian telah menetapkan AP `trash_plastic` sebagai metrik
primer, konflik tersebut tidak mengubah pemenang. Akan tetapi, konflik ini perlu
menjadi batas ilmiah dan motivasi analisis komplementaritas: pasangan terpilih
belum tentu terbaik untuk seluruh kelas.

## 10. FN Kritis dan Interpretasi Kesalahan

ResNet50 memiliki FN kritis terendah secara agregat (82/270), diikuti ConvNeXt
pada dua run valid (87), EfficientNet-B0 (92), PVTv2-B1 (94), dan Swin-T (95,3).
PVT seed 2026 memiliki AP target tinggi tetapi FN tertinggi, 105/270. Ini bukan
kontradiksi: AP mengintegrasikan precision--recall pada berbagai threshold dan
IoU, sedangkan FN kritis memakai satu score dan IoU threshold.

Implikasinya, threshold operasional tidak boleh dipilih hanya dari ranking AP.
Setelah backbone dipilih, analisis kurva precision--recall dan kalibrasi
threshold pada `val_internal` perlu dilakukan sebagai tahap terpisah dan sudah
didefinisikan sebelumnya agar tidak menjadi tuning pascahoc.

## 11. Dinamika Training dan Checkpoint

Tiga belas dari empat belas run finite mencapai AP terbaik sebelum epoch 30;
hanya Swin-T seed 2026 mencapai nilai terbaik pada epoch 30. Rata-rata epoch
terbaik adalah 16 untuk ResNet50, 21 untuk EfficientNet-B0, 12 untuk ConvNeXt,
19 untuk Swin-T, dan 14 untuk PVTv2-B1.

Pada hampir semua run, AP checkpoint akhir lebih rendah daripada checkpoint
terbaik walaupun loss training terus menurun. Penurunan AP dari best ke final
rata-rata berkisar sekitar 0,29--0,67 poin, tergantung backbone. Ini mendukung
checkpoint selection berbasis validasi dan menunjukkan gejala generalization
gap, tetapi belum cukup untuk menyimpulkan overfitting tanpa analisis kurva yang
lebih rinci.

## 12. Efisiensi Sumber Daya Training

| Backbone | Peak allocated mean (GiB) | Peak reserved mean (GiB) | Durasi tercatat mean (jam) |
|---|---:|---:|---:|
| ResNet50 | 1,55 | 2,23 | 5,17 |
| EfficientNet-B0 | **1,12** | **1,67** | 5,71 |
| ConvNeXt-Tiny (2 valid) | 1,56 | 2,13 | 5,89 |
| Swin-T | 1,66 | 2,29 | 7,30 |
| PVTv2-B1 | 1,34 | 1,99 | 5,50 |

EfficientNet-B0 paling hemat VRAM training di keluarga CNN; PVTv2-B1 lebih hemat
daripada Swin-T. Namun durasi lintas semua job belum layak menjadi klaim formal
karena implementasi timer berubah antara generasi runner. Benchmark inference
formal dengan batch 1, warm-up, repetisi, sinkronisasi CUDA, dan urutan
counterbalanced masih wajib dijalankan sebelum frontier Pareto final dibentuk.

## 13. Provenance dan Ancaman Validitas

Empat hash source tree tercatat:

- Job 1--5: `171c0885...`;
- Job 6--9: `8be41a82...`;
- Job 10--12: `0f5b98ad...`;
- Job 13--15: `4ede5e93...`.

Perubahan yang telah diketahui mencakup dukungan full-state checkpoint/resume,
perbaikan resolver resume, dan pengelolaan protokol beku. Kesetaraan arsitektur,
loss, transformasi, optimizer, scheduler, dan evaluator antarempat source tree
harus disahkan dalam attestation final berbasis diff. Tanpa attestation ini,
perbandingan tetap informatif tetapi klaim bahwa hanya backbone yang berubah
belum sepenuhnya tertutup.

Ancaman validitas lain:

- hanya tiga seed dan tidak ada klaim inferensial;
- satu reviewer pada audit visual;
- satu dataset dan satu GPU, sehingga generalisasi eksternal terbatas;
- ConvNeXt hanya memiliki dua run valid;
- Job 9 adalah run resume dan belum diuji ekuivalensi bitwise;
- benchmark inference formal belum dilakukan;
- analisis komplementaritas kesalahan per citra/objek belum dilakukan;
- official test tetap terkunci sehingga angka ini adalah hasil validasi internal,
  bukan estimasi final pada test resmi.

## 14. Keputusan dan Tahap Berikutnya

### Keputusan yang sudah didukung

- Rekomendasi CNN berdasarkan protokol akurasi/stabilitas: **EfficientNet-B0**.
- Rekomendasi Transformer berdasarkan metrik primer: **PVTv2-B1**.
- ConvNeXt-Tiny tidak direkomendasikan dari matriks formal ini.
- Hasil satu seed terbukti tidak memadai untuk menentukan ranking.

### Keputusan yang belum boleh dibekukan

- frontier Pareto lengkap karena latensi/throughput inference belum tersedia;
- pasangan final DAFNet sebelum audit provenance dan benchmark selesai;
- klaim komplementaritas CNN--Transformer;
- threshold operasional;
- performa final pada official test.

### Urutan pekerjaan yang direkomendasikan

1. Buat attestation diff untuk empat source tree dan pastikan tidak ada perubahan
   pada arsitektur, loss, data, atau aturan evaluasi yang memengaruhi fairness.
2. Verifikasi SHA-256 dan reload inference checkpoint terbaik dari setiap run
   valid, terutama kandidat EfficientNet-B0 dan PVTv2-B1.
3. Jalankan benchmark inference formal pada lingkungan yang sama sesuai protokol.
4. Bentuk frontier Pareto per keluarga dan terapkan tie-break final.
5. Analisis overlap TP/FP/FN EfficientNet-B0 dan PVTv2-B1 untuk menguji
   komplementaritas sebelum membangun DAFNet.
6. Bekukan keputusan backbone beserta reviewer, commit, hash protokol, hash
   checkpoint, dan alasan keputusan.
7. Setelah seluruh keputusan development dibekukan, jalankan official test satu
   kali sesuai protokol final.

## 15. Kesimpulan

Eksperimen formal Job 1--15 memberikan bukti bahwa stabilitas seed dan fokus
metrik target mengubah interpretasi seleksi backbone. EfficientNet-B0 merupakan
CNN paling stabil dan efisien pada AP `trash_plastic`, sedangkan PVTv2-B1
memberikan AP target tertinggi serta stabilitas terbaik pada keluarga
Transformer. ResNet50 dan Swin-T tetap unggul pada beberapa aspek multikelas dan
FN, sehingga hasil tidak boleh diringkas sebagai dominasi universal.

Rekomendasi berbasis training formal adalah pasangan **EfficientNet-B0 +
PVTv2-B1**. Statusnya masih rekomendasi bersyarat sampai benchmark inference,
Pareto lengkap, reload checkpoint, dan attestation source selesai. Official test
tetap tidak boleh digunakan pada tahap ini.

## 16. Sumber Artefak

- Matriks formal: `Riset/kaggle-lab/trashcan-phase3-runtime-v1/protocol/formal_matrix.json`.
- Protokol mesin-baca: `Riset/configs/phase3_training_protocol.json`.
- Freeze formal: `Riset/output/research_phase3/phase3_formal_protocol_freeze.json`.
- Run: `Riset/output/runs/phase3_formal_*/` sesuai pemetaan Job 1--15.
- Evaluasi per run: `metrics/evaluation_001` sampai `evaluation_010`.
- Resource: `metrics/resource_usage.json`.
- Riwayat: `metrics/epoch_metrics.csv` dan `metrics/history.json`.
- Ulasan sebelumnya: `Riset/output/research_phase3/phase3_jobs01_10_interim_review_20260917.md`.