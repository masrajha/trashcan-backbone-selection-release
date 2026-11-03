# Hasil Benchmark Inferensi Formal, Attestation, dan Frontier Pareto Fase 3

**Tanggal analisis:** 21 September 2026  
**Cakupan:** Job formal 1--15, benchmark inferensi tag `20260920_v1`  
**Artefak utama:** `formal_inference_benchmark.json`  
**Status:** benchmark selesai; 14 run valid dianalisis; Job 3 dikecualikan dari agregasi karena `numerical_failure`

## 1. Ringkasan Eksekutif

Benchmark inferensi formal selesai untuk seluruh checkpoint yang secara ilmiah
layak dianalisis. Sebanyak **14 checkpoint valid x 3 repetisi = 42 record**
berstatus `completed`. Setiap repetisi menggunakan perangkat, precision, ukuran
input, batch size, warm-up, dan split validasi yang sama. Seluruh quality gate
reproduksi AP lulus dengan deviasi absolut maksimum **0,001016**, jauh di bawah
toleransi **0,01**.

Hasil utama adalah sebagai berikut.

1. **PVTv2-B1** memberikan kompromi terbaik secara keseluruhan: mean AP target
   tertinggi (**0,0854**), median latency antarseed terendah (**121,587 ms**),
   throughput tertinggi (**7,760 FPS**), dan VRAM inferensi yang tetap rendah
   (**398 MB maksimum**). PVTv2-B1 mendominasi Swin-T pada ketiga dimensi Pareto.
2. Frontier CNN terdiri dari **ResNet50** dan **EfficientNet-B0**. ResNet50 lebih
   cepat, sedangkan EfficientNet-B0 memiliki mean AP lebih tinggi, variasi AP
   lebih kecil, dan penggunaan VRAM lebih rendah. ConvNeXt-Tiny didominasi oleh
   ResNet50 dan satu seed-nya gagal secara numerik.
3. Keputusan sesuai aturan beku tetap **EfficientNet-B0 sebagai CNN terpilih**
   dan **PVTv2-B1 sebagai Transformer terpilih**. Benchmark inferensi memperkuat
   pilihan PVTv2-B1 dan tidak mengubah tie-break CNN yang sebelumnya ditentukan
   oleh akurasi target dan stabilitas multiseed.
4. Semua **15** file `best_model.pth` tersedia, mempunyai SHA-256 unik, dan dapat
   di-reload secara strict. Namun, kelulusan reload Job 3 hanya membuktikan
   integritas struktural checkpoint; Job 3 tetap tidak sah untuk agregasi hasil.
5. Empat source-tree hash dapat diterima sebagai **ekuivalen secara kondisional**
   terhadap kontrak eksperimen beku. Perubahan antarkelompok terutama berupa
   dukungan full-state checkpoint/resume dan dokumentasi, bukan perubahan
   arsitektur, loss, split, optimizer, scheduler, atau evaluator. Job 9 tetap
   diberi catatan khusus karena merupakan run resume dan ekuivalensi bitwise
   terhadap run tanpa interupsi belum diuji.

## 2. Artefak dan Ruang Lingkup Audit

Sumber data analisis:

- registry: `Riset/configs/phase3_formal_inference_runs.json`;
- hasil agregat: `Riset/output/research_phase3/formal_inference_benchmark/20260920_v1/formal_inference_benchmark.json`;
- tabel datar: `Riset/output/research_phase3/formal_inference_benchmark/20260920_v1/formal_inference_benchmark.csv`;
- record individual: `jobXX_repY.json` pada direktori benchmark yang sama;
- evaluasi training: `phase3_jobs01_15_comprehensive_evaluation_20260918.md`;
- snapshot `resolved_config.json`, `run_metadata.json`, dan checkpoint masing-masing run.

Benchmark berlangsung dari **20 September 2026 16:49:02 UTC** sampai
**22:22:18 UTC**, atau sekitar **5 jam 33 menit 16 detik**. Job 3 tidak
dijalankan pada benchmark karena registry telah membekukannya sebagai
`eligible=false` akibat loss/metric non-finite.

## 3. Protokol Benchmark Inferensi Formal

| Komponen | Nilai tetap |
|---|---:|
| Perangkat | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB |
| Compute capability / driver | 8.6 / 566.07 |
| PyTorch / torchvision | 2.13.0+cu126 / 0.28.0+cu126 |
| CUDA / cuDNN | 12.6 / 91002 |
| Precision | AMP FP16 |
| Batch size | 1 |
| Ukuran tensor input | 1 x 3 x 320 x 576 |
| Warm-up | 50 iterasi per load checkpoint |
| Sampel terukur | seluruh 1.211 citra `val_internal` per repetisi |
| Repetisi | 3 per checkpoint |
| Urutan | cyclic counterbalanced |
| Kondisi daya yang dideklarasikan | AC, high performance |
| Protocol SHA-256 | `844a7eaa380b01e8f6f2b8b02fd4036d87cc0441e0f2bee07d9b8074879634f1` |
| Manifest SHA-256 | `eb4dae55b2c4afb4a2167448ad81b5cd8fed7bc7399ae57a36b27851300c2914` |

Latency *end-to-end* mencakup pemindahan tensor ke GPU, forward inference,
pemindahan output ke CPU, dan sinkronisasi CUDA, tetapi tidak mencakup pembacaan
berkas citra. Throughput dihitung dari mean latency setiap repetisi. Angka per
run pada bagian berikut adalah median dari tiga median repetisi; agregat per
backbone menggunakan median antarseed agar lebih tahan terhadap outlier.

## 4. Validasi Kualitas dan Reload Checkpoint

Untuk repetisi pertama setiap checkpoint, prediksi dievaluasi kembali terhadap
COCO ground truth `val_internal`. Semua 14 checkpoint valid memenuhi:

- backbone, seed, protocol hash, manifest hash, dan stage formal cocok;
- target AP finite;
- selisih target AP terhadap registry <= 0,01;
- strict reload berhasil pada ketiga repetisi.

Dengan demikian terdapat **42 strict reload yang berhasil** pada checkpoint
valid. Job 3 diuji terpisah pada CPU dan berhasil di-reload dengan
`strict=True` (220 tensor parameter), sehingga total cakupan audit struktural
adalah **15/15 best checkpoint**. Job 3 tidak dipulihkan status ilmiahnya karena
reload tidak menghapus bukti kegagalan numerik selama training.

### 4.1 Checksum seluruh best checkpoint

| Job | Backbone | Seed | SHA-256 `best_model.pth` | Status |
|---:|---|---:|---|---|
| 1 | ResNet50 | 42 | `d78d6622068929fb632be5319f273b66fda7d67416e9a99f8f52b776cb3d137b` | Valid; reload lulus |
| 2 | EfficientNet-B0 | 42 | `a8a666deb3d2a8971a95b2ef162ca23450331a1cf50d3f5b4f4b574647937194` | Valid; reload lulus |
| 3 | ConvNeXt-Tiny | 42 | `b792849c2c4bf7a2e88161d3adbc687db931f9dbb8959cf030bf1f7db263b89e` | Reload lulus; **tidak eligible** |
| 4 | Swin-T | 42 | `aae267e60af796f4cbd607849d919f2c1951b6fabe5aa4b46aa3a648117b1df6` | Valid; reload lulus |
| 5 | PVTv2-B1 | 42 | `ef92f56b374bc2dc69cbb9b907d016da89cf0f7cd15d42074e4cc438973ac1b6` | Valid; reload lulus |
| 6 | EfficientNet-B0 | 123 | `61b256fb049b57dcbb168c1b2301a5d023466704024d598068922bcefd001036` | Valid; reload lulus |
| 7 | ConvNeXt-Tiny | 123 | `0d8ef1dcdce617aa96368469a6349cd57e55c88f42ab3b91051dd6c003d7b8c6` | Valid; reload lulus |
| 8 | Swin-T | 123 | `1b1d52e10a5c4bb89669e652e18d8ccb6b03a9ec24485612e133f644f7ca2b69` | Valid; reload lulus |
| 9 | PVTv2-B1 | 123 | `73267f01e9e78553bf6047f6f26532cfaac7e17f51ec305f7f27dd72fef9953b` | Valid; reload lulus; resume |
| 10 | ResNet50 | 123 | `d7be0142f9c6852dd22ee5d968f6140a118627fa5382401a38ed5a4e738084d9` | Valid; reload lulus |
| 11 | ConvNeXt-Tiny | 2026 | `fbb1ae8848d08889d5dd7821fd256c0ffbfd43b566d56a27a1be81cbd8cb10ec` | Valid; reload lulus |
| 12 | Swin-T | 2026 | `97c2a0343004574242e9080c77489dbf85961281125246857e3e6969fa9043c0` | Valid; reload lulus |
| 13 | PVTv2-B1 | 2026 | `8438d3356ef045fa1d4dcfdd5b3dbdcdd61d61bfccd4aa89a66d77906d245c6a` | Valid; reload lulus |
| 14 | ResNet50 | 2026 | `c171b34af5de7f56311bb8f510b7df9f83af85956fc7dee4e1156a0f43232c3a` | Valid; reload lulus |
| 15 | EfficientNet-B0 | 2026 | `1585add173ca2babbd2b933998efbe44118798ca409e66f6c87579dcb6a023e2` | Valid; reload lulus |

Checksum dihitung kembali pada 21 September 2026 dan cocok dengan hash yang
tercatat pada ketiga record benchmark untuk setiap run valid.

## 5. Hasil Benchmark per Job

| Job | Backbone | Seed | AP target | E2E median (ms) | Model-only median (ms) | Throughput (FPS) | Peak reserved VRAM (MB) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ResNet50 | 42 | 0,0793 | 147,972 | 161,772 | 6,435 | 436 |
| 2 | EfficientNet-B0 | 42 | 0,0793 | 159,818 | 173,158 | 5,704 | 318 |
| 4 | Swin-T | 42 | 0,0923 | 167,656 | 177,565 | 5,689 | 430 |
| 5 | PVTv2-B1 | 42 | 0,0825 | 143,286 | 150,556 | 6,455 | 380 |
| 6 | EfficientNet-B0 | 123 | 0,0740 | 149,406 | 161,881 | 6,335 | 364 |
| 7 | ConvNeXt-Tiny | 123 | 0,0334 | 146,227 | 158,082 | 6,422 | 502 |
| 8 | Swin-T | 123 | 0,0653 | 147,029 | 159,814 | 6,442 | 428 |
| 9 | PVTv2-B1 | 123 | 0,0884 | 121,587 | 129,514 | 7,760 | 398 |
| 10 | ResNet50 | 123 | 0,0555 | 131,299 | 142,144 | 7,096 | 416 |
| 11 | ConvNeXt-Tiny | 2026 | 0,0573 | 143,522 | 154,736 | 6,607 | 446 |
| 12 | Swin-T | 2026 | 0,0545 | 151,089 | 165,011 | 6,240 | 476 |
| 13 | PVTv2-B1 | 2026 | 0,0852 | 118,245 | 127,830 | 7,894 | 352 |
| 14 | ResNet50 | 2026 | 0,0836 | 125,056 | 137,123 | 7,439 | 418 |
| 15 | EfficientNet-B0 | 2026 | 0,0787 | 154,053 | 165,590 | 6,187 | 328 |

AP pada tabel adalah hasil quality gate benchmark dan dapat berbeda sangat kecil
dari pembulatan metrik training. Throughput bukan kebalikan langsung dari median
latency karena runner menghitungnya dari mean latency per repetisi.

## 6. Agregasi per Backbone

| Keluarga | Backbone | n valid | AP target mean ± SD | E2E median antarseed (ms) | E2E mean seed-median ± SD (ms) | Throughput median (FPS) | Peak reserved maksimum (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| CNN | ResNet50 | 3 | 0,0727 ± 0,0146 | **131,299** | 134,776 ± 11,847 | **7,096** | 436 |
| CNN | EfficientNet-B0 | 3 | **0,0778 ± 0,0024** | 154,053 | 154,426 ± 5,216 | 6,187 | **364** |
| CNN | ConvNeXt-Tiny | 2 | 0,0449 ± 0,0164 | 144,875 | 144,875 ± 1,913 | 6,515 | 502 |
| Transformer | Swin-T | 3 | 0,0709 ± 0,0194 | 151,089 | 155,258 ± 10,927 | 6,240 | 476 |
| Transformer | PVTv2-B1 | 3 | **0,0854 ± 0,0026** | **121,587** | 127,706 ± 13,596 | **7,760** | **398** |

Interpretasi:

- PVTv2-B1 bukan hanya paling akurat untuk `trash_plastic`, tetapi juga paling
  cepat pada perangkat dan workload formal ini.
- EfficientNet-B0 adalah CNN paling stabil dan paling hemat VRAM, tetapi tidak
  paling cepat. ResNet50 memberikan latency CNN terbaik.
- ConvNeXt-Tiny tidak menawarkan keuntungan pada AP, latency, maupun VRAM dan
  hanya mempunyai dua seed valid.
- Swin-T sensitif terhadap seed dan kalah dari PVTv2-B1 pada semua dimensi
  Pareto utama.

Hasil latency bersifat **hardware- dan implementasi-spesifik**. Ranking ini tidak
boleh digeneralisasikan ke GPU, batch size, precision, atau runtime lain tanpa
benchmark ulang.

## 7. Attestation Empat Source-Tree Hash

Empat source-tree hash yang tercatat pada Job 1--15 adalah:

| Kelompok job | Source-tree SHA-256 | Penilaian |
|---|---|---|
| 1--5 | `171c0885a65f02fefdcf7af57b7705270f43c6efcecf33d8ced74e9dc096e72b` | Baseline eksekusi formal |
| 6--9 | `8be41a822bc560ee9da4891b9a0441d11bd1b615d18a77b5dc80efa1b37d2c75` | Dukungan full-state checkpoint/resume |
| 10--12 | `0f5b98adf24923a04647fe3e649a2ab940955f74681287d5250aee689c6f2f02` | Resolver/run directory dibuat resume-aware |
| 13--15 | `4ede5e9359bdfb4831622f0a6545eb4dea9ce1665b193afaddcf97c486a6cd9b` | Perubahan non-core/dokumentasi |

Semua kelompok memiliki protocol hash dan manifest hash yang sama, training
worktree tercatat bersih, dan official test tidak diakses. Audit diff menunjukkan:

1. Kelompok pertama ke kedua menambahkan penyimpanan/pemulihan model, optimizer,
   scheduler, scaler AMP, epoch, dan RNG state. Jalur training dari awal tidak
   mengubah arsitektur, loss, optimizer, scheduler, transformasi, atau evaluator.
2. Kelompok kedua ke ketiga memperbaiki resolver dan snapshot konfigurasi agar
   resume dapat menggunakan direktori run yang telah ada; tidak ada perubahan
   rumus optimisasi.
3. Kelompok ketiga ke keempat tidak mengubah file core `main.py`, `config*`,
   `data/`, `models/`, `training/`, `evaluation/`, `utils/experiment.py`, atau
   `utils/reproducibility.py`.

**Kesimpulan attestation:** Job 1--8 dan 10--15 dapat diperlakukan ekuivalen
secara ilmiah terhadap kontrak eksperimen beku. Job 9 dapat diterima secara
bersyarat karena resume memulihkan seluruh training state yang dicatat, tetapi
belum ada uji A/B yang membuktikan kesetaraan bitwise dengan run tanpa interupsi.
Empat hash harus tetap dilaporkan, bukan diganti dengan satu hash retrospektif.

Attestation ini adalah audit teknis internal. Untuk menjadi attestation formal,
tambahkan nama reviewer independen, tanggal review, keputusan, tanda tangan atau
identitas digital, dan SHA-256 dokumen final.

## 8. Frontier Pareto Formal

Dimensi sesuai protokol adalah:

- maksimalkan mean AP `trash_plastic` multiseed;
- minimalkan median latency end-to-end;
- minimalkan peak reserved VRAM inferensi;
- bentuk frontier terpisah untuk CNN dan Transformer.

Kandidat A mendominasi B apabila A tidak lebih buruk pada semua dimensi dan
lebih baik secara ketat pada sedikitnya satu dimensi.

### 8.1 Frontier CNN

- **ResNet50 — Pareto:** latency lebih rendah daripada EfficientNet-B0.
- **EfficientNet-B0 — Pareto:** AP lebih tinggi, variasi AP lebih kecil, dan
  VRAM lebih rendah daripada ResNet50.
- **ConvNeXt-Tiny — non-Pareto:** didominasi ResNet50 karena AP lebih rendah,
  latency lebih tinggi, dan VRAM lebih besar. Selain itu satu dari tiga seed
  gagal secara numerik.

ResNet50 dan EfficientNet-B0 tidak saling mendominasi. Berdasarkan tie-break
yang dibekukan, selisih AP keduanya (**0,0051**) masih di bawah toleransi
ekuivalensi **0,01**, kemudian stabilitas menjadi pemutus. Karena SD AP
EfficientNet-B0 **0,0024** lebih rendah daripada ResNet50 **0,0146**,
**EfficientNet-B0 tetap dipilih**.

### 8.2 Frontier Transformer

- **PVTv2-B1 — Pareto dan terpilih.**
- **Swin-T — non-Pareto:** PVTv2-B1 mempunyai AP lebih tinggi, latency lebih
  rendah, dan peak reserved VRAM lebih rendah.

Selisih mean AP PVTv2-B1 terhadap Swin-T adalah **0,0145**, melampaui toleransi
ekuivalensi 0,01. Karena itu keputusan sudah ditentukan oleh metrik primer,
sementara hasil inferensi menguatkannya melalui dominasi Pareto penuh.

### 8.3 Keputusan akhir seleksi backbone

| Keluarga | Backbone terpilih | Dasar keputusan |
|---|---|---|
| CNN | **EfficientNet-B0** | Pareto; AP target lebih tinggi; stabilitas terbaik; VRAM terendah; menang tie-break beku |
| Transformer | **PVTv2-B1** | Mean AP tertinggi dan mendominasi Swin-T pada AP, latency, serta VRAM |

Pasangan **EfficientNet-B0 + PVTv2-B1** kini didukung oleh training multiseed,
quality-gated checkpoint reload, benchmark inferensi formal, dan analisis Pareto.

## 9. Deviasi, Keterbatasan, dan Kehati-hatian Interpretasi

1. Benchmark dijalankan saat `benchmark_git.git_worktree_clean=false`. Commit
   benchmark tercatat `3a33d9b7a92ec1f0d1bda5ce62b9d46f1435cae9`, tetapi perubahan lokal harus
   diarsipkan atau dibuatkan patch/commit untuk provenance yang sepenuhnya dapat
   direkonstruksi.
2. Kondisi daya dideklarasikan sebagai AC/high-performance, tetapi hasil tidak
   menyimpan time series temperatur, clock, dan power draw. Counterbalancing dan
   tiga repetisi mengurangi bias urutan, namun belum menggantikan telemetri termal.
3. Pada ringkasan agregat, model-only median beberapa run lebih tinggi daripada
   end-to-end median. Kedua jalur diukur melalui dua forward pass terpisah dalam
   satu loop, sehingga perbedaan cache/order dapat membias nilai absolut. Frontier
   memakai end-to-end sesuai protokol; model-only disajikan sebagai diagnostik,
   bukan dasar keputusan.
4. PVTv2-B1 seed 123 adalah hasil resume. Full state telah dipulihkan dan semua
   quality gate lulus, tetapi ekuivalensi bitwise terhadap run uninterrupted
   belum dibuktikan.
5. AP absolut masih rendah. Seleksi ini bersifat komparatif untuk backbone pada
   protokol dan data yang sama, bukan bukti bahwa sistem sudah siap diterapkan.
6. Official test set tetap belum boleh disentuh sampai model/hiperparameter final
   dibekukan.

Berdasarkan butir tersebut, status yang tepat adalah **benchmark tervalidasi
secara teknis dengan catatan provenance/telemetri**, bukan klaim benchmark lintas
perangkat yang universal.

## 10. Tindak Lanjut yang Direkomendasikan

1. Bekukan laporan ini bersama hash artefak JSON/CSV benchmark, patch atau commit
   kode benchmark, reviewer, dan tanggal keputusan.
2. Jika diperlukan untuk publikasi, ulangi satu sesi konfirmasi setelah worktree
   bersih sambil merekam temperatur, clock, dan power GPU; jangan mengubah
   checkpoint, precision, warm-up, urutan, atau split.
3. Jalankan analisis overlap TP/FP/FN EfficientNet-B0 dan PVTv2-B1 untuk menguji
   apakah kedua backbone benar-benar komplementer sebelum merancang fusi.
4. Bekukan pasangan terpilih dan aturan downstream; setelah itu barulah lakukan
   evaluasi official test satu kali sesuai protokol.

## 11. Pernyataan Hasil

Dengan bukti yang tersedia, tahapan benchmark inferensi formal, pemeriksaan
checksum/reload checkpoint, attestation teknis empat source tree, dan pembentukan
frontier Pareto telah selesai. Keputusan seleksi backbone akhir adalah
**EfficientNet-B0 untuk cabang CNN** dan **PVTv2-B1 untuk cabang Transformer**,
dengan catatan provenance dan telemetri pada Bagian 9 harus dibawa ke laporan
publikasi dan proses freeze final.
