# Protokol Praregistrasi Seleksi Backbone Fase 3

**Status dokumen:** Aturan formal dan eksekusi pekerjaan 1--3 telah diimplementasikan; freeze formal harus dibuat ulang setelah perubahan di-commit
**Tanggal penyusunan:** 10 September 2026  
**Fokus:** Seleksi satu backbone CNN dan satu backbone Transformer untuk DAFNet  
**Metrik primer:** AP `trash_plastic` (category ID 14)

## 1. Tujuan dan Batas Eksperimen

Fase ini bertujuan memilih satu backbone CNN dan satu backbone Transformer melalui perbandingan cabang tunggal yang adil, reproducible, dan mempertimbangkan akurasi serta efisiensi. Seleksi tidak digunakan untuk mengoptimalkan modul DFF; CCA, SA, AWG, dan fusi adaptif belum digunakan pada tahap ini.

Model tetap dilatih sebagai detektor multikelas untuk 16 kelas objek. AP kelas `trash_plastic` menjadi kriteria utama, sedangkan performa seluruh kelas digunakan untuk memastikan fokus tersebut tidak merusak deteksi multikelas secara berlebihan.

Data val resmi berperan sebagai `locked_final_test` dan dilarang digunakan untuk memilih backbone, checkpoint, hiperparameter, threshold, adapter, atau aturan keputusan.

## 2. Protokol Data yang Dikunci

- Dataset: TrashCan 1.0 versi material.
- Anotasi sumber: `material_version/instances_train_trashcan.json`.
- Manifest development beku: `Riset/output/research_phase1/visual_review/web_app/rebuilds/candidate_10ab2a514764/visual_review/web_app/rebuilds/candidate_eb4dae55b2c4/visual_review/web_app/frozen/internal_split_manifest_eb4dae55b2c4afb4a2167448ad81b5cd8fed7bc7399ae57a36b27851300c2914.json`.
- Hash manifest: `eb4dae55b2c4afb4a2167448ad81b5cd8fed7bc7399ae57a36b27851300c2914`.
- `train_internal`: 4.800 citra.
- `val_internal`: 1.211 citra.
- Overlap citra dan grup: nol.
- Seluruh 16 kelas hadir pada validasi internal.
- Tiga putaran review visual selesai; putaran final menghasilkan 389/389 keputusan `safe`.

Eksperimen formal menggunakan seluruh data development. Opsi `--data-fraction` hanya boleh dipakai pada feasibility screening dan tidak boleh digunakan untuk menghasilkan tabel akurasi formal. Pemakaian 100% `train_internal` memastikan seluruh variasi kelas dan kondisi citra ikut menguji kapasitas backbone; subset 1% dan 10% hanya cukup untuk memeriksa kelayakan teknis.

## 3. Bukti Kesiapan Teknis

Lingkungan yang telah diverifikasi:

- Python 3.10.21 pada `.trashcan-env`.
- PyTorch `2.13.0+cu126`.
- Torchvision `0.28.0+cu126`.
- CUDA runtime 12.6 dan cuDNN 91002.
- NVIDIA GeForce RTX 3060 Laptop GPU 6GB.
- AMP FP16 aktif.
- Batch size aman awal: 2.
- 18 unit/integration test lulus.

Run `phase3_frozen_smoke_gpu_20260910` membuktikan jalur GPU 1% satu epoch selesai. Run `phase3_pilot_gpu_10pct_3epoch_20260910` membuktikan pilot 10% tiga epoch selesai dengan loss 1,1776; 0,7569; dan 0,6192. Nilai AP pilot tidak dipakai sebagai hasil seleksi karena anggarannya tidak formal dan hanya memakai satu seed.

## 4. Kandidat Backbone

Daftar kandidat berikut harus ditinjau dan dibekukan sebelum screening:

| Keluarga | Kandidat | Peran | Status sebelum freeze |
|---|---|---|---|
| CNN | ResNet50 | Baseline kontrol implementasi lama | Kandidat utama |
| CNN | EfficientNet-B0 | Kandidat CNN efisien | Kandidat utama |
| CNN | ConvNeXt-Tiny | Kandidat CNN modern | Aktif sebagai kandidat utama; lulus screening dan pilot |
| Transformer | Swin-T | Transformer hierarkis berbasis window | Kandidat utama |
| Transformer | PVTv2-B1 | Transformer pyramid untuk dense prediction | Kandidat utama |
| Transformer | Swin-V2-T | Kandidat cadangan keluarga Swin | Cadangan; lulus teknis, tidak masuk 15 run utama |

Untuk setiap kandidat harus dicatat sumber dan versi pretrained weights, lisensi, jumlah parameter, cara mengekstrak fitur multilevel, serta alasan ilmiah pemilihannya. Kandidat tambahan hanya boleh dimasukkan sebelum dokumen dibekukan atau melalui amendemen formal.

### 4.1 Justifikasi kandidat CNN

#### ResNet50 — baseline kontrol

ResNet50 dipertahankan karena pipeline saat ini telah menggunakan Mask R-CNN ResNet50-FPN. Stage residual C2–C5 dapat dipetakan langsung ke P2–P5, implementasi dan pretrained weights matang, serta perilakunya pada deteksi dan segmentasi telah dikenal luas. Residual learning diperkenalkan untuk mempermudah optimasi jaringan yang lebih dalam dan dilaporkan memberi peningkatan pada deteksi COCO ([He et al., 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)). Kandidat ini menjadi kontrol regresi: implementasi melalui registry baru harus mempertahankan kontrak data, head, evaluator, dan perilaku teknis baseline lama.

Kelemahan ResNet50 adalah usia arsitektur dan biaya komputasi yang dapat lebih tinggi daripada CNN efisien. Karena itu, ResNet50 tidak otomatis dipilih sebagai CNN final meskipun menjadi baseline wajib.

#### EfficientNet-B0 — kandidat efisien

EfficientNet-B0 dipilih untuk membentuk titik efisiensi pada frontier Pareto. Compound scaling menyeimbangkan kedalaman, lebar, dan resolusi untuk memperoleh kompromi akurasi–efisiensi yang lebih baik ([Tan dan Le, 2019](https://proceedings.mlr.press/v97/tan19a.html)), sedangkan varian B0 relatif ringan untuk GPU 6GB. Kandidat ini menguji apakah representasi CNN berbiaya rendah masih mampu mempertahankan sensitivitas terhadap objek plastik kecil dan kondisi bawah air yang sulit.

EfficientNet-B0 awalnya dirancang untuk klasifikasi. Pemilihan stage, kanal, dan adapter P2–P5 harus didokumentasikan; jumlah parameter yang kecil juga tidak boleh diasumsikan identik dengan latensi GPU yang rendah karena depthwise convolution perlu diukur pada perangkat aktual.

#### ConvNeXt-Tiny — kandidat CNN modern bersyarat

ConvNeXt-Tiny dipilih untuk membandingkan ResNet klasik dengan CNN modern yang mengadopsi sejumlah praktik desain era Transformer sambil mempertahankan operasi konvolusional. ConvNeXt dikembangkan melalui modernisasi bertahap atas ResNet dan dilaporkan kompetitif dengan Transformer pada klasifikasi, deteksi COCO, serta segmentasi ADE20K ([Liu et al., 2022a](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html)). Struktur hierarkisnya sesuai untuk fitur multilevel. Kandidat ini memperluas keragaman CNN dan dapat menunjukkan apakah peningkatan berasal dari attention Transformer atau dari modernisasi desain CNN.

ConvNeXt-Tiny berpotensi lebih mahal daripada EfficientNet-B0 pada VRAM, waktu training, dan ukuran checkpoint. Kandidat ini hanya diteruskan ke eksperimen formal jika screening membuktikan forward, backward, AMP, reload, dan benchmark berada dalam batas sumber daya yang dipraregistrasikan.

### 4.2 Justifikasi kandidat Transformer

#### Swin-T — Transformer hierarkis berbasis window

Swin-T menjadi kandidat utama karena membangun feature hierarchy melalui patch merging dan menggunakan shifted-window attention. Bentuk tersebut lebih alami untuk P2–P5 daripada vanilla ViT yang menghasilkan representasi single-scale. Window attention membatasi komputasi pada window lokal, sedangkan shifted windows memungkinkan hubungan lintas-window; desain hierarkis ini ditujukan untuk klasifikasi dan dense prediction ([Liu et al., 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html)). Karena itu, Swin-T lebih realistis daripada global-attention ViT untuk screening pada GPU 6GB.

Risikonya adalah sensitivitas terhadap ukuran window, divisibilitas resolusi, dan konsumsi aktivasi. Semua keputusan padding, pemilihan stage, serta adapter harus dibuat eksplisit dan sama antar-run.

#### PVTv2-B1 — Transformer pyramid untuk dense prediction

PVTv2-B1 dipilih karena keluarga PVT dirancang khusus untuk dense prediction melalui progressive shrinking pyramid dan spatial-reduction attention ([Wang et al., 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Pyramid_Vision_Transformer_A_Versatile_Backbone_for_Dense_Prediction_Without_ICCV_2021_paper.html)). PVTv2 menambahkan overlapping patch embedding, convolutional feed-forward network, dan linear-complexity attention untuk memperbaiki baseline tersebut ([Wang et al., 2022](https://arxiv.org/abs/2106.13797)). Kandidat ini memberikan keragaman mekanisme Transformer yang lebih kuat daripada hanya membandingkan dua versi Swin.

Varian B1 dipilih sebagai kompromi awal antara kapasitas dan batas VRAM. Risiko utamanya adalah PVTv2 tidak tersedia melalui jalur torchvision yang sama dengan ResNet, EfficientNet, ConvNeXt, dan Swin. Sumber implementasi, lisensi Apache 2.0 dari [repositori resmi PVT](https://github.com/whai362/PVT), hash kode, pretrained weights, preprocessing, dan pemetaan stage harus dicatat untuk menjaga reproducibility dan fairness.

#### Swin-V2-T — kandidat cadangan

Swin-V2-T mempertahankan feature hierarchy Swin serta menambahkan residual post-normalization, scaled cosine attention, dan log-spaced continuous relative position bias untuk meningkatkan stabilitas serta transfer antarresolusi ([Liu et al., 2022b](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.html)). Namun, Swin-V2-T sangat dekat secara keluarga dengan Swin-T sehingga nilai keragaman ilmiahnya lebih rendah daripada PVTv2-B1. Manfaat transfer resolusi tinggi juga belum tentu dominan pada input 576 x 320.

Swin-V2-T tidak masuk matriks formal utama secara otomatis. Kandidat ini diaktifkan apabila PVTv2-B1 gagal karena masalah implementasi atau reproducibility yang tidak dapat diselesaikan dalam batas yang dipraregistrasikan, atau apabila protokol secara eksplisit menyediakan anggaran tambahan untuk studi intra-keluarga Swin. Kegagalan PVTv2 karena performa akurasi rendah setelah hasil formal terlihat tidak boleh menjadi alasan menggantinya dengan Swin-V2-T.

### 4.3 Arsitektur yang tidak diprioritaskan

- Vanilla ViT tidak diprioritaskan karena keluarannya cenderung single-scale dan beresolusi rendah untuk dense prediction, sedangkan global attention menambah biaya komputasi; keterbatasan tersebut menjadi salah satu motivasi pengembangan PVT ([Wang et al., 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Pyramid_Vision_Transformer_A_Versatile_Backbone_for_Dense_Prediction_Without_ICCV_2021_paper.html)).
- ResNet18/34 terlalu dekat dengan baseline ResNet50 untuk pertanyaan utama, sedangkan ResNet101 berisiko menambah biaya tanpa memperluas keluarga arsitektur.
- MobileNetV3 dapat menjadi pembanding efisiensi ekstrem, tetapi tidak diprioritaskan karena penelitian belum menargetkan deployment mobile atau edge.
- YOLO, DETR, Faster R-CNN alternatif, dan RT-DETR diperlakukan sebagai detektor lengkap, bukan kandidat backbone yang adil untuk tabel ini.

### 4.4 Implementasi registry dan verifikasi sumber pretrained

Registry kandidat telah diimplementasikan pada `models/backbone_registry.py` dan dihubungkan ke CLI melalui `--backbone`. Semua kandidat menghasilkan fitur stride 4, 8, 16, dan 32 yang dipetakan oleh FPN ke 256 kanal. Adapter Swin menangani input dinamis 320 × 576 serta mengubah keluaran channels-last menjadi NCHW sebelum FPN.

Sumber implementasi dan weights yang diverifikasi adalah:

| Kandidat | Provider | Implementation/weights ID | SHA-256 body pretrained |
|---|---|---|---|
| ResNet50 | torchvision | `ResNet50_Weights.IMAGENET1K_V2` | `7412a3113342ea157019f5885b5739cd5316ed39f25208123fc4e8d7d284dfd3` |
| EfficientNet-B0 | timm 1.0.29 | `efficientnet_b0.ra_in1k` | `fd3edc67ee0aed753d14e38dfa85481fa5fd21d05fc0a535fb86ae033f13d11e` |
| ConvNeXt-Tiny | timm 1.0.29 | `convnext_tiny.fb_in1k` | `44a459116957faf9c0395353b451a9c1ba583150075010f1a75e89da0fe9e07f` |
| Swin-T | timm 1.0.29 | `swin_tiny_patch4_window7_224.ms_in1k` | `820e1f3a08119bfb20e93fbc05bc4b70a672bb3a39f1d07e021ccc61cc0b0739` |
| PVTv2-B1 | timm 1.0.29 | `pvt_v2_b1.in1k` | `9058ee77d89190fcc2f7243700fd01611f733964e8afe3f39834b64b3317ad9f` |
| Swin-V2-T | timm 1.0.29 | `swinv2_tiny_window8_256.ms_in1k` | `815307ee5845c3c2d056343df4780aa9cf37c4d586f1f18936f2a6c4fa4e4c2c` |

Kebijakan fairness dikunci pada `backbone_only`: hanya bagian backbone memakai weights ImageNet-1K. FPN, RPN, ROI heads, dan mask head selalu diinisialisasi acak; pretrained detector COCO tidak digunakan. Manifest mesin-baca berada di `Riset/configs/phase3_pretrained_weights_lock.json`, sedangkan bukti pemuatan weights, bentuk tensor, jumlah parameter, dan hash berada di `Riset/output/research_phase3/pretrained_weight_audit.json`. Audit aktual lulus 6/6.

Status manifest adalah `verified_not_frozen`: kandidat, provider, ID weights, versi dependensi, dan hash sudah terverifikasi; status baru boleh diubah menjadi `frozen` setelah commit Git, hash final protokol, waktu, dan reviewer dicatat. Dengan demikian, verifikasi teknis tidak disalahartikan sebagai praregistrasi final.

Perintah reproduksi:

```powershell
.trashcan-env\python.exe scripts\audit_phase3_backbones.py --pretrained
.trashcan-env\python.exe main.py --backbone pvtv2_b1 --dry-run --run-id <run_id>
```

### 4.5 Referensi ilmiah utama

- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016: https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
- Tan dan Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019: https://proceedings.mlr.press/v97/tan19a.html
- Liu et al., *A ConvNet for the 2020s*, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html
- Liu et al., *Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows*, ICCV 2021: https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html
- Wang et al., *Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction Without Convolutions*, ICCV 2021: https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Pyramid_Vision_Transformer_A_Versatile_Backbone_for_Dense_Prediction_Without_ICCV_2021_paper.html
- Wang et al., *PVTv2: Improved Baselines with Pyramid Vision Transformer*, 2022, implementasi resmi: https://github.com/whai362/PVT
- Liu et al., *Swin Transformer V2: Scaling Up Capacity and Resolution*, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.html

## 5. Kontrak Arsitektur yang Dikontrol

Semua kandidat mengikuti jalur:

```text
input -> backbone -> adapter terkontrol -> P2/P3/P4/P5 -> FPN/head Mask R-CNN yang sama
```

Kontrak minimum:

- Fitur `P2`, `P3`, `P4`, dan `P5` tersedia.
- Stride spasial masing-masing 4, 8, 16, dan 32 terhadap input.
- Adapter memetakan setiap level ke 256 kanal.
- Detection head, RPN, anchor, RoIAlign, predictor, evaluator, dan format prediksi identik.
- Jumlah keluaran model adalah 17: 16 kelas objek ditambah background.
- Transformasi citra, augmentasi, normalisasi, resolusi, dan aturan sampling identik.
- Nama backbone dan konfigurasi adapter tersimpan dalam resolved config, metadata run, dan checkpoint.

Perbedaan yang hanya diperlukan untuk menghasilkan kontrak fitur dicatat sebagai bagian adapter, bukan sebagai peningkatan bebas pada detection head.

## 6. Konfigurasi Eksperimen Formal

Kontrak berikut telah ditetapkan dalam konfigurasi mesin-baca. Status `Siap dibekukan` berarti nilainya sudah diterapkan dan diuji, tetapi metadata freeze formal masih menunggu commit, hash dokumen, waktu, dan reviewer:

| Komponen | Nilai ditetapkan | Status |
|---|---|---|
| Resolusi input | 576 x 320 | Siap dibekukan |
| Batch size | 2 | Siap dibekukan |
| Precision | AMP FP16 | Siap dibekukan |
| Optimizer | SGD | Siap dibekukan |
| Learning rate | 0,001 | Siap dibekukan |
| Momentum | 0,9 | Siap dibekukan |
| Weight decay | 0,0005 | Siap dibekukan |
| Scheduler | StepLR, step size 10, gamma 0,1 | Siap dibekukan |
| Seed formal | 42, 123, 2026 | Siap dibekukan |
| Data formal | 100% development | Dikunci |
| Checkpoint utama | AP `trash_plastic` terbaik pada `val_internal` | Dikunci |
| Jumlah epoch formal | 30 | Siap dibekukan |
| Validasi | Setiap 3 epoch dan pada epoch terakhir | Siap dibekukan |
| Early stopping | Dinonaktifkan agar anggaran setara | Siap dibekukan |
| Batas VRAM | 5,5 GiB; OOM menjadi `resource_infeasible` | Siap dibekukan |

Konfigurasi yang berbeda per backbone hanya diperbolehkan bila secara teknis tidak dapat dihindari, telah dipraregistrasikan, dan dilaporkan sebagai potensi pengganggu fairness.

Justifikasi ringkas nilai formal:

- **100% data development** dipakai agar ranking tidak dibentuk oleh subset kecil yang mungkin tidak mewakili kelas, ukuran objek, dan kondisi bawah laut.
- **30 epoch** merupakan anggaran tetap yang memberi waktu lebih memadai untuk melihat pola konvergensi daripada pilot tiga epoch, tanpa mengklaim sebagai epoch optimal universal.
- **Validasi setiap tiga epoch** menghasilkan 10 titik evaluasi per run dengan biaya evaluasi yang masih terkendali.
- **Tiga seed (42, 123, 2026)** mengukur variasi akibat inisialisasi dan urutan sampling; jumlah ini mendukung ringkasan deskriptif, bukan klaim signifikansi inferensial.
- **Early stopping dinonaktifkan** agar seluruh backbone menerima anggaran optimasi yang sama dan tidak mengalami bias akibat dinamika konvergensi yang berbeda.
- **Batch 2 dan akumulasi gradien 2** mempertahankan beban GPU yang layak pada RTX 3060 6 GB sekaligus memberikan effective batch size 4.
- **Batas peak reserved VRAM 5,5 GiB** menyisakan margin operasional pada GPU 6 GB; OOM tidak diatasi dengan konfigurasi khusus per backbone karena akan merusak kesetaraan.

Implementasi kontrak:

- sumber kebenaran: `Riset/configs/phase3_training_protocol.json`;
- loader dan validator: `utils/phase3_protocol.py`;
- `main.py --phase3-stage smoke|pilot|formal` menerapkan nilai protokol dan mengabaikan override konfigurasi inti;
- `--seed` pada tahap formal hanya menerima 42, 123, atau 2026;
- hash file protokol dicatat pada `resolved_config.json` dan checkpoint;
- detection head mengambil anchor, RoIAlign, threshold IoU, NMS, proposal limit, dan jumlah deteksi langsung dari kontrak;
- gradient accumulation tidak lagi di-hardcode pada trainer;
- `utils/phase3_formal_preflight.py` memverifikasi freeze dan sidecar SHA-256, hash seluruh input, manifest, kandidat, seed, official-test lock, ancestry commit, serta kebersihan worktree sebelum stage `formal` dimulai;
- `evaluation/detection_evaluator.py` menghasilkan `critical_false_negatives.json` dan `.csv` pada setiap evaluasi formal;
- `scripts/run_phase3_formal_multiseed.ps1` menjalankan matriks lima backbone x tiga seed secara berurutan, menghentikan eksekusi pada kegagalan pertama, dan mendukung resume hanya untuk run yang telah selesai secara valid.

Perintah pemeriksaan tanpa training:

```powershell
.trashcan-env\python.exe main.py --mode train --config rtx3060 --backbone pvtv2_b1 --phase3-stage smoke --manifest-path "<manifest-beku.json>" --run-id <run_id> --dry-run
```


## 7. Feasibility Screening

Screening menilai kelayakan teknis dan tidak menentukan pemenang akurasi.

### 7.1 Pemeriksaan kontrak tensor

Untuk setiap kandidat:

- lakukan forward satu batch;
- verifikasi bentuk, kanal, dan stride P2–P5;
- lakukan backward dan verifikasi gradien mencapai backbone serta adapter;
- verifikasi AMP FP16;
- pastikan loss dan gradien tidak mengandung NaN/Inf;
- simpan dan muat ulang checkpoint;
- jalankan inference dan evaluator;
- catat parameter, peak VRAM, dan waktu iterasi.

### 7.2 Smoke 1%

- Data development 1%.
- Satu epoch.
- Seed 42.
- Batch size dan resolusi sesuai protokol.
- Kriteria lulus: exit code 0, status `completed`, tidak OOM, tidak NaN/Inf, evaluator selesai, serta checkpoint dapat dimuat ulang.

### 7.3 Pilot 10%

- Data development 10%.
- Tiga epoch.
- Seed 42.
- Evaluasi mengikuti frekuensi yang sama untuk semua kandidat.
- Kriteria lulus: loss tidak divergen, tidak OOM, artefak lengkap, peak VRAM dan durasi berada dalam batas praregistrasi.

Kandidat yang gagal karena bug adapter diberi status `implementation_failed`. Kandidat yang secara benar melampaui anggaran perangkat diberi status `resource_infeasible`. Keduanya tidak boleh dinyatakan kalah secara akurasi.

### 7.4 Hasil smoke feasibility enam backbone

Screening smoke dan benchmark training telah selesai pada RTX 3060 Laptop GPU. Keenam kandidat berstatus `completed`, memakai AMP FP16, menghasilkan loss terbatas (bukan NaN/Inf), dapat memuat ulang checkpoint, serta menyelesaikan inference dan evaluasi tanpa OOM. Seluruh kandidat berada jauh di bawah batas peak allocated VRAM 5,5 GiB.

| Backbone | Iterasi | Mean train/batch (detik) | P95 (detik) | Batch/detik | Peak allocated (MB) | Keputusan smoke |
|---|---:|---:|---:|---:|---:|---|
| ResNet50 | 33 | 0,2671 | 0,3051 | 3,743 | 952,0 | Lulus |
| EfficientNet-B0 | 33 | 0,2636 | 0,2936 | 3,793 | 680,5 | Lulus |
| ConvNeXt-Tiny | 33 | 0,2574 | 0,2895 | 3,885 | 949,7 | Lulus; status kondisional diteruskan ke pilot |
| Swin-T | 33 | 0,3232 | 0,3627 | 3,094 | 1.059,8 | Lulus |
| PVTv2-B1 | 33 | 0,2838 | 0,3152 | 3,524 | 800,6 | Lulus |
| Swin-V2-T | 33 | 0,3833 | 0,4322 | 2,609 | 1.203,2 | Lulus; tetap kandidat cadangan hingga keputusan aktivasi |

Batas benchmark adalah maksimum 50 iterasi, tetapi loader smoke 1% hanya memiliki 33 batch. Karena keenam kandidat memproses 33 batch yang sama, hasil tetap sebanding untuk pemeriksaan kelayakan teknis. Benchmark inference juga selesai untuk 6/6 kandidat. Namun, checkpoint smoke menghasilkan jumlah deteksi berbeda (0 pada empat kandidat dan 100 pada PVTv2-B1 serta Swin-V2-T); karena itu latensi inference pada tahap ini hanya membuktikan jalur eksekusi dan belum boleh dipakai untuk pemeringkatan efisiensi final.

Keputusan: keenam backbone diizinkan masuk pilot 10% tiga epoch. Hasil smoke tidak digunakan untuk memilih pemenang akurasi. Artefak agregat tersedia pada `Riset/output/research_phase3/feasibility_screening_summary.json` dan `.csv`.

### 7.5 Hasil pilot enam backbone

Pilot ulang yang menjadi dasar keputusan menggunakan tag `20260912_v2`, stage
`pilot`, 10% data development, tiga epoch, seed 42, konfigurasi `rtx3060`,
dan manifest beku. Keenam run berstatus `completed` dengan `exit_code=0`,
memiliki tiga loss finite, evaluasi pada 137 citra validasi dengan 38 GT
`trash_plastic`, checkpoint lengkap, serta hash protokol dan manifest yang
konsisten.

| Backbone | Loss epoch 1 | Loss epoch 3 | Durasi total (menit) | Peak allocated (MB) | Peak reserved (MB) | Keputusan teknis |
|---|---:|---:|---:|---:|---:|---|
| ResNet50 | 1,3005 | 0,8353 | 4,02 | 1.377,7 | 1.580 | Lulus |
| EfficientNet-B0 | 2,0238 | 0,8141 | 5,53 | 967,0 | 1.534 | Lulus |
| ConvNeXt-Tiny | 3,9137 | 0,9554 | 5,32 | 1.395,7 | 1.872 | Lulus; syarat kandidat kondisional terpenuhi |
| Swin-T | 1,5077 | 0,9048 | 4,43 | 1.499,3 | 1.708 | Lulus |
| PVTv2-B1 | 1,5144 | 0,9055 | 4,19 | 1.143,4 | 1.376 | Lulus |
| Swin-V2-T | 4,6042 | 4,3336 | 6,27 | 1.624,6 | 1.956 | Lulus teknis; risiko konvergensi dicatat |

Seluruh peak reserved VRAM berada di bawah batas 5,5 GiB. AP pilot hanya
diagnostik dan tidak digunakan untuk menentukan pemenang karena pilot memakai
10% data, tiga epoch, dan satu seed. Artefak agregat final tersedia pada
`Riset/output/research_phase3/phase3_pilot_final_summary_20260912_v2.json`
dan `.csv`.

### 7.6 Freeze aktivasi backbone

Keputusan aktivasi telah dibekukan oleh Didik Kurniawan pada
11 September 2026 melalui
`Riset/output/research_phase3/backbone_activation_freeze.json`. Lima kandidat
utama adalah ResNet50, EfficientNet-B0, ConvNeXt-Tiny, Swin-T, dan PVTv2-B1.
Swin-V2-T tetap sebagai kandidat cadangan. Dengan tiga seed formal, matriks
utama berjumlah 15 run.

| Artefak/identitas | Nilai |
|---|---|
| Status | `frozen` |
| Tag pilot | `20260912_v2` |
| Hash freeze | `6152c0164d529bd0fe4bb85c9019ae23763d66880725738c3678cc726e732e83` |
| Hash summary pilot | `d9bb7987ca85b441d2187af885a7055f8a9726672df8bf473d4ea37298d10a04` |
| Hash protokol eksekusi pilot | `c675830fe00c38dd7e38c68de9fe46f094827f08a613203003d153118fdf431a` |
| Hash manifest | `eb4dae55b2c4afb4a2167448ad81b5cd8fed7bc7399ae57a36b27851300c2914` |
| Commit sumber freeze | `6be92177487c68708661143bc5992bebdded0a14` |
| Commit artefak freeze | `98a3cd3f0e22044eab6aeb15251f80e1cb9170c7` |
| Official test diakses | Tidak |

Hash freeze dan hash summary telah diverifikasi cocok dengan berkas aktual,
commit sumber merupakan ancestor commit freeze, dan working tree bersih saat
verifikasi. Metadata `pilot_runs_were_dirty=true` dipertahankan sebagai
keterbatasan transparan; eksperimen formal wajib dijalankan dari commit dan
working tree bersih. Freeze ini hanya mengunci aktivasi kandidat, bukan
membekukan seluruh protokol eksperimen formal.

## 8. Eksperimen Formal Multiseed

Hanya kandidat yang lulus screening yang diteruskan. Matriks utama memuat lima kandidat dan tiga seed sehingga rencana maksimum utamanya adalah 15 run formal. ConvNeXt-Tiny hanya masuk jika lulus screening. Swin-V2-T adalah cadangan dan tidak dihitung dalam 15 run utama; jika diaktifkan, maksimum menjadi 18 run dan alasan aktivasinya harus dicatat sebelum hasil formal dibandingkan.

| Backbone | Seed 42 | Seed 123 | Seed 2026 |
|---|---|---|---|
| ResNet50 | Direncanakan | Direncanakan | Direncanakan |
| EfficientNet-B0 | Direncanakan | Direncanakan | Direncanakan |
| ConvNeXt-Tiny | Direncanakan; lulus screening | Direncanakan; lulus screening | Direncanakan; lulus screening |
| Swin-T | Direncanakan | Direncanakan | Direncanakan |
| PVTv2-B1 | Direncanakan | Direncanakan | Direncanakan |
| Swin-V2-T | Cadangan | Cadangan | Cadangan |

Setiap run wajib:

- memakai 100% `train_internal` dan mengevaluasi `val_internal`;
- lulus formal preflight sebelum direktori run dibuat;
- tidak membuka official test;
- menggunakan run ID unik yang memuat backbone dan seed;
- menyimpan resolved config, environment lock, status, checkpoint, history, prediksi, dan metrik;
- memilih checkpoint berdasarkan AP `trash_plastic` pada validasi internal;
- merekam hash commit, source tree, manifest, dan checkpoint;
- menghasilkan 10 artefak evaluasi critical FN sesuai jadwal validasi 30/3 epoch.

Format run ID:

```text
phase3_formal_<backbone>_seed<seed>_<tanggal>
```

## 9. Metrik dan Benchmark

Metrik primer:

- AP IoU 0,50:0,95 untuk `trash_plastic`.

Metrik sekunder:

- AP50, AP75, dan AR100 `trash_plastic`;
- jumlah serta karakter FP/FN `trash_plastic`;
- overall bbox AP, AP50, AP75, dan AR100;
- AP per kelas untuk pemeriksaan kerusakan performa multikelas;
- rata-rata, simpangan baku, minimum, dan maksimum antar-seed.

Indikator efisiensi:

- jumlah parameter;
- FLOPs/MACs dengan definisi input yang sama;
- ukuran checkpoint;
- peak VRAM training dan inference;
- durasi training per epoch;
- latensi inference dan throughput pada perangkat yang sama.

Benchmark formal ditetapkan sebagai single-stream batch 1 pada seluruh `val_internal`,
input 576 x 320, AMP FP16, TF32 nonaktif, 50 warm-up, dan tiga repetisi.
Urutan model dibuat counterbalanced. Waktu CUDA disinkronkan dan dilaporkan
terpisah sebagai latensi model-only serta end-to-end tanpa file I/O. Statistik
minimum mencakup mean, median, simpangan baku sampel, P90, P95, P99, dan
throughput. Benchmark hanya sah jika quality gate evaluasi terpenuhi; model
dengan keluaran nol tidak boleh dinyatakan paling efisien tanpa catatan kualitas.

Metadata wajib mencakup ID run/backbone, hash checkpoint, manifest dan protokol,
commit sumber serta status worktree, versi Python/PyTorch/torchvision/CUDA/cuDNN,
GPU dan driver, kondisi daya, bentuk input, precision, batch, warm-up, jumlah
citra terukur, nomor repetisi, urutan model, serta timestamp UTC awal/akhir.

## 10. Aturan Pareto dan Keputusan

Frontier Pareto dibentuk terpisah untuk CNN dan Transformer. Dimensi yang
dimaksimalkan adalah mean AP `trash_plastic`; dimensi yang diminimalkan adalah
peak reserved VRAM dan median latensi end-to-end. Kandidat A mendominasi B jika
A tidak lebih buruk pada seluruh dimensi tersebut dan unggul ketat pada
setidaknya satu dimensi. Pemenang wajib berasal dari frontier Pareto keluarga masing-masing.

Unit agregasi adalah satu training seed independen. Ketiga seed formal
42, 123, dan 2026 dipasangkan antar-backbone. Untuk setiap backbone dilaporkan
nilai per-seed, mean, simpangan baku sampel, minimum, maksimum, serta jumlah
kemenangan berpasangan. Dengan hanya tiga seed, hasil digunakan secara
deskriptif dan tidak untuk klaim signifikansi inferensial.

Dua kandidat diperlakukan setara secara praktis bila selisih mean AP absolut
tidak lebih dari 0,01. Urutan keputusan yang dibekukan adalah:

1. mean AP `trash_plastic`;
2. simpangan baku AP yang lebih rendah jika berada dalam toleransi 0,01;
3. critical false-negative rate yang lebih rendah;
4. AP75 `trash_plastic` yang lebih tinggi;
5. peak reserved VRAM yang lebih rendah;
6. median latensi end-to-end yang lebih rendah;
7. jumlah parameter yang lebih rendah;
8. implementasi yang lebih sederhana dan dapat direproduksi.

FN kritis adalah GT `trash_plastic` area medium/large menurut rentang area COCO pada skala evaluasi yang tidak memiliki
prediksi cocok satu-ke-satu pada IoU >= 0,50 dengan score >= 0,05. Jumlah dan rate terhadap
seluruh GT medium/large wajib dilaporkan. Area kecil dikeluarkan agar indikator
berfokus pada objek yang secara operasional cukup terlihat; IoU 0,50 menyatakan
lokalisasi minimum yang dapat diterima dan score 0,05 mengikuti ambang operasional
model. Pencocokan greedy berdasarkan skor mencegah satu prediksi menghitung lebih
dari satu GT. Skor komposit berbobot dan pemeringkatan akurasi berdasarkan
smoke/pilot dilarang.

Swin-V2-T hanya boleh diaktifkan sebagai pengganti kandidat Transformer akibat
kegagalan implementasi atau reproduktibilitas yang sudah didefinisikan, sebelum
perbandingan hasil formal, dengan alasan dan reviewer tercatat. Jika lebih dari
satu kandidat tetap berada pada Pareto, uji kompatibilitas pasangan boleh
dilakukan menggunakan alignment dan fusi tetap 0,5/0,5. DFF, CCA, SA, dan AWG
dilarang digunakan untuk memilih backbone.

## 11. Artefak Wajib

Setiap run menyimpan:

```text
Riset/output/runs/<run_id>/
|-- resolved_config.json
|-- environment_lock.json
|-- run_metadata.json
|-- run_status.json
|-- checkpoints/
|-- metrics/
|-- benchmarks/
`-- training_history.png
```

Agregasi Fase 3 harus menghasilkan:

- tabel hasil setiap seed;
- tabel mean dan simpangan baku setiap backbone;
- tabel feasibility screening;
- frontier Pareto CNN dan Transformer;
- analisis FP/FN dan komplementaritas kesalahan;
- keputusan satu CNN dan satu Transformer;
- hash protokol, config, manifest, checkpoint, dan commit sumber.

## 12. Penguncian dan Amendemen

Record freeze formal versi awal pernah dibuat pada commit sumber
`58d1583a5e61b1aec53addbcd466cac210c14377`, tetapi belum mencakup implementasi
preflight, critical FN, runner multiseed, dan perubahan dokumen ini. Karena
protokol mewajibkan source tree serta hash input identik, record tersebut tidak
boleh dipakai untuk memulai run formal. Sebelum run formal pertama:

1. Tinjau aturan formal di `Riset/configs/phase3_training_protocol.json`.
2. Jalankan `scripts/freeze_phase3_formal_protocol.ps1 -ValidateOnly`.
3. Pastikan registry serta seluruh uji kontrak lulus dan official test tetap terkunci.
4. Commit kode dan dokumen tanpa checkpoint `.pth`, lalu pastikan worktree bersih.
5. Jalankan `scripts/freeze_phase3_formal_protocol.ps1 -Reviewer "<nama>" -Force` untuk mengganti record lama secara sadar.
6. Commit record JSON dan sidecar SHA-256 hasil pembekuan ulang.
7. Pastikan worktree bersih dan jalankan `scripts/validate_phase3_formal_preflight.py`.
8. Jalankan `scripts/run_phase3_formal_multiseed.ps1 -PlanOnly` sebelum eksekusi aktual.

Setiap perubahan setelah freeze harus dicatat pada tabel berikut dan tidak boleh menghapus keputusan lama.

| Waktu | Bagian | Nilai lama | Nilai baru | Alasan | Dampak pada run |
|---|---|---|---|---|---|
| - | - | - | - | Belum ada amendemen | - |

## 13. Checklist Sebelum Run Formal

- [x] Manifest development dibekukan dan diverifikasi.
- [x] Review visual selesai.
- [x] Smoke GPU, feasibility screening, dan pilot 10% tiga epoch selesai untuk enam backbone.
- [x] Kandidat, provider, ID weights, versi, dan hash body pretrained diverifikasi serta tercatat pada freeze awal; pembekuan ulang menunggu commit implementasi terbaru.
- [x] Registry backbone serta adapter P2–P5 selesai; audit kontrak dan pretrained lulus 6/6.
- [x] Detection head dan konfigurasi training ditetapkan, diterapkan, dan lulus validasi; metadata harus dicatat kembali pada freeze terbaru.
- [x] Epoch, validasi, early stopping, dan batas VRAM ditetapkan dalam konfigurasi mesin-baca.
- [x] Seed, agregasi, toleransi AP 0,01, FN kritis, Pareto, dan tie-break telah ditetapkan serta divalidasi mesin.
- [x] Benchmark smoke selesai; kontrak benchmark formal dan metadata wajib telah ditetapkan serta divalidasi mesin.
- [x] Uji kontrak, training smoke, checkpoint reload, inference, dan evaluasi seluruh kandidat lulus.
- [x] Preflight formal, evaluator critical FN, dan runner berurutan 15 run telah diimplementasikan serta diuji secara terarah.
- [ ] Perubahan pekerjaan 1--3 dan dokumen telah di-commit pada worktree bersih.
- [ ] Record freeze formal dan sidecar SHA-256 telah dibuat ulang dari commit implementasi terbaru.
- [ ] Preflight aktual tanpa `--allow-dirty` lulus sebelum run formal pertama.
- [x] Official test tetap tidak diakses; ditegaskan dalam freeze aktivasi.

Run seleksi backbone formal hanya boleh dimulai setelah semua item yang relevan bertanda selesai.
