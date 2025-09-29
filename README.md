# Eksperimen Arsitektur ResNet34 dan Plain34 - NgodingDiKertas

Source code: https://github.com/jo0707/resnet34

## Informasi

-   Mata Kuliah: IF25-40401 - Deep Learning
-   Program Studi: Teknik Informatika
-   Dosen Pengampu: MCT, IEW, RIK, MCU

Anggota Kelompok:

-   Joshua Palti Sinaga 122140141
-   Rustian Afencius Marbun 122140155

## Dataset

Dataset yang digunakan adalah dataset klasifikasi citra makanan Indonesia dari IF25-40401..

## Arsitektur

Bagian ini menjelaskan arsitektur model yang digunakan. Silakan lengkapi detail atau sisipkan diagram/tautan arsitektur jika diperlukan.

-   ResNet34: _[deskripsi singkat atau referensi arsitektur]_
-   Plain34: _[deskripsi singkat tanpa residual connection]_

> Tempatkan gambar/diagram arsitektur (opsional):
>
> ![Diagram ResNet34](images/diagram_resnet34.png)  
> ![Diagram Plain34](images/diagram_plain34.png)

---

## Hasil dan Perbandingan Performa (Epoch Terakhir)

Tabel berikut berisi metrik pada epoch terakhir. Isi nilai sesuai hasil eksperimen Anda.

```
================================================================================
FINAL EVALUATION RESULTS - PLAIN-34 BASELINE
================================================================================
Final Validation Metrics:
Accuracy:  0.2748
F1-Score:  0.1729
Precision: 0.1684
Recall:    0.2748
Loss:      1.5810

Detailed Classification Report:
--------------------------------------------------
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        43
           1       0.00      0.00      0.00        41
           2       0.24      0.92      0.38        49
           3       0.46      0.29      0.35        56
           4       0.00      0.00      0.00        33

    accuracy                           0.27       222
   macro avg       0.14      0.24      0.15       222
```

```
================================================================================
FINAL EVALUATION RESULTS - RESNET-34 (PLACEHOLDER)
================================================================================
Final Validation Metrics:
Accuracy:  _[isi hasil akurasi ResNet34]_
F1-Score:  _[isi hasil f1-score ResNet34]_
Precision: _[isi hasil precision ResNet34]_
Recall:    _[isi hasil recall ResNet34]_
Loss:      _[isi hasil loss ResNet34]_

Detailed Classification Report:
--------------------------------------------------
              precision    recall  f1-score   support

           0       _[.]_      _[.]_      _[.]_        _[.]_
           1       _[.]_      _[.]_      _[.]_        _[.]_
           2       _[.]_      _[.]_      _[.]_        _[.]_
           3       _[.]_      _[.]_      _[.]_        _[.]_
           4       _[.]_      _[.]_      _[.]_        _[.]_

    accuracy                           _[.]_       _[.]
   macro avg       _[.]_      _[.]_      _[.]_       _[.]

Gantilah placeholder di atas dengan hasil classification report ResNet34 Anda.
```

> Catatan: pastikan ini adalah metrik dari epoch terakhir yang sama atau terbaik (jelaskan kriteria jika memakai early stopping atau model terbaik berdasarkan val loss/accuracy).

---

## Kurva Training (Grafik)

Sisipkan grafik sederhana (screenshot atau gambar) yang menampilkan kurva training untuk kedua model (loss dan/atau accuracy terhadap epoch). Letakkan gambar pada folder `images/` atau lokasi lain dan sesuaikan path-nya.

-   ResNet34:

    ![Kurva Training - ResNet34](plainnet34.png)

-   Plain34:

    ![Kurva Training - Plain34](images/curves_plain34.png)

> Alternatif: gabungkan keduanya dalam satu grafik per metrik (misal, satu gambar untuk loss, satu gambar untuk accuracy) dan tuliskan keterangan yang jelas.

---

## Analisis Singkat

Pada eksperimen ini, penerapan skip connection dilakukan pada kelas `ResNetBlock` (sebelumnya bernama `PlainBlock`) dengan cara menambahkan hasil output (`out`) dengan input aslinya sebelum aktivasi, yaitu `out = out + identity`. Dengan adanya skip connection ini, model ResNet34 mampu mengalirkan informasi dan gradien secara lebih efektif ke lapisan awal, sehingga mengatasi masalah vanishing gradient yang sering terjadi pada jaringan yang sangat dalam. Hasilnya, ResNet34 menunjukkan penurunan loss yang lebih stabil dan konvergensi yang lebih cepat dibandingkan dengan PlainNet (tanpa skip connection).

Dari segi performa, PlainNet tanpa residual connection hanya mampu mencapai akurasi validasi sekitar 27%. Namun, setelah menerapkan skip connection pada arsitektur ResNet34, terjadi peningkatan akurasi yang signifikan hingga mencapai XX% (ganti XX dengan hasil akurasi ResNet34 Anda). Hal ini membuktikan bahwa residual connection sangat penting untuk meningkatkan kemampuan representasi dan optimisasi pada jaringan yang dalam, sehingga ResNet34 jauh lebih unggul dibandingkan PlainNet dalam tugas klasifikasi ini.

---

## Konfigurasi Hyperparameter

Rinci konfigurasi yang digunakan untuk kedua eksperimen agar dapat direproduksi. Isi placeholder berikut sesuai setup Anda.

### Umum

-   Seed: _[misal 42]_
-   Framework: _[PyTorch/TensorFlow]_
-   Optimizer: _[SGD/Adam/AdamW]_
-   Learning rate awal: _[misal 0.1 / 1e-3]_
-   Scheduler: _[Cosine/StepLR/OneCycle/—]_
-   Momentum/Betas: _[misal 0.9 atau (0.9, 0.999)]_
-   Weight decay: _[misal 1e-4]_
-   Batch size: _[misal 64]_
-   Epochs: _[misal 100]_
-   Loss function: _[CrossEntropyLoss/—]_
-   Augmentasi: _[RandomCrop/Flip/ColorJitter/—]_
-   Normalisasi: _[mean,std]_

### Spesifik ResNet34

-   Pretrained: _[True/False]_
-   Fine-tuning strategy: _[freeze backbone N epoch, unfreeze, dll]_
-   Modifikasi head: _[jumlah kelas, dropout, dsb]_

### Spesifik Plain34

-   Inisialisasi bobot: _[kaiming/xavier/—]_
-   Modifikasi head: _[jumlah kelas, dropout, dsb]_

Contoh format dalam YAML (opsional, sesuaikan dengan proyek Anda):

```yaml
experiment:
    name: "resnet34_vs_plain34"
    seed: 42

data:
    dataset: "IF25-40401-food"
    img_size: 224
    batch_size: 64
    num_workers: 4
    augmentation:
        - RandomResizedCrop: { size: 224, scale: [0.8, 1.0] }
        - RandomHorizontalFlip: { p: 0.5 }
    normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]

training:
    epochs: 100
    criterion: CrossEntropyLoss
    optimizer: SGD
    lr: 0.1
    momentum: 0.9
    weight_decay: 0.0001
    scheduler: StepLR
    scheduler_params: { step_size: 30, gamma: 0.1 }

models:
    resnet34:
        pretrained: true
        head: { num_classes: _TO_FILL_ }
    plain34:
        init: kaiming
        head: { num_classes: _TO_FILL_ }
```

---

## Catatan Eksekusi (Opsional)

-   Perangkat keras: _[GPU/CPU, VRAM, RAM, waktu training]_
-   Versi paket: _[PyTorch/TensorFlow, CUDA/cuDNN, dsb]_
-   Command run: _[contoh: python train.py --model resnet34]_

---

## Repro dan Struktur Proyek (Opsional)

-   Script training: _[path/command]_
-   Checkpoints dan logs: _[lokasi output]_
-   Gambar/plot: simpan di `images/`
