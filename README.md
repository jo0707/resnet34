# Eksperimen Arsitektur ResNet34 dan Plain34 - NgodingDiKertas

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

| Model    | Training Accuracy   | Validation Accuracy | Training Loss       | Validation Loss     |
| -------- | ------------------- | ------------------- | ------------------- | ------------------- |
| ResNet34 | _[isi: e.g., 0.94]_ | _[isi: e.g., 0.92]_ | _[isi: e.g., 0.18]_ | _[isi: e.g., 0.24]_ |
| Plain34  | _[isi: e.g., 0.90]_ | _[isi: e.g., 0.87]_ | _[isi: e.g., 0.28]_ | _[isi: e.g., 0.35]_ |

> Catatan: pastikan ini adalah metrik dari epoch terakhir yang sama atau terbaik (jelaskan kriteria jika memakai early stopping atau model terbaik berdasarkan val loss/accuracy).

---

## Kurva Training (Grafik)

Sisipkan grafik sederhana (screenshot atau gambar) yang menampilkan kurva training untuk kedua model (loss dan/atau accuracy terhadap epoch). Letakkan gambar pada folder `images/` atau lokasi lain dan sesuaikan path-nya.

-   ResNet34:

    ![Kurva Training - ResNet34](images/curves_resnet34.png)

-   Plain34:

    ![Kurva Training - Plain34](images/curves_plain34.png)

> Alternatif: gabungkan keduanya dalam satu grafik per metrik (misal, satu gambar untuk loss, satu gambar untuk accuracy) dan tuliskan keterangan yang jelas.

---

## Analisis Singkat

Tuliskan 2–3 paragraf analisis mengenai perbedaan performa dan dampak residual connection.

Paragraf 1 (perbandingan umum): _[contoh: ResNet34 cenderung mencapai akurasi validasi lebih tinggi serta konvergensi lebih stabil dibanding Plain34. Pada kurva training, ResNet34 menunjukkan penurunan loss yang lebih konsisten dan menghindari degradasi performa ketika kedalaman jaringan meningkat.]_

Paragraf 2 (dampak residual connection): _[contoh: Residual connection mempermudah aliran gradien ke lapisan awal (mitigasi vanishing gradients) sehingga training menjadi lebih dalam dan efektif. Hal ini membantu ResNet34 mempertahankan representasi fitur yang lebih kaya tanpa mengorbankan stabilitas optimisasi.]_

Paragraf 3 (implikasi praktis, opsional): _[contoh: Dalam skenario data serupa, ResNet34 direkomendasikan karena trade-off antara kompleksitas dan performa yang menguntungkan, sementara Plain34 dapat menjadi baseline atau digunakan saat sumber daya sangat terbatas.]_

---

## Konfigurasi Hyperparameter

Rinci konfigurasi yang digunakan untuk kedua eksperimen agar dapat direproduksi. Isi placeholder berikut sesuai setup Anda.

### Umum

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
-   Seed: _[misal 42]_

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
