# PRD Final: Klasifikasi Citra Digit Tulisan Tangan (Chars74k – Hnd) dengan CLAHE, HOG, PCA, dan XGBoost

## 1. Role & Tujuan Proyek

Bertindaklah sebagai Senior Machine Learning & Computer Vision Engineer. Bangun pipeline *end-to-end* yang sederhana, modular, efisien, dan bebas *data leakage* untuk mengklasifikasikan citra angka tulisan tangan (digit 0–9) dari subset **Chars74k – Handwritten (`Hnd`)** pada dataset **Numerical Images** (Kaggle), menggunakan **XGBoost** sebagai model klasifikasi utama.

Tujuan proyek:

1. Menggunakan dataset numeric image sesuai instruksi tugas, yaitu folder `archive/numbers/chars74k_png/Hnd`.
2. Mengurangi dimensi/fitur menggunakan **Principal Component Analysis (PCA)**.
3. Membandingkan performa XGBoost **tanpa PCA** dan **dengan PCA**, meliputi akurasi, metrik klasifikasi, jumlah fitur, serta waktu pelatihan.
4. Menyajikan hasil evaluasi dan visualisasi yang dapat digunakan dalam laporan akademik.
5. Menjaga proses eksperimen bebas *data leakage*.

> **Batasan cakupan:** Hanya subset `Hnd` Chars74k yang digunakan. MNIST, grup `Fnt`, `GoodImg`, `BadImg`, serta model Decision Tree dan Random Forest tidak digunakan.

---

## 2. Spesifikasi Dataset

- **Sumber:** https://www.kaggle.com/datasets/pintowar/numerical-images
- **Lokasi subset setelah ekstraksi:** `archive/numbers/chars74k_png/Hnd`
- **Isi:** citra digit tulisan tangan 0–9, umumnya berformat `.png`.
- **Label:** gunakan CSV metadata dataset sebagai sumber label utama apabila tersedia. Filter baris berdasarkan `origin == 'chars74k'` dan `group == 'Hnd'`, lalu gunakan kolom `file` dan `label`.
- **Validasi label:** pastikan label berhasil dipetakan menjadi integer `0` sampai `9`. Jangan menyimpulkan label hanya dari nama file kecuali format penamaan sudah diverifikasi.
- **Validasi file:** periksa keberadaan file, kemampuan membaca citra, ekstensi, dan duplikasi path.
- **Jumlah data:** jangan mengasumsikan jumlah gambar tertentu. Hitung jumlah aktual setelah proses validasi dan tampilkan distribusi per kelas.

### Data audit yang wajib ditampilkan

1. Total gambar yang ditemukan.
2. Total gambar valid dan gambar yang gagal dibaca.
3. Distribusi jumlah gambar per digit (0–9).
4. Jumlah data per kelas sebelum split.
5. Contoh citra dari setiap kelas.

---

## 3. Tech Stack & Library

- **Bahasa:** Python 3.10+
- **Pengolahan citra dan fitur:** `opencv-python`, `scikit-image` (`skimage.feature.hog`)
- **Data:** `numpy`, `pandas`, `tqdm`
- **Machine Learning:** `scikit-learn`, `xgboost`
- **Visualisasi:** `matplotlib`, `seaborn`
- **Penyimpanan model:** `joblib`

---

## 4. Arsitektur Pipeline & Spesifikasi Teknis

### Tahap 1: Data Loading, Validasi, dan Eksplorasi

1. Muat daftar gambar dan label dari subset `Hnd`, menggunakan CSV metadata sebagai sumber label utama.
2. Filter metadata untuk subset Chars74k `Hnd`.
3. Validasi:
   - path gambar tersedia;
   - gambar dapat dibaca;
   - label termasuk `0–9`;
   - path tidak terduplikasi.
4. Catat gambar yang gagal dibaca dan jangan memasukkannya ke eksperimen.
5. Cetak serta plot distribusi jumlah sampel per kelas (0–9).
6. Tampilkan beberapa contoh gambar untuk setiap digit.

### Tahap 2: Stratified Train-Test Split (Anti Data Leakage)

1. Pisahkan data menjadi **Train (80%)** dan **Test (20%)** menggunakan:
   ```python
   train_test_split(
       X_paths,
       y,
       test_size=0.2,
       stratify=y,
       random_state=42
   )
   ```
2. Split dilakukan sebelum augmentasi atau proses yang mempelajari distribusi data.
3. Data test harus tetap berisi citra asli dan tidak digunakan untuk pemilihan parameter, fitting scaler, fitting PCA, atau fitting model.
4. Jika jumlah sampel suatu kelas terlalu sedikit untuk stratifikasi, tampilkan pesan yang jelas dan hentikan proses atau minta penyesuaian split; jangan melakukan split diam-diam tanpa stratifikasi.

### Tahap 3: Preprocessing & Ekstraksi Fitur (CLAHE + HOG)

Terapkan proses yang sama pada seluruh gambar:

1. **Grayscale:** baca menggunakan `cv2.IMREAD_GRAYSCALE`.
2. **Resize:** ubah ukuran menjadi `28 × 28` piksel menggunakan `cv2.INTER_AREA`.
3. **CLAHE:** gunakan parameter:
   - `clipLimit=2.0`
   - `tileGridSize=(4, 4)`
4. **HOG:** ekstraksi menggunakan `skimage.feature.hog`:
   - `orientations=9`
   - `pixels_per_cell=(4, 4)`
   - `cells_per_block=(2, 2)`
   - `block_norm='L2-Hys'`
   - `visualize=False`
5. Hasil HOG diharapkan berupa vektor 1D **1.296 fitur** dengan tipe `np.float32` untuk setiap gambar.
6. Gunakan `tqdm` sebagai progress bar. Paralelisasi `joblib` bersifat opsional karena dataset Hnd relatif kecil.
7. Simpan fitur hasil ekstraksi agar proses eksperimen tidak perlu mengulang ekstraksi citra setiap kali model dilatih.

### Tahap 4: Eksperimen PCA (Inti Soal)

Bandingkan dua skenario fitur:

- **Skenario A — Tanpa PCA:** HOG → XGBoost.
- **Skenario B — Dengan PCA:** HOG → StandardScaler → PCA → XGBoost.

Ketentuan:

1. PCA menggunakan `n_components=0.95` untuk mempertahankan 95% variansi, dengan `random_state=42` jika solver yang digunakan mendukung parameter tersebut.
2. Catat:
   - jumlah fitur awal (1.296);
   - jumlah komponen PCA;
   - rasio reduksi dimensi;
   - cumulative explained variance.
3. **StandardScaler dan PCA hanya boleh di-fit pada data train.** Data test hanya dikenai `transform`.
4. Untuk eksperimen CV, gunakan `sklearn.pipeline.Pipeline` agar StandardScaler dan PCA di-fit ulang secara terpisah pada setiap fold. Jangan melakukan fitting PCA pada seluruh data train sebelum cross-validation.
5. StandardScaler tidak wajib untuk XGBoost berbasis pohon. Untuk eksperimen yang adil dan konsisten:
   - skenario tanpa PCA menggunakan fitur HOG asli;
   - skenario dengan PCA menggunakan StandardScaler + PCA.

   Jika ingin menguji pengaruh StandardScaler secara terpisah, jadikan eksperimen tambahan dan jelaskan dalam laporan.

### Tahap 5: Pelatihan Model XGBoost

Gunakan XGBoost sebagai satu-satunya algoritma klasifikasi pada dua skenario fitur di Tahap 4.

Konfigurasi awal:

```python
XGBClassifier(
    objective='multi:softprob',
    num_class=10,
    tree_method='hist',
    n_estimators=400,
    learning_rate=0.08,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42
)
```

Ketentuan pelatihan:

1. Catat waktu pelatihan setiap skenario.
2. Gunakan **Stratified 5-Fold Cross-Validation** pada data train untuk mengukur kestabilan performa.
3. Cross-validation harus menggunakan pipeline pada skenario PCA sehingga preprocessing yang memerlukan fitting dilakukan hanya pada bagian train dari setiap fold.
4. Laporkan rata-rata dan standar deviasi accuracy CV.
5. Setelah evaluasi CV selesai, latih ulang model pada seluruh data train dan evaluasi satu kali pada data test.
6. (Opsional) Catat kurva `mlogloss` train dan validasi untuk memeriksa overfitting.
7. Jika kelas terlalu sedikit untuk 5-fold CV, sesuaikan jumlah fold berdasarkan kelas dengan sampel paling sedikit dan jelaskan penyesuaian tersebut.

### Tahap 6: Evaluasi & Visualisasi

Evaluasi dilakukan pada **test set yang tidak digunakan selama training dan pemilihan model**.

1. **Tabel perbandingan dua skenario:**
   - XGBoost tanpa PCA;
   - XGBoost dengan PCA.

   Metrik:
   - Accuracy;
   - Macro Precision;
   - Macro Recall;
   - Macro F1-Score;
   - Training Time (detik);
   - jumlah fitur;
   - CV Accuracy mean ± standard deviation.

2. **Confusion Matrix (10 × 10):**
   - raw counts;
   - normalized (`normalize='true'`);
   - subplot `1 × 2` untuk membandingkan kedua skenario;
   - heatmap `seaborn.heatmap` dengan `annot=True` dan `cmap='Blues'`.

3. **Analisis kesalahan:**
   - ekstrak 5 pasangan digit yang paling sering tertukar dari elemen off-diagonal confusion matrix;
   - tampilkan label asli, prediksi, jumlah kesalahan, dan persentase terhadap kelas asli.

4. **Metrik per kelas:**
   - TP, FP, FN, TN untuk digit 0–9;
   - `classification_report` dengan Precision, Recall, F1-score, dan Support.

5. **Visualisasi tambahan:**
   - distribusi kelas sebelum augmentasi (dan sesudah augmentasi jika fitur augmentasi diaktifkan);
   - cumulative explained variance PCA;
   - contoh citra sebelum dan sesudah CLAHE;
   - visualisasi gradien HOG untuk beberapa digit;
   - (opsional) scatter plot komponen PCA 1 dan 2 dengan warna berdasarkan kelas.

6. Simpan semua plot ke `outputs/plots/` dengan `dpi=300` dan tabel ke `outputs/reports/` dalam format CSV.

### Tahap 7: Augmentasi (Opsional)

Augmentasi bukan persyaratan utama tugas dan secara default dinonaktifkan.

Jika diperlukan untuk eksperimen tambahan:

1. Split data terlebih dahulu.
2. Augmentasi hanya data train.
3. Gunakan transformasi ringan seperti rotasi, translasi, atau zoom yang tidak mengubah identitas digit.
4. Jangan melakukan augmentasi terhadap test set.
5. Laporkan distribusi kelas sebelum dan sesudah augmentasi.

---

## 5. Struktur Kode & Deliverables

```text
project_root/
├── data/
│   └── chars74k_png/
│       └── Hnd/                # Citra digit tulisan tangan
├── metadata/                    # CSV metadata bila tersedia
├── src/
│   ├── config.py               # Path, seed, dan hyperparameter
│   ├── data.py                 # Loading, validasi, audit, dan split stratified
│   ├── preprocess.py            # Grayscale, resize, CLAHE, ekstraksi HOG
│   ├── pipeline.py              # Pipeline fitur dan definisi XGBoost
│   ├── train.py                 # CV dan training final dua skenario
│   ├── evaluate.py               # Metrik, confusion matrix, analisis kesalahan
│   └── main.py                   # Orkestrasi end-to-end dan penyimpanan artefak
├── outputs/
│   ├── models/                   # Model/pipeline hasil training (.joblib)
│   ├── reports/                  # CSV hasil evaluasi dan distribusi
│   └── plots/                     # Heatmap, PCA, CLAHE, dan HOG
├── cache/                          # Cache fitur HOG (opsional)
└── requirements.txt
```

Alternatif: seluruh alur boleh dikemas dalam satu Jupyter Notebook apabila lebih sesuai untuk pengumpulan tugas.

---

## 6. Instruksi Eksekusi untuk AI

1. Tulis kode bersih, modular, menggunakan type hints dan docstrings.
2. Gunakan `np.float32` untuk fitur HOG jika memungkinkan.
3. Berikan logging waktu pada setiap tahap dan `tqdm` pada ekstraksi fitur.
4. Pastikan tidak ada data leakage:
   - split sebelum augmentasi;
   - scaler dan PCA hanya fit pada data train;
   - preprocessing dalam cross-validation dilakukan melalui `Pipeline`;
   - test set hanya digunakan untuk evaluasi akhir.
5. Validasi label dan jumlah sampel sebelum training.
6. Jangan menambahkan dataset lain atau model klasifikasi lain tanpa persetujuan.
7. Gunakan hanya subset `Hnd` Chars74k dan XGBoost sebagai model klasifikasi.
8. Mulailah dengan `requirements.txt` dan `src/config.py`, kemudian lanjutkan secara berurutan hingga `src/main.py` siap dijalankan.
9. Jelaskan cara menjalankan proyek pada Windows melalui terminal/PowerShell, termasuk instalasi dependensi dan perintah eksekusi.
10. Buat ringkasan hasil eksperimen yang dapat digunakan untuk penulisan laporan, tanpa mengarang nilai metrik sebelum training dijalankan.

---

## 7. Kriteria Keberhasilan

Proyek dianggap selesai apabila:

- Dataset yang dipakai benar-benar berasal dari `chars74k_png/Hnd`.
- Label 0–9 tervalidasi dan distribusi data terdokumentasi.
- Pipeline preprocessing CLAHE + HOG berjalan.
- PCA berhasil mengurangi jumlah fitur dan hasilnya tercatat.
- XGBoost dilatih dalam skenario tanpa PCA dan dengan PCA.
- Cross-validation berjalan tanpa data leakage.
- Evaluasi test set mencakup metrik klasifikasi dan confusion matrix.
- Waktu pelatihan dan jumlah fitur kedua skenario dapat dibandingkan.
- Model, tabel, dan visualisasi tersimpan pada folder output.
- Hasil akhir dapat dijelaskan secara objektif dalam laporan akademik.
