# Klasifikasi Digit Tulisan Tangan Chars74k-Hnd

Pipeline end-to-end untuk mengklasifikasikan digit **0–9** dari subset
**Chars74k – Handwritten (`Hnd`)** menggunakan **CLAHE + HOG**, membandingkan
**XGBoost tanpa PCA** dan **XGBoost dengan PCA**.

Seluruh konfigurasi mengikuti `PRD.md`. Dataset yang dipakai adalah
`archive/numbers/chars74k_png/Hnd` dari Kaggle
[`pintowar/numerical-images`](https://www.kaggle.com/datasets/pintowar/numerical-images).

---

## 1. Ringkasan Hasil (run aktual)

Dataset: **550 citra valid**, tepat **55 citra per digit** (seimbang).
Split stratified 80/20: **440 train / 110 test**, 5-fold stratified CV.

| Skenario | Accuracy | Macro Precision | Macro Recall | Macro F1 | Training Time (s) | Jumlah Fitur | CV Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost tanpa PCA | **0.8727** | 0.8794 | 0.8727 | 0.8732 | 17.62 | **1.296** | 0.7614 ± 0.0394 |
| XGBoost dengan PCA | **0.7182** | 0.7198 | 0.7182 | 0.7158 | 4.46 | **157** | 0.6818 ± 0.0425 |

PCA: **157 komponen** dari 1.296 fitur (**reduksi 87,9%**) dengan
**cumulative explained variance 95,07%**.

**Interpretasi singkat:** PCA memangkas dimensi dan waktu training secara
signifikan, tetapi pada eksperimen ini menurunkan akurasi test sekitar
**15,5 poin persentase**. XGBoost tanpa PCA menjadi model terbaik pada test set.
Perbedaan accuracy CV (≈76%) dan test (≈87%) perlu dibahas di laporan sebagai
konsekuensi ukuran test yang kecil (110 citra; 11 per kelas).

> Semua angka di atas berasal dari `outputs/reports/metrics_comparison.csv`
> hasil eksekusi, bukan estimasi.

---

## 2. Struktur Proyek

```text
project_root/
├── archive/
│   ├── numbers.csv                  # metadata Kaggle
│   └── numbers/chars74k_png/Hnd/    # 550 citra digit tulisan tangan
├── src/
│   ├── config.py                    # path, seed, CLAHE/HOG/PCA/XGBoost
│   ├── data.py                      # loading, validasi, audit, split
│   ├── preprocess.py                # grayscale, resize, CLAHE, HOG, cache
│   ├── pipeline.py                  # Pipeline no-PCA dan PCA
│   ├── train.py                     # CV, training final, learning curve
│   ├── evaluate.py                  # metrik, confusion matrix, error analysis
│   └── main.py                      # orkestrasi end-to-end
├── outputs/
│   ├── models/                      # pipeline .joblib
│   ├── reports/                     # CSV/JSON
│   └── plots/                       # PNG dpi=300
├── cache/                           # cache fitur HOG (.npz)
├── requirements.txt
└── README.md
```

---

## 3. Instalasi (Windows / PowerShell)

Prasyarat: **Python 3.10+** (proyek ini diuji dengan Python 3.10.21).
Jalankan dari root proyek:

```powershell
# 1. Buat virtual environment
python -m venv .venv

# 2. Aktifkan
.\.venv\Scripts\Activate.ps1

# Jika PowerShell memblokir aktivasi:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Install dependensi
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Alternatif tanpa aktivasi:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 4. Menyiapkan Dataset

Struktur yang diharapkan:

```text
archive/
├── numbers.csv
└── numbers/
    └── chars74k_png/
        └── Hnd/
            ├── Sample0/
            ├── ...
            └── Sample9/
```

Jika belum ada, unduh dari Kaggle lalu ekstrak ke `archive/`:

```powershell
# Opsi A - Kaggle CLI (butuh kredensial Kaggle)
kaggle datasets download -d pintowar/numerical-images -p downloads --unzip

# Opsi B - unduh manual
# https://www.kaggle.com/datasets/pintowar/numerical-images
# Letakkan numbers.csv dan folder numbers/ di dalam archive/
```

Metadata dibaca dari `archive/numbers.csv` (fallback:
`metadata/numbers.csv`). Pipeline memfilter `origin == 'chars74k'` dan
`group == 'Hnd'`, lalu memvalidasi label, path, duplikasi, dan kemampuan baca
citra.

---

## 5. Menjalankan Pipeline

```powershell
# Eksekusi penuh (audit -> split -> HOG -> CV -> training -> evaluasi)
.\.venv\Scripts\python.exe -m src.main
```

Opsi tambahan:

```powershell
# Abaikan cache fitur HOG dan ekstraksi ulang
.\.venv\Scripts\python.exe -m src.main --force-features

# Tambahkan kurva mlogloss train/validasi internal (opsional)
.\.venv\Scripts\python.exe -m src.main --learning-curve

# Tentukan path metadata/dataset secara eksplisit
.\.venv\Scripts\python.exe -m src.main `
  --metadata "D:\path\numbers.csv" `
  --data-dir "D:\path\chars74k_png\Hnd"
```

Pipeline aman dijalankan ulang: fitur HOG dimuat dari `cache/` jika fingerprint
path, ukuran file, dan parameter masih sama.

---

## 6. Anti Data Leakage

- Split stratified 80/20 dilakukan **sebelum** scaler, PCA, atau model dilatih.
- `StandardScaler` dan `PCA` hanya di-`fit` pada data train di setiap fold CV
  melalui `sklearn.pipeline.Pipeline`.
- Model final di-`fit` pada seluruh train setelah CV selesai.
- Test set hanya dipakai satu kali per skenario, yaitu saat `predict`.
- Visualisasi PCA 2D juga di-`fit` pada train saja.

---

## 7. Artefak Output

### Reports (`outputs/reports/`)
| File | Isi |
|---|---|
| `metrics_comparison.csv` | Tabel perbandingan dua skenario |
| `classification_report_*.csv` | Precision/Recall/F1/Support per kelas |
| `per_class_metrics_*.csv` | TP, FP, FN, TN, support per digit |
| `top_confusions_*.csv` | 5 pasangan digit paling sering tertukar |
| `pca_variance.csv` | Explained variance per komponen |
| `cv_results.csv`, `cv_folds_*.csv` | Skor per fold CV |
| `learning_curve_*.csv` | mlogloss train/validasi internal (opsional) |
| `class_distribution.csv`, `split_distribution.csv` | Distribusi kelas |
| `data_audit.json`, `features_summary.json`, `summary.json` | Audit & ringkasan |
| `predictions_*.csv` | Prediksi test per sampel |

### Plots (`outputs/plots/`)
- `class_distribution.png`
- `sample_images_per_class.png`
- `clahe_before_after.png`
- `hog_visualization.png`
- `confusion_matrix_counts.png`
- `confusion_matrix_normalized.png`
- `pca_cumulative_variance.png`
- `pca_scatter.png`
- `cv_accuracy.png`
- `learning_curve_*.png` (jika `--learning-curve`)

### Models (`outputs/models/`)
- `xgb_tanpa_pca.joblib`
- `xgb_dengan_pca.joblib`

Contoh memuat model:

```python
import joblib
from src.preprocess import extract_hog_from_path

model = joblib.load("outputs/models/xgb_tanpa_pca.joblib")
prediction = model.predict(extract_hog_from_path("gambar_digit.png").reshape(1, -1))
print(prediction[0])
```

---

## 8. Catatan Akademik

- Dataset Hnd pada paket Kaggle ini hanya **550 citra** (55/kelas), sehingga
  test set berukuran 110 citra. Satu kesalahan mengubah accuracy sekitar
  0,91 poin persentase; interval kepercayaan perlu disampaikan di laporan.
- HOG menghasilkan **1.296 fitur** sesuai perhitungan
  `6 × 6 blok × 2 × 2 sel × 9 orientasi`.
- Hasil menunjukkan PCA berhasil mengurangi dimensi dan waktu training, tetapi
  tidak selalu meningkatkan akurasi. Ini adalah temuan yang valid dan objektif,
  bukan kegagalan pipeline.
- Augmentasi tidak diaktifkan (default PRD). Jika diaktifkan, augmentasi hanya
  boleh diterapkan pada train setelah split.
