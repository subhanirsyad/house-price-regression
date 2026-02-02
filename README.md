# 🏠 House Price Regression — Streamlit Inference App

Repo ini berisi:
- **Model regresi harga rumah** (Python / scikit-learn)
- **Aplikasi inference** menggunakan **Streamlit** (`app.py`) — siap dijalankan lokal maupun deploy
- Dataset & laporan proyek asli

> Fokus repo ini adalah **deployment/inference** (prediksi) yang rapi untuk GitHub + Streamlit.

---

## ✨ Fitur
- Form input fitur untuk memprediksi harga rumah (single prediction)
- **Batch prediction**: upload CSV → download hasil prediksi
- Menampilkan metrik model (R², MAE, RMSE) + feature importance (indikatif)

---

## 🧱 Struktur Folder
```
.
├─ app.py                      # Streamlit app (inference)
├─ requirements.txt            # dependency untuk deploy
├─ artifacts/
│  ├─ model.joblib             # model terlatih (dipakai app)
│  ├─ model_metadata.json      # info fitur + metrik
│  └─ feature_stats.json       # min/max/median (untuk default input)
├─ src/
│  └─ train.py                 # script training untuk regenerate model
├─ data/
│  └─ house_price_regression_dataset.csv
├─ report/
│  └─ report.pdf
├─ r/
│  └─ house_price_regression.R # kode R asli (referensi)
├─ LICENSE
└─ .gitignore
```

---

## 🚀 Menjalankan Secara Lokal

### 1) Setup environment
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Jalankan Streamlit
```bash
streamlit run app.py
```
Buka URL yang muncul (biasanya `http://localhost:8501`).

---

## 🔁 (Opsional) Training ulang model
Kalau kamu ingin regenerate model dari dataset:

```bash
python src/train.py
```

Output model akan tersimpan ke folder `artifacts/`.

---

## 🌍 Deploy ke Streamlit Community Cloud (gratis)

1. Upload repo ini ke GitHub (public)
2. Buka Streamlit Community Cloud → **New app**
3. Pilih repo + branch
4. Pada **Main file path**, isi: `app.py`
5. Klik **Deploy**

Setelah berhasil, aplikasi akan punya link dengan format mirip:
`https://<streamlit-username>-<repo-name>.streamlit.app`

> Catatan: link final tergantung username Streamlit & nama repo GitHub kamu.

---

## 📌 Kolom fitur yang dibutuhkan (CSV batch)
Pastikan CSV batch memiliki kolom berikut:
- `Square_Footage`
- `Num_Bedrooms`
- `Num_Bathrooms`
- `Year_Built`
- `Lot_Size`
- `Garage_Size`
- `Neighborhood_Quality`

Aplikasi juga menyediakan tombol **Download CSV template**.

---

## 📝 Kredit & Catatan
- Dataset pada repo ini digunakan untuk keperluan edukasi/demonstrasi.
- Kode R asli tetap disertakan di folder `r/` sebagai referensi analisis/statistik.

