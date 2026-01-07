# House Price Regression (R)

Proyek ini memodelkan **harga rumah** menggunakan **regresi linear** di R, berdasarkan fitur:
- `Square_Footage`
- `Num_Bedrooms`
- `Num_Bathrooms`
- `Year_Built`
- `Lot_Size`
- `Garage_Size`
- `Neighborhood_Quality`

📄 Laporan: [`report/report.pdf`](./report/report.pdf)  
📁 Dataset: [`data/house_price_regression_dataset.csv`](./data/house_price_regression_dataset.csv)  
🧠 Kode: [`src/house_price_regression.R`](./src/house_price_regression.R)

---

## Metode yang digunakan
- Analisis deskriptif + visualisasi korelasi
- Model regresi linear (OLS)
- **Stepwise regression** (AIC) untuk seleksi variabel (`MASS::stepAIC`)
- Uji asumsi klasik:
  - Linearitas (residual plot)
  - Independensi residual (Durbin–Watson: `car::durbinWatsonTest`)
  - Homoskedastisitas (Breusch–Pagan: `lmtest::bptest`)
  - Normalitas residual (Shapiro–Wilk)
  - Multikolinearitas (VIF: `car::vif`)
- Validasi model: **10-fold cross-validation** (`caret`)

---

## Tools / Packages
Script memakai beberapa paket berikut:
`readr`, `corrplot`, `reshape2`, `ggplot2`, `MASS`, `car`, `lmtest`, `performance`, `caret`

---

## Struktur Folder
```
house-price-regression/
  data/        # dataset (CSV)
  src/         # script R
  report/      # laporan PDF
  figures/     # output plot (opsional)
```

