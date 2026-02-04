from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --- KONFIGURASI PATH ---
APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_DIR / "artifacts"
DATA_DIR = APP_DIR / "data"

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
META_PATH = ARTIFACTS_DIR / "model_metadata.json"
STATS_PATH = ARTIFACTS_DIR / "feature_stats.json"


# --- FUNGSI UTILITAS ---
def format_currency(x: float) -> str:
    """Format angka menjadi gaya mata uang atau angka yang mudah dibaca."""
    return f"{x:,.0f}"

@st.cache_resource
def load_artifacts():
    """
    Memuat model + metadata dari folder artifacts.
    Jika gagal (misal file tidak ada), melatih ulang model dari dataset bawaan.
    """
    try:
        model = joblib.load(MODEL_PATH)
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        return model, meta, stats
    except Exception as e:
        st.warning(f"⚠️ Artifact model tidak ditemukan atau rusak ({type(e).__name__}). Melatih ulang model...")
        
        # Cek apakah dataset tersedia
        data_csv = DATA_DIR / "house_price_regression_dataset.csv"
        if not data_csv.exists():
            st.error("❌ Dataset tidak ditemukan di folder 'data'. Pastikan file CSV tersedia untuk pelatihan ulang.")
            st.stop()

        # Import library hanya jika diperlukan training
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(data_csv)

        target = "House_Price"
        # Ambil kolom numerik saja sebagai fitur sederhana
        features = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
        
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, pred))
        try:
            from sklearn.metrics import root_mean_squared_error
            rmse = float(root_mean_squared_error(y_test, pred))
        except Exception:
            rmse = float(mean_squared_error(y_test, pred, squared=False))
        r2 = float(r2_score(y_test, pred))

        meta = {
            "model_type": "RandomForestRegressor",
            "target": target,
            "features": features,
            "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
            "train_test_split": {"test_size": 0.2, "random_state": 42},
        }

        stats = {}
        for c in features:
            s = df[c]
            stats[c] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "median": float(s.median()),
                "mean": float(s.mean()),
            }

        # Simpan artifact untuk penggunaan berikutnya
        try:
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        except Exception:
            pass

        return model, meta, stats

# --- HALAMAN UTAMA ---
def main():
    st.set_page_config(
        page_title="Estimasi Harga Rumah",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS untuk mempercantik
    st.markdown("""
        <style>
        .big-font { font-size: 24px !important; font-weight: bold; }
        .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; }
        </style>
    """, unsafe_allow_html=True)

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🏠 House Price AI")
        st.markdown("Aplikasi cerdas untuk estimasi nilai properti.")
        st.info(
            "**Tentang Aplikasi:**\n"
            "Model ini dilatih menggunakan algoritma *Random Forest Regression*. "
            "Data yang digunakan mencakup fitur fisik bangunan dan lokasi."
        )
        st.caption("v1.0.0 | Dibuat dengan Streamlit")

    # --- HEADER & PENDAHULUAN ---
    st.title("🏠 Prediksi Harga Rumah")
    
    with st.expander("📖 Baca Pendahuluan & Cara Penggunaan", expanded=True):
        st.markdown("""
        ### Selamat Datang!
        Aplikasi ini dirancang untuk membantu Anda **memperkirakan harga pasar rumah** berdasarkan spesifikasi properti.
        
        **Cara Menggunakan:**
        1.  **Tab Prediksi Cepat:** Masukkan detail rumah (jumlah kamar, luas, tahun dibangun, dll) secara manual untuk mendapatkan estimasi instan.
        2.  **Tab Batch CSV:** Jika Anda agen properti atau analis, upload file CSV berisi banyak data rumah untuk mendapatkan prediksi sekaligus.
        3.  **Tab Model Info:** Lihat seberapa akurat model ini dan faktor apa saja yang paling mempengaruhi harga.
        
        > *Catatan: Hasil prediksi adalah estimasi statistik berdasarkan data historis, bukan penilaian appraisal resmi.*
        """)

    # Load Model
    with st.spinner("Sedang memuat model kecerdasan buatan..."):
        model, meta, stats = load_artifacts()

    features = meta["features"]
    target = meta.get("target", "House_Price")

    # --- TABS NAVIGASI ---
    tab1, tab2, tab3 = st.tabs(["🔍 Prediksi Cepat", "📂 Batch Upload (CSV)", "📊 Info Model"])

    # --- TAB 1: PREDIKSI SATUAN ---
    with tab1:
        st.subheader("Simulasi Harga Properti")
        st.markdown("Sesuaikan parameter di bawah ini dengan kondisi rumah.")

        with st.form("predict_form"):
            # Menggunakan Grid Layout (2 Kolom) agar form tidak terlalu panjang
            cols = st.columns(2)
            inputs = {}
            
            for i, f in enumerate(features):
                col = cols[i % 2] # Ganti kolom kiri/kanan
                
                f_stats = stats.get(f, {})
                vmin = f_stats.get("min", 0.0)
                vmax = f_stats.get("max", 1000.0)
                vmed = f_stats.get("median", (vmin + vmax) / 2)

                # Logika step size agar input lebih nyaman
                is_int_like = f in {"Num_Bedrooms", "Num_Bathrooms", "Year_Built", "Garage_Size", "Neighborhood_Quality", "Overall_Condition"}
                step = 1.0 if is_int_like else (vmax - vmin) / 100
                if step == 0: step = 0.1

                with col:
                    inputs[f] = st.number_input(
                        label=f.replace("_", " ").title(), # Mempercantik label (e.g., Num_Bedrooms -> Num Bedrooms)
                        min_value=float(vmin),
                        max_value=float(vmax),
                        value=float(vmed),
                        step=float(step),
                        help=f"Min: {vmin} | Max: {vmax}"
                    )

            st.markdown("---")
            submitted = st.form_submit_button("🔮 Hitung Estimasi Harga", use_container_width=True, type="primary")

        if submitted:
            row = pd.DataFrame([inputs], columns=features)
            pred = float(model.predict(row)[0])
            
            st.markdown("### Hasil Analisis")
            c_res1, c_res2 = st.columns([2, 1])
            
            with c_res1:
                st.success("Kalkulasi Selesai!")
                st.metric(
                    label="Estimasi Harga Jual", 
                    value=f"IDR {format_currency(pred)}", 
                    delta="Berdasarkan data pasar"
                )
            with c_res2:
                with st.expander("Lihat Data Input"):
                    st.dataframe(row.T, use_container_width=True)

    # --- TAB 2: BATCH PREDICTION ---
    with tab2:
        st.subheader("Analisis Massal (Batch Processing)")
        st.markdown("Upload file CSV berisi daftar rumah untuk diprediksi sekaligus.")

        # Download Template
        col_dl, col_ul = st.columns([1, 2])
        with col_dl:
            st.markdown("**1. Download Template**")
            template = pd.DataFrame([{f: stats.get(f, {}).get("median", 0) for f in features}])
            template_buf = io.StringIO()
            template.to_csv(template_buf, index=False)
            st.download_button(
                "⬇️ Download Template CSV",
                data=template_buf.getvalue().encode("utf-8"),
                file_name="template_rumah.csv",
                mime="text/csv",
            )

        with col_ul:
            st.markdown("**2. Upload Data**")
            uploaded = st.file_uploader("Drop file CSV di sini", type=["csv"])

        if uploaded is not None:
            batch = pd.read_csv(uploaded)
            missing = [c for c in features if c not in batch.columns]
            
            if missing:
                st.error(f"❌ Format CSV salah. Kolom berikut hilang: {', '.join(missing)}")
            else:
                try:
                    preds = model.predict(batch[features])
                    out = batch.copy()
                    out[f"Prediksi_{target}"] = preds
                    
                    st.divider()
                    st.success(f"✅ Berhasil memproses {len(out)} data properti.")
                    
                    # Tampilkan Preview
                    st.dataframe(out.head(), use_container_width=True)

                    # Download Hasil
                    out_buf = io.StringIO()
                    out.to_csv(out_buf, index=False)
                    st.download_button(
                        "⬇️ Download Hasil Prediksi Full",
                        data=out_buf.getvalue().encode("utf-8"),
                        file_name="hasil_prediksi_rumah.csv",
                        mime="text/csv",
                        type="primary"
                    )
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses data: {e}")

    # --- TAB 3: INFO MODEL ---
    with tab3:
        st.subheader("Kinerja & Statistik Model")
        
        m = meta.get("metrics", {})
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.metric("R² Score (Akurasi)", f"{m.get('R2', 0):.2%}", help="Semakin mendekati 100%, semakin akurat model menjelaskan variasi data.")
        with c2:
            st.metric("Mean Absolute Error", format_currency(float(m.get('MAE', 0))), help="Rata-rata kesalahan prediksi dalam satuan harga.")
        with c3:
            st.metric("RMSE", format_currency(float(m.get('RMSE', 0))))

        st.divider()
        
        # Feature Importance
        if hasattr(model, "feature_importances_"):
            st.markdown("#### Faktor Penentu Harga")
            st.caption("Grafik ini menunjukkan fitur mana yang paling berpengaruh terhadap harga rumah menurut model.")
            
            imp = pd.DataFrame(
                {"Fitur": features, "Kepentingan": model.feature_importances_}
            ).sort_values("Kepentingan", ascending=True) # Ascending for bar chart
            
            st.bar_chart(imp.set_index("Fitur"), color="#4F8BF9", horizontal=True)

if __name__ == "__main__":
    main()
