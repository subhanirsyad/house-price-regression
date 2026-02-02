from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
META_PATH = ARTIFACTS_DIR / "model_metadata.json"
STATS_PATH = ARTIFACTS_DIR / "feature_stats.json"


@st.cache_resource
def load_artifacts():
    """
    Load model + metadata from artifacts.
    If loading fails (e.g., version mismatch), train a fresh model from the bundled dataset.
    """
    try:
        model = joblib.load(MODEL_PATH)
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        return model, meta, stats
    except Exception as e:
        st.warning(f"Gagal memuat model artifact ({type(e).__name__}). Melatih ulang model dari dataset...")
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        data_csv = APP_DIR / "data" / "house_price_regression_dataset.csv"
        df = pd.read_csv(data_csv)

        target = "House_Price"
        features = [c for c in df.columns if c != target]
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=400,
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

        # Best-effort: persist new artifacts
        try:
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        except Exception:
            pass

        return model, meta, stats
def format_number(x: float) -> str:
    # Avoid assuming any currency; just format a readable number.
    return f"{x:,.0f}"


def main():
    st.set_page_config(
        page_title="House Price Prediction",
        page_icon="🏠",
        layout="wide",
    )

    st.title("🏠 House Price Prediction (Regression)")
    st.caption(
        "Aplikasi inference sederhana berbasis Streamlit untuk memprediksi harga rumah dari fitur numerik."
    )

    with st.spinner("Memuat model..."):
        model, meta, stats = load_artifacts()

    features = meta["features"]
    target = meta.get("target", "House_Price")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Input fitur")
        st.write("Isi nilai fitur di bawah, lalu klik **Prediksi**.")

        with st.form("predict_form", clear_on_submit=False):
            # Build inputs using dataset-derived ranges and medians (stored in feature_stats.json)
            inputs = {}
            for f in features:
                f_stats = stats.get(f, {})
                vmin = f_stats.get("min", 0.0)
                vmax = f_stats.get("max", 1.0)
                vmed = f_stats.get("median", (vmin + vmax) / 2)

                # Pick sensible step sizes (integers for count-like features)
                is_int_like = f in {"Num_Bedrooms", "Num_Bathrooms", "Year_Built", "Garage_Size", "Neighborhood_Quality"}
                step = 1 if is_int_like else (vmax - vmin) / 100 if vmax != vmin else 0.01

                inputs[f] = st.number_input(
                    label=f,
                    min_value=float(vmin),
                    max_value=float(vmax),
                    value=float(vmed),
                    step=float(step) if step else 1.0,
                    help=f"Rentang data: {vmin:.3g} sampai {vmax:.3g}. Default: median ({vmed:.3g}).",
                )

            submitted = st.form_submit_button("🔮 Prediksi")

        if submitted:
            row = pd.DataFrame([inputs], columns=features)
            pred = float(model.predict(row)[0])
            st.success("Prediksi berhasil dibuat.")
            st.metric(label=f"Prediksi {target}", value=format_number(pred))

            with st.expander("Lihat input yang dipakai", expanded=False):
                st.dataframe(row, use_container_width=True)

    with right:
        st.subheader("Info model")
        st.write(
            f"**Model:** {meta.get('model_type','-')}  \n"
            f"**Fitur:** {len(features)}  \n"
            f"**Target:** `{target}`"
        )

        m = meta.get("metrics", {})
        if m:
            c1, c2, c3 = st.columns(3)
            c1.metric("R² (test)", f"{m.get('R2', 0):.4f}")
            c2.metric("MAE (test)", format_number(float(m.get('MAE', 0))))
            c3.metric("RMSE (test)", format_number(float(m.get('RMSE', 0))))

        # Feature importance (if available)
        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame(
                {"feature": features, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)
            st.markdown("**Feature importance (indikatif)**")
            st.dataframe(imp, use_container_width=True)

        st.divider()
        st.subheader("Batch prediction (CSV)")
        st.write(
            "Upload file CSV dengan kolom fitur yang sama seperti input di kiri. "
            "Aplikasi akan menambahkan kolom prediksi dan kamu bisa download hasilnya."
        )

        # Download a template
        template = pd.DataFrame([{f: stats.get(f, {}).get("median", 0) for f in features}])
        template_buf = io.StringIO()
        template.to_csv(template_buf, index=False)
        st.download_button(
            "⬇️ Download CSV template",
            data=template_buf.getvalue().encode("utf-8"),
            file_name="template_house_features.csv",
            mime="text/csv",
        )

        uploaded = st.file_uploader("Upload CSV untuk batch prediction", type=["csv"])
        if uploaded is not None:
            batch = pd.read_csv(uploaded)
            missing = [c for c in features if c not in batch.columns]
            if missing:
                st.error(f"Kolom berikut wajib ada tapi belum ditemukan: {missing}")
            else:
                preds = model.predict(batch[features])
                out = batch.copy()
                out[target + "_Predicted"] = preds
                st.success(f"Sukses memproses {len(out)} baris.")
                st.dataframe(out.head(50), use_container_width=True)

                out_buf = io.StringIO()
                out.to_csv(out_buf, index=False)
                st.download_button(
                    "⬇️ Download hasil prediksi",
                    data=out_buf.getvalue().encode("utf-8"),
                    file_name="house_price_predictions.csv",
                    mime="text/csv",
                )

    st.divider()
    st.caption(
        "Catatan: prediksi mengikuti pola dataset yang tersedia pada repo ini; gunakan untuk edukasi/demonstrasi."
    )


if __name__ == "__main__":
    main()
