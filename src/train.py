from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train(
    data_csv: Path,
    artifacts_dir: Path,
    target: str = "House_Price",
    test_size: float = 0.2,
    random_state: int = 42,
):
    df = pd.read_csv(data_csv)
    features = [c for c in df.columns if c != target]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=random_state,
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

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / "model.joblib")

    meta = {
        "model_type": "RandomForestRegressor",
        "target": target,
        "features": features,
        "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
        "train_test_split": {"test_size": test_size, "random_state": random_state},
    }
    (artifacts_dir / "model_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    stats = {}
    for c in features:
        s = df[c]
        stats[c] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "mean": float(s.mean()),
        }
    (artifacts_dir / "feature_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    return meta


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_csv = repo_root / "data" / "house_price_regression_dataset.csv"
    artifacts_dir = repo_root / "artifacts"

    meta = train(data_csv=data_csv, artifacts_dir=artifacts_dir)
    print("Training selesai.")
    print(json.dumps(meta["metrics"], indent=2))


if __name__ == "__main__":
    main()
