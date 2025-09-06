import json
import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/data.csv"
PROCESSED_PATH = "data/processed.csv"
SCALER_PATH = "model/scaler.pkl"
STATS_PATH = "model/prepare_stats.json"  # для отладки/контроля


def main():
    if not os.path.isfile(RAW_PATH):
        raise FileNotFoundError(f"Не найден сырой датасет: {RAW_PATH}")

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    if "diagnosis" not in df.columns:
        raise ValueError("Ожидается колонка 'diagnosis' (B/M) в сыром датасете")

    y = (df["diagnosis"].astype(str).str.upper() == "M").astype(int)

    drop_cols = [c for c in ["id", "diagnosis", "Unnamed: 32"] if c in df.columns]
    X = df.drop(columns=drop_cols).astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, SCALER_PATH)

    out = pd.DataFrame(X_scaled, columns=scaler.feature_names_in_)
    out["target"] = y.values
    out.to_csv(PROCESSED_PATH, index=False)

    stats = {
        "n_rows": int(out.shape[0]),
        "n_features": int(len(scaler.feature_names_in_)),
        "target_positive": int(out["target"].sum()),
        "target_negative": int(out.shape[0] - out["target"].sum()),
    }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f" Prepared: {PROCESSED_PATH}")
    print(f" Scaler:   {SCALER_PATH}")
    print(f" Stats:    {STATS_PATH} -> {stats}")


if __name__ == "__main__":
    main()
