from __future__ import annotations

import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """
    Оборачивает StandardScaler, гарантируя сохранение feature_names_in_.
    """

    def __init__(self, scaler_path: str):
        self.scaler_path = scaler_path
        self.scaler: StandardScaler | None = None

    def fit(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Ожидается pandas.DataFrame для сохранения feature_names_in_"
            )
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            raise RuntimeError("Scaler не обучен: вызовите fit()")
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(
            X_scaled, columns=self.scaler.feature_names_in_, index=X.index
        )

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

    def save(self):
        if self.scaler is None:
            raise RuntimeError("Нечего сохранять: scaler не обучен")
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)

    def load(self):
        if not os.path.isfile(self.scaler_path):
            raise FileNotFoundError(f"Не найден scaler: {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)
        return self
