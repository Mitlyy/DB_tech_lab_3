import json
import os

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


class InferenceService:
    """
    Единая точка загрузки артефактов (Keras-модель + StandardScaler) и инференса.
    - Возвращает как вероятности, так и бинарные метки по порогу.
    """

    def __init__(
        self,
        model_path: str = "model/model.keras",
        scaler_path: str = "model/scaler.pkl",
        threshold: float = 0.5,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Keras-модель не найдена по пути: {model_path}")
        if not os.path.isfile(scaler_path):
            raise FileNotFoundError(f"Scaler не найден по пути: {scaler_path}")

        self.model_path = model_path
        self.scaler_path = scaler_path
        self.threshold = float(threshold)

        self.model = load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        if hasattr(self.scaler, "feature_names_in_"):
            self.feature_names = list(self.scaler.feature_names_in_)
        else:
            raise AttributeError("У скейлера нет feature_names_in_")

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Выравнивает входной DataFrame под необходимый набор/порядок признаков.
        """
        X = df.copy()

        drop_cols = {"id", "diagnosis", "Unnamed: 32"}
        for c in drop_cols:
            if c in X.columns:
                X = X.drop(columns=[c])

        for c in self.feature_names:
            if c not in X.columns:
                X[c] = 0.0

        X = X[self.feature_names]

        X = X.astype(float)

        return X

    def predict_proba_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Принимает DataFrame, выравнивает под нужные признаки, скейлит
        и возвращает вероятности класса 1 в диапазоне [0, 1], shape: (n_samples,).
        """
        X = self._align_features(df)
        X_scaled = self.scaler.transform(X)

        raw = self.model.predict(X_scaled, verbose=0)
        raw = np.asarray(raw)

        def _sigmoid(x: np.ndarray) -> np.ndarray:
            out = np.empty_like(x, dtype=np.float64)
            pos_mask = x >= 0
            neg_mask = ~pos_mask
            out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
            expx = np.exp(x[neg_mask])
            out[neg_mask] = expx / (1.0 + expx)
            return out

        def _softmax_2(z: np.ndarray) -> np.ndarray:
            z = z.astype(np.float64)
            z = z - np.max(z, axis=1, keepdims=True)
            ez = np.exp(z)
            denom = np.sum(ez, axis=1, keepdims=True)
            return ez / denom

        if raw.ndim == 2 and raw.shape[1] == 1:
            v = raw.reshape(-1)  # (n,)
            if np.nanmin(v) < 0.0 or np.nanmax(v) > 1.0:
                v = _sigmoid(v)
            proba = v

        elif raw.ndim == 2 and raw.shape[1] == 2:
            sm = _softmax_2(raw)
            proba = sm[:, 1]

        else:
            v = raw.reshape(-1)
            proba = _sigmoid(v)

        proba = np.nan_to_num(proba, nan=0.5, posinf=1.0, neginf=0.0)
        proba = np.clip(proba, 0.0, 1.0)

        return proba

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Возвращает бинарные метки по порогу self.threshold. Shape: (n,)
        """
        proba = self.predict_proba_df(df)
        labels = (proba >= self.threshold).astype(int)
        return labels

    def predict_proba_records(self, records: list[dict]) -> list[float]:
        """
        Принимает список dict'ов {feature: value}, возвращает список вероятностей.
        """
        df = pd.DataFrame.from_records(records)
        proba = self.predict_proba_df(df)
        return proba.astype(float).tolist()

    def predict_records(self, records: list[dict]) -> list[int]:
        """
        Принимает список dict'ов {feature: value}, возвращает список меток {0,1}.
        """
        df = pd.DataFrame.from_records(records)
        labels = self.predict_df(df)
        return [int(x) for x in labels]

    def predict_proba_array(self, data: list[list[float]]) -> list[float]:
        """
        Принимает список списков (матрица), где порядок колонок = self.feature_names.
        Использовать только если клиент гарантирует порядок!
        """
        df = pd.DataFrame(data, columns=self.feature_names)
        proba = self.predict_proba_df(df)
        return proba.astype(float).tolist()

    def predict_array(self, data: list[list[float]]) -> list[int]:
        df = pd.DataFrame(data, columns=self.feature_names)
        labels = self.predict_df(df)
        return [int(x) for x in labels]
