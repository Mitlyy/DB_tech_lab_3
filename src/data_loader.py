from __future__ import annotations

import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    """
    Загружает исходный CSV и формирует X, y.
    Для набора Breast Cancer: y = (diagnosis == 'M') -> 1/0.
    """

    def __init__(self, csv_path: str, test_size: float = 0.2, random_state: int = 42):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Файл данных не найден: {csv_path}")
        self.csv_path = csv_path
        self.test_size = test_size
        self.random_state = random_state

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        return df

    def build_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.copy()

        if "diagnosis" not in df.columns:
            raise ValueError("В данных нет столбца 'diagnosis'")

        y = (df["diagnosis"].astype(str).str.upper() == "M").astype(int)

        drop_cols = [c for c in ["id", "diagnosis", "Unnamed: 32"] if c in df.columns]
        X = df.drop(columns=drop_cols)

        X = X.astype(float)

        return X, y

    def train_test_split(self, X: pd.DataFrame, y: pd.Series):
        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
