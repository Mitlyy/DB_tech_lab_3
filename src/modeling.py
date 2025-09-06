from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


class BreastCancerClassifier:
    """
    Keras-модель для бинарной классификации.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: list[int] = [64, 32],
        dropout: float = 0.1,
        activation: str = "relu",
        output_activation: str = "none",
        from_logits: bool = True,
        learning_rate: float = 1e-3,
        model_path: str = "model/model.keras",
        metrics_path: str = "model/metrics.json",
    ):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.dropout = float(dropout)
        self.activation = activation
        self.output_activation = (
            None if output_activation == "none" else output_activation
        )
        self.from_logits = bool(from_logits)
        self.learning_rate = float(learning_rate)
        self.model_path = model_path
        self.metrics_path = metrics_path

        self.model: Sequential | None = None

    def build(self):
        m = Sequential(name="breast_cancer_classifier")
        m.add(
            Dense(
                self.hidden_units[0],
                activation=self.activation,
                input_shape=(self.input_dim,),
            )
        )
        if self.dropout > 0:
            m.add(Dropout(self.dropout))
        for h in self.hidden_units[1:]:
            m.add(Dense(h, activation=self.activation))
            if self.dropout > 0:
                m.add(Dropout(self.dropout))
        m.add(Dense(1, activation=self.output_activation))  # None -> логиты

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=self.from_logits)
        opt = Adam(learning_rate=self.learning_rate)
        m.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
        self.model = m
        return self

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        epochs: int = 10,
        batch_size: int = 32,
        patience: int = 5,
        validation_split: float = 0.1,
    ):
        if self.model is None:
            self.build()

        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            )
        ]

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            validation_split=None if X_val is not None else validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks,
        )
        return history

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель не инициализирована")
        raw = self.model.predict(X, verbose=0).reshape(-1)

        if self.from_logits or (np.nanmin(raw) < 0.0 or np.nanmax(raw) > 1.0):
            pos = raw >= 0
            neg = ~pos
            out = np.empty_like(raw, dtype=np.float64)
            out[pos] = 1.0 / (1.0 + np.exp(-raw[pos]))
            ex = np.exp(raw[neg])
            out[neg] = ex / (1.0 + ex)
            proba = out
        else:
            proba = raw

        proba = np.nan_to_num(proba, nan=0.5, posinf=1.0, neginf=0.0)
        proba = np.clip(proba, 0.0, 1.0)
        return proba

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        proba = self.predict_proba(X_test)
        preds = (proba >= 0.5).astype(int)

        metrics = {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds)),
        }
        return metrics

    def save(self):
        if self.model is None:
            raise RuntimeError("Нет модели для сохранения")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)

    def save_metrics(self, metrics: Dict[str, Any]):
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
