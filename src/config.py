from __future__ import annotations

import os
from configparser import ConfigParser
from dataclasses import dataclass


@dataclass(frozen=True)
class PathsCfg:
    data_csv: str
    model_dir: str
    model_path: str
    scaler_path: str
    metrics_path: str
    export_zip: str


@dataclass(frozen=True)
class TrainCfg:
    random_state: int
    test_size: float
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    validation_split: float


@dataclass(frozen=True)
class ModelCfg:
    hidden_units: list[int]
    dropout: float
    activation: str
    output_activation: str
    from_logits: bool


@dataclass(frozen=True)
class InferenceCfg:
    threshold: float


@dataclass(frozen=True)
class AppConfig:
    paths: PathsCfg
    train: TrainCfg
    model: ModelCfg
    inference: InferenceCfg


def load_config(path: str = "config.ini") -> AppConfig:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config.ini не найден: {path}")

    cp = ConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    cp.read(path, encoding="utf-8")
    hidden_units = [
        int(x.strip()) for x in cp.get("model", "hidden_units").split(",") if x.strip()
    ]

    return AppConfig(
        paths=PathsCfg(
            data_csv=cp.get("paths", "data_csv"),
            model_dir=cp.get("paths", "model_dir"),
            model_path=cp.get("paths", "model_path"),
            scaler_path=cp.get("paths", "scaler_path"),
            metrics_path=cp.get("paths", "metrics_path"),
            export_zip=cp.get("paths", "export_zip"),
        ),
        train=TrainCfg(
            random_state=cp.getint("train", "random_state"),
            test_size=cp.getfloat("train", "test_size"),
            epochs=cp.getint("train", "epochs"),
            batch_size=cp.getint("train", "batch_size"),
            learning_rate=cp.getfloat("train", "learning_rate"),
            patience=cp.getint("train", "patience"),
            validation_split=cp.getfloat("train", "validation_split"),
        ),
        model=ModelCfg(
            hidden_units=hidden_units,
            dropout=cp.getfloat("model", "dropout"),
            activation=cp.get("model", "activation"),
            output_activation=cp.get("model", "output_activation").strip().lower(),
            from_logits=cp.getboolean("model", "from_logits"),
        ),
        inference=InferenceCfg(threshold=cp.getfloat("inference", "threshold")),
    )
