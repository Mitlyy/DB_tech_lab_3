from __future__ import annotations

import argparse
import os
import zipfile

import pandas as pd
from src.config import load_config
from src.data_loader import DataLoader
from src.modeling import BreastCancerClassifier
from src.preprocess import Preprocessor


def cmd_train():
    cfg = load_config()

    processed_path = "data/processed.csv"
    use_processed = os.path.isfile(processed_path)

    if use_processed:
        df = pd.read_csv(processed_path)
        if "target" not in df.columns:
            raise ValueError("В processed.csv не найдена колонка 'target'")

        feature_cols = [c for c in df.columns if c != "target"]
        X_all = df[feature_cols]
        y_all = df["target"].astype(int)

        from src.preprocess import Preprocessor

        pre = Preprocessor(cfg.paths.scaler_path).load()

        from sklearn.model_selection import train_test_split

        X_train_s, X_test_s, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=cfg.train.test_size,
            random_state=cfg.train.random_state,
            stratify=y_all,
        )
    else:
        from src.data_loader import DataLoader
        from src.preprocess import Preprocessor

        loader = DataLoader(
            cfg.paths.data_csv, cfg.train.test_size, cfg.train.random_state
        )
        df = loader.load()
        X, y = loader.build_xy(df)

        pre = Preprocessor(cfg.paths.scaler_path)
        X_train, X_test, y_train, y_test = loader.train_test_split(X, y)
        X_train_s = pre.fit_transform(X_train)
        X_test_s = pre.transform(X_test)
        pre.save()

    from src.modeling import BreastCancerClassifier

    model = BreastCancerClassifier(
        input_dim=X_train_s.shape[1],
        hidden_units=cfg.model.hidden_units,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation,
        output_activation=cfg.model.output_activation,
        from_logits=cfg.model.from_logits,
        learning_rate=cfg.train.learning_rate,
        model_path=cfg.paths.model_path,
        metrics_path=cfg.paths.metrics_path,
    )
    model.build()

    print(">>> START TRAINING")
    model.train(
        X_train_s,
        y_train,
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        patience=cfg.train.patience,
        validation_split=cfg.train.validation_split,
    )

    model.save()
    print(">>> MODEL", cfg.paths.model_path)

    metrics = model.evaluate(X_test_s, y_test)
    model.save_metrics(metrics)
    print("Metrics:", metrics)


def cmd_evaluate():
    cfg = load_config()

    loader = DataLoader(cfg.paths.data_csv, cfg.train.test_size, cfg.train.random_state)
    df = loader.load()
    X, y = loader.build_xy(df)

    pre = Preprocessor(cfg.paths.scaler_path).load()
    X_s = pre.transform(X)

    model = BreastCancerClassifier(
        input_dim=X_s.shape[1],
        hidden_units=cfg.model.hidden_units,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation,
        output_activation=cfg.model.output_activation,
        from_logits=cfg.model.from_logits,
        learning_rate=cfg.train.learning_rate,
        model_path=cfg.paths.model_path,
        metrics_path=cfg.paths.metrics_path,
    )
    import tensorflow as tf

    model.model = tf.keras.models.load_model(cfg.paths.model_path)

    metrics = model.evaluate(X_s, y)
    model.save_metrics(metrics)
    print("Оценка завершена. Метрики:", metrics)


def cmd_export():
    cfg = load_config()
    os.makedirs(os.path.dirname(cfg.paths.export_zip), exist_ok=True)

    with zipfile.ZipFile(cfg.paths.export_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in [
            cfg.paths.model_path,
            cfg.paths.scaler_path,
            cfg.paths.metrics_path,
            "config.ini",
        ]:
            if os.path.isfile(p):
                z.write(p, arcname=os.path.basename(p))
    print(f"Экспортирован архив: {cfg.paths.export_zip}")


def main():
    ap = argparse.ArgumentParser(description="ML pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train", help="Обучить модель и сохранить артефакты")
    sub.add_parser("evaluate", help="Оценить модель и обновить метрики")
    sub.add_parser("export", help="Собрать zip-дистрибутив артефактов")

    args = ap.parse_args()

    if args.cmd == "train":
        cmd_train()
    elif args.cmd == "evaluate":
        cmd_evaluate()
    elif args.cmd == "export":
        cmd_export()


if __name__ == "__main__":
    main()
