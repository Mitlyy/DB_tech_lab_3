import io
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import redis
import requests
from flask import Flask, jsonify, render_template, request, send_file

# ------------------- Конфигурация через ENV -------------------
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "4000"))

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

INPUT_LIST = os.getenv("ML_INPUT_LIST", "ml:input")
PRED_LIST = os.getenv("ML_PREDICTIONS_LIST", "ml:predictions")

PREDICT_URL = os.getenv("PREDICT_URL", f"http://127.0.0.1:{APP_PORT}/predict")

ENABLE_REDIS_WORKER = os.getenv("ENABLE_REDIS_WORKER", "0") == "1"
BLPOP_TIMEOUT = 5
IDLE_SLEEP = 0.5

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

_inference_service: Optional["InferenceService"] = None  # type: ignore[name-defined]


def get_inference():
    """
    Ленивая загрузка модели/скейлера при первом обращении.
    Избегаем тяжёлых импортов на старте процесса (совместимо с Flask 3).
    """
    global _inference_service
    if _inference_service is None:
        from src.inference import InferenceService

        _inference_service = InferenceService(
            model_path="model/model.keras",
            scaler_path="model/scaler.pkl",
            threshold=0.5,
        )
    return _inference_service


_worker_thread: Optional[threading.Thread] = None


def _connect_redis() -> redis.Redis:
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
        health_check_interval=30,
    )


def _infer_via_http(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Вызов локального /predict (наши же процессы), чтобы использовать общую логику инференса.
    Возвращает словарь JSON или None при ошибке.
    """
    try:
        resp = requests.post(PREDICT_URL, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[redis-worker] HTTP /predict failed: {e}")
        return None


def worker_loop():
    """
    Цикл воркера:
      1) BLPOP из INPUT_LIST,
      2) вызов /predict,
      3) запись результата в PRED_LIST (RPUSH).
    Формат входа (рекоменд.): {"id":"...", "instances":[{feature: value, ...}]}
    """
    r = _connect_redis()
    print(
        f"[redis-worker] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}. "
        f"IN='{INPUT_LIST}' OUT='{PRED_LIST}' URL='{PREDICT_URL}'"
    )
    while True:
        try:
            item = r.blpop(INPUT_LIST, timeout=BLPOP_TIMEOUT)
            if not item:
                time.sleep(IDLE_SLEEP)
                continue

            _list_name, raw = item
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[redis-worker] Skip invalid JSON: {raw}")
                continue

            result = _infer_via_http(payload)
            if result is None:
                continue

            out = {
                "id": payload.get("id"),
                "ts": int(time.time()),
            }
            if isinstance(result, dict) and "predictions" in result:
                out["prediction"] = result["predictions"]
            else:
                out["prediction"] = result  # положим весь ответ

            r.rpush(PRED_LIST, json.dumps(out))
            print(f"[redis-worker] Wrote prediction for id={out.get('id')}")
        except Exception as e:
            print(f"[redis-worker] Loop error: {e}")
            time.sleep(1)


def start_worker_background():
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return
    _worker_thread = threading.Thread(
        target=worker_loop, name="redis_worker", daemon=True
    )
    _worker_thread.start()
    print("[redis-worker] Started background worker thread")


if ENABLE_REDIS_WORKER:
    try:
        start_worker_background()
        print("[app] Redis worker ENABLED (started at import time)")
    except Exception as e:
        print(f"[app] Failed to start worker: {e}")
else:
    print("[app] Redis worker DISABLED (set ENABLE_REDIS_WORKER=1 to enable)")


@app.route("/", methods=["GET"])
def index():
    template_path = os.path.join(app.root_path, "templates", "index.html")
    if os.path.isfile(template_path):
        return render_template("index.html")
    return (
        "<h2>Breast Cancer Prediction</h2>"
        "<p>POST CSV на /upload или JSON на /predict</p>",
        200,
        {"Content-Type": "text/html"},
    )


@app.route("/health/worker", methods=["GET"])
def health_worker():
    alive = _worker_thread.is_alive() if _worker_thread else False
    return (
        jsonify(
            {
                "worker_enabled_env": ENABLE_REDIS_WORKER,
                "worker_alive": bool(alive),
                "redis_host": REDIS_HOST,
                "redis_port": REDIS_PORT,
                "input_list": INPUT_LIST,
                "pred_list": PRED_LIST,
                "predict_url": PREDICT_URL,
            }
        ),
        200,
    )


@app.route("/upload", methods=["POST"])
def upload():
    """
    Принимает multipart/form-data (поле 'file' с CSV).
    Возвращает predictions.csv для скачивания.
    """
    if "file" not in request.files:
        return jsonify({"error": "Отсутствует поле 'file' с CSV"}), 400

    file = request.files["file"]
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Не удалось прочитать CSV: {e}"}), 400

    try:
        infer = get_inference()
        labels = infer.predict_df(df)
        proba = infer.predict_proba_df(df)

        out_df = df.copy()
        out_df["probability"] = proba
        out_df["prediction"] = labels

        buf = io.StringIO()
        out_df.to_csv(buf, index=False)
        buf.seek(0)

        return send_file(
            io.BytesIO(buf.getvalue().encode("utf-8")),
            mimetype="text/csv",
            as_attachment=True,
            download_name="predictions.csv",
        )
    except Exception as e:
        return jsonify({"error": f"Ошибка инференса: {e}"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON API. Поддерживаются два формата:
      1) records: {"instances": [ {feature: value, ...}, ... ]}
      2) matrix:  {"data": [[...], [...]], "feature_names": ["f1", ...]? (опц.)}
    """
    try:
        payload: Dict[str, Any] = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Требуется JSON в теле запроса"}), 400

    if not isinstance(payload, dict):
        return jsonify({"error": "Неверный формат JSON"}), 400

    infer = get_inference()

    if "instances" in payload:
        instances = payload["instances"]
        if not isinstance(instances, list) or not all(
            isinstance(x, dict) for x in instances
        ):
            return (
                jsonify(
                    {
                        "error": "Поле 'instances' должно быть списком объектов {feature: value}"
                    }
                ),
                400,
            )

        proba = infer.predict_proba_records(instances)
        labels = infer.predict_records(instances)
        return (
            jsonify(
                {
                    "count": len(labels),
                    "probabilities": proba,
                    "predictions": labels,
                    "threshold": infer.threshold,
                }
            ),
            200,
        )

    if "data" in payload:
        data = payload["data"]
        if not isinstance(data, list) or (
            len(data) > 0 and not isinstance(data[0], list)
        ):
            return (
                jsonify({"error": "Поле 'data' должно быть списком списков чисел"}),
                400,
            )

        req_feature_names: Optional[List[str]] = None
        if "feature_names" in payload:
            req_feature_names = payload["feature_names"]
            if not isinstance(req_feature_names, list) or not all(
                isinstance(x, str) for x in req_feature_names
            ):
                return (
                    jsonify(
                        {"error": "Поле 'feature_names' должно быть списком строк"}
                    ),
                    400,
                )

        try:
            if req_feature_names is not None:
                df = pd.DataFrame(data, columns=req_feature_names)
                proba = infer.predict_proba_df(df)
                labels = infer.predict_df(df)
            else:
                proba = infer.predict_proba_array(data)
                labels = infer.predict_array(data)
        except Exception as e:
            return jsonify({"error": f"Ошибка инференса: {e}"}), 400

        return (
            jsonify(
                {
                    "count": len(labels),
                    "probabilities": proba,
                    "predictions": labels,
                    "threshold": infer.threshold,
                    "feature_order": infer.feature_names,
                }
            ),
            200,
        )

    return jsonify({"error": "Ожидались поля 'instances' или 'data'"}), 400


if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug=False)
