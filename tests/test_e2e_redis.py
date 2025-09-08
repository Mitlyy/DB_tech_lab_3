import json
import os
import time
import uuid

import redis
import requests

# Подключение к Redis и приложению (локально)
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
INPUT_LIST = os.getenv("ML_INPUT_LIST", "ml:input")
PRED_LIST = os.getenv("ML_PREDICTIONS_LIST", "ml:predictions")
APP_URL = os.getenv("APP_URL", "http://127.0.0.1:4000")

# Полный набор 30 признаков sklearn breast_cancer
FEATURES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]


def _make_instance():
    # База — нули, чтобы не заморачиваться с реальными значениями
    inst = {name: 0.0 for name in FEATURES}
    # Чуть-чуть зададим осмысленных значений, не принципиально
    inst["mean radius"] = 14.5
    inst["mean texture"] = 20.1
    inst["worst radius"] = 16.8
    inst["worst area"] = 900.0
    return inst


def test_e2e_prediction_flow():
    r = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True
    )
    assert r.ping() is True

    # sanity: HTTP жив
    resp = requests.get(f"{APP_URL}/")
    assert resp.status_code == 200

    # кладём задание
    rec_id = f"e2e-{uuid.uuid4().hex[:8]}"
    payload = {"id": rec_id, "instances": [_make_instance()]}
    r.rpush(INPUT_LIST, json.dumps(payload))

    # ждём результат до 20 сек
    deadline = time.time() + 20
    got = None
    last = []
    while time.time() < deadline:
        items = r.lrange(PRED_LIST, -10, -1)
        last = items
        for it in items:
            try:
                obj = json.loads(it)
            except Exception:
                continue
            if obj.get("id") == rec_id:
                got = obj
                break
        if got:
            break
        time.sleep(0.5)

    assert (
        got is not None
    ), f"Не нашли предсказание для id={rec_id}. Последние элементы: {last}"
    assert "prediction" in got and "ts" in got
