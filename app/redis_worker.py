import json
import os
import threading
import time
from typing import Optional

import redis
import requests

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
INPUT_LIST = os.getenv("ML_INPUT_LIST", "ml:input")
PRED_LIST = os.getenv("ML_PREDICTIONS_LIST", "ml:predictions")
PREDICT_URL = os.getenv("PREDICT_URL", "http://127.0.0.1:8000/predict")

BLPOP_TIMEOUT = 5
IDLE_SLEEP = 0.5


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


def _infer(payload: dict) -> Optional[dict]:
    """
    Вызов локального HTTP эндпоинта модели.
    """
    try:
        resp = requests.post(PREDICT_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[redis_worker] HTTP call failed: {e}")
        return None


def worker_loop():
    r = _connect_redis()
    print(
        f"[redis_worker] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}. "
        f"IN='{INPUT_LIST}' OUT='{PRED_LIST}' URL='{PREDICT_URL}'"
    )
    while True:
        try:
            item = r.blpop(INPUT_LIST, timeout=BLPOP_TIMEOUT)
            if not item:
                time.sleep(IDLE_SLEEP)
                continue

            _list, raw = item
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[redis_worker] Skip invalid JSON: {raw}")
                continue

            result = _infer(payload)
            if result is None:
                continue

            out = {
                "id": payload.get("id"),
                "prediction": result.get("prediction", result),
                "ts": int(time.time()),
            }
            r.rpush(PRED_LIST, json.dumps(out))
            print(f"[redis_worker] Wrote prediction for id={out['id']}")
        except Exception as e:
            print(f"[redis_worker] Loop error: {e}")
            time.sleep(1)


_worker_thread: Optional[threading.Thread] = None


def start_worker_background():
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return
    _worker_thread = threading.Thread(
        target=worker_loop, name="redis_worker", daemon=True
    )
    _worker_thread.start()
