import json
import os
import sys
import time
import uuid

import pytest
import redis
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
INPUT_LIST = os.getenv("ML_INPUT_LIST", "ml:input")
PRED_LIST = os.getenv("ML_PREDICTIONS_LIST", "ml:predictions")
APP_URL = os.getenv("APP_URL", "http://127.0.0.1:4000")


@pytest.fixture(scope="session")
def rconn():
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
    )
    pong = r.ping()
    assert pong is True
    return r


def test_e2e_prediction_flow(rconn):
    _len = rconn.llen(PRED_LIST)
    if _len and _len > 500:
        rconn.ltrim(PRED_LIST, -100, -1)

    rec_id = f"e2e-{uuid.uuid4().hex[:8]}"
    payload = {"id": rec_id, "instances": [{"mean radius": 14.5, "mean texture": 20.1}]}

    resp = requests.get(f"{APP_URL}/")
    assert resp.status_code == 200

    rconn.rpush(INPUT_LIST, json.dumps(payload))

    deadline = time.time() + 10
    got = None
    last = None
    while time.time() < deadline:
        items = rconn.lrange(PRED_LIST, -5, -1)
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

    assert "prediction" in got
    assert "ts" in got
