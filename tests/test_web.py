import json

import numpy as np
import pandas as pd
import pytest
from src.inference import InferenceService
from web.app import app


@pytest.fixture(scope="module")
def client():
    app.testing = True
    with app.test_client() as c:
        yield c


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_predict_instances(client):
    infer = InferenceService()
    cols = infer.feature_names

    rec1 = {c: float(i) for i, c in enumerate(cols[:10])}
    rec2 = {c: float(i) for i, c in enumerate(cols)}

    payload = {"instances": [rec1, rec2]}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "predictions" in data and "probabilities" in data
    assert len(data["predictions"]) == 2
    assert all(p in [0, 1] for p in data["predictions"])
    assert all(0.0 <= x <= 1.0 for x in data["probabilities"])


def test_predict_data_matrix(client):
    infer = InferenceService()
    cols = infer.feature_names

    mat = (np.random.randn(3, len(cols))).tolist()
    payload = {"data": mat}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["predictions"]) == 3
    assert "feature_order" in data
