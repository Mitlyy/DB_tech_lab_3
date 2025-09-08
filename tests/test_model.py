import os
import sys

import numpy as np
import pandas as pd
import pytest

from src.inference import InferenceService

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def infer():
    assert os.path.isfile("model/model.keras"), "Отсутствует model/model.keras"
    assert os.path.isfile("model/scaler.pkl"), "Отсутствует model/scaler.pkl"
    return InferenceService(
        model_path="model/model.keras", scaler_path="model/scaler.pkl", threshold=0.5
    )


def test_predict_binary_output(infer: InferenceService):
    cols = infer.feature_names
    X = pd.DataFrame(np.random.randn(4, len(cols)), columns=cols)

    labels = infer.predict_df(X)
    proba = infer.predict_proba_df(X)

    assert labels.shape == (4,)
    assert proba.shape == (4,)
    assert set(np.unique(labels)).issubset({0, 1})
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_alignment_with_extra_and_missing_columns(infer: InferenceService):
    cols = infer.feature_names

    df = pd.DataFrame(
        {
            "id": [1, 2],
            "diagnosis": ["M", "B"],
            "Unnamed: 32": [np.nan, np.nan],
            cols[0]: [0.1, -0.2],
            cols[1]: [1.0, 2.0],
            "totally_extra_col": [123, 456],
        }
    )

    labels = infer.predict_df(df)
    assert labels.shape == (2,)
    assert set(np.unique(labels)).issubset({0, 1})
