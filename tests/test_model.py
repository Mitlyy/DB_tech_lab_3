import os
import joblib
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import model, scaler


def test_model_load():
    """ Проверка загрузки модели и скейлера """
    assert model is not None
    assert scaler is not None

def test_model_prediction():
    """ Проверка работы модели на случайном входе """
    X_test = np.random.rand(1, 30)
    X_scaled = scaler.transform(X_test)
    prediction = model.predict(X_scaled)
    assert prediction in [0, 1]

def test_data_processing():
    """ Проверка предобработки данных """
    df = pd.DataFrame(np.random.rand(5, 30))  
    assert not df.isnull().values.any()  
    assert df.shape[1] == 30  
