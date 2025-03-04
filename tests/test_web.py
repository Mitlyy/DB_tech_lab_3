import os
import sys
import io
import pytest
import pandas as pd
import numpy as np
from flask import template_rendered

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from web.app import app  

@pytest.fixture
def client():
    app.config['TESTING'] = True
    client = app.test_client()
    yield client

def test_index(client):
    """ Проверка загрузки главной страницы """
    response = client.get('/')
    assert response.status_code == 200
    assert "Загрузите CSV-файл для предсказания" in response.data.decode('utf-8')

def test_upload_no_file(client):
    """ Проверка загрузки без файла """
    response = client.post('/upload', data={})
    assert response.status_code == 200
    assert "Точность модели" not in response.data.decode('utf-8')

def test_upload_invalid_file(client):
    """ Проверка загрузки пустого файла """
    data = {'file': (io.BytesIO(b""), 'empty.csv')}
    response = client.post('/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert "Точность модели" not in response.data.decode('utf-8')

def test_upload_valid_file(client):
    """ Проверка загрузки корректного CSV-файла """
    df = pd.DataFrame(np.random.rand(5, 30))
    file = io.StringIO()
    df.to_csv(file, index=False)
    file.seek(0)

    data = {'file': (file, 'test.csv')}
    response = client.post('/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert "Точность модели" in response.data.decode('utf-8')

def test_download(client):
    """ Проверка скачивания файла с предсказаниями """
    response = client.get('/download')
    assert response.status_code == 200
    assert response.headers["Content-Disposition"].startswith("attachment")
