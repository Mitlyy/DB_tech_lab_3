import os
import redis
from flask import Flask, request, render_template, send_file
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os.path as pt


load_dotenv()

redis_host = os.getenv('REDIS_HOST', '192.168.0.17')
redis_port = os.getenv('REDIS_PORT', 6379)
redis_password = os.getenv('REDIS_PASSWORD', 'your_password_here')


app = Flask(__name__, template_folder="templates")
main_path = pt.join(pt.dirname(__file__), "../")


redis_client = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)

model = joblib.load(pt.join(main_path, "model/model.pkl"))
scaler = joblib.load(pt.join(main_path, "model/scaler.pkl"))

@app.route('/')
def index():
    return render_template('index.html', accuracy=None, download_link=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', accuracy=None, download_link=None)

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', accuracy=None, download_link=None)
    
    df = pd.read_csv(file)
    if 'diagnosis' in df.columns:
        y_true = df['diagnosis'].map({'M': 1, 'B': 0}).values  
        df.drop(columns=['diagnosis'], inplace=True)
    else:
        y_true = None
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
    if 'Unnamed: 32' in df.columns:
        df.drop(columns=['Unnamed: 32'], inplace=True)
    

    features = scaler.transform(df.values)
    predictions = model.predict(features)
    

    predictions = np.where(predictions == 'M', 1, 0)
    df['prediction'] = predictions
    

    accuracy = None
    if y_true is not None:
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_true, predictions)
    

    results_path = pt.join(main_path, "results.csv")
    df.to_csv(results_path, index=False)
    

    redis_client.set('model_results', results_path)

    return render_template('index.html', accuracy=accuracy, download_link="/download")

@app.route('/download')
def download_file():
    results_path = pt.join(main_path, "results.csv")
    return send_file(results_path, as_attachment=True, download_name="predictions.csv")

@app.route('/fill_training_data', methods=['POST'])
def fill_training_data():
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    df = pd.read_csv(file)

    training_data_json = df.to_json(orient='split')

    redis_client.set('training_data', training_data_json)
    
    return "Training data successfully saved to Redis", 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)
