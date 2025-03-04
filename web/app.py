from flask import Flask, request, render_template, send_file
import joblib
import numpy as np
import pandas as pd
import os.path as pt

app = Flask(__name__, template_folder="templates")

main_path = pt.join(pt.dirname(__file__), "../")

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
    
    return render_template('index.html', accuracy=accuracy, download_link="/download")

@app.route('/download')
def download_file():
    results_path = pt.join(main_path, "results.csv")
    return send_file(results_path, as_attachment=True, download_name="predictions.csv")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)
