import io
import os
import sys
from typing import Any, Dict, List

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import InferenceService

APP_HOST = "0.0.0.0"
APP_PORT = 4000

app = Flask(__name__)

_inference_service: InferenceService | None = None


def get_inference() -> InferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService(
            model_path="model/model.keras",
            scaler_path="model/scaler.pkl",
            threshold=0.5,  # Единый порог бинаризации
        )
    return _inference_service


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


@app.route("/upload", methods=["POST"])
def upload():
    """
    Принимает multipart/form-data с CSV-файлом (поле 'file').
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

        # Готовим CSV для скачивания
        buffer = io.StringIO()
        out_df.to_csv(buffer, index=False)
        buffer.seek(0)

        return send_file(
            io.BytesIO(buffer.getvalue().encode("utf-8")),
            mimetype="text/csv",
            as_attachment=True,
            download_name="predictions.csv",
        )
    except Exception as e:
        return jsonify({"error": f"Ошибка инференса: {e}"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON API.
    Поддерживаем ДВА формата:
    1) records: {"instances": [ {feature: value, ...}, ... ]}
    2) matrix:  {"data": [[...], [...]], "feature_names": ["mean radius", ...]? (опционально)}

    Рекомендуемый — формат 1 (records), т.к. порядок фич тогда не важен.
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

        req_feature_names: List[str] | None = None
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
