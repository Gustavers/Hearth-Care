from flask import Flask, render_template, request
from models.heart_model import HeartDiseasePredictor
from config import Config

app = Flask(__name__)

predictor = HeartDiseasePredictor(Config.DATASET_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    feature_importance = None

    if request.method == "POST":
        try:
            input_features = {
                'age': float(request.form.get('age', 0)),
                'sex': int(request.form.get('sex', 0)),
                'cp': int(request.form.get('cp', 0)),
                'trestbps': float(request.form.get('trestbps', 0)),
                'chol': float(request.form.get('chol', 0)),
                'fbs': int(request.form.get('fbs', 0)),
                'thalach': float(request.form.get('thalach', 0)),

            }

            prediction = predictor.predict_heart_disease(input_features)
            feature_importance = predictor.get_feature_importance()

        except ValueError as ve:
            error = str(ve)
        except Exception as e:
            error = f"Terjadi kesalahan: {str(e)}"

    model_metrics = predictor.get_model_metrics()

    return render_template("index.html",
                           prediction=prediction,
                           error=error,
                           metrics=model_metrics,
                           feature_importance=feature_importance)

if __name__ == "__main__":
    app.run(debug=True)
