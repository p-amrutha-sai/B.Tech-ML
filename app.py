from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path

app = Flask(
    __name__,
    template_folder="web",
    static_folder="web",
    static_url_path="/web"
)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"

# Load the already-trained model once when the server starts.
model = joblib.load(MODEL_PATH)

FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form = {}

    if request.method == "POST":
        try:
            form = request.form.to_dict()

            customer = pd.DataFrame([{
                "gender": form["gender"],
                "SeniorCitizen": int(form["SeniorCitizen"]),
                "Partner": form["Partner"],
                "Dependents": form["Dependents"],
                "tenure": int(form["tenure"]),
                "PhoneService": form["PhoneService"],
                "MultipleLines": form["MultipleLines"],
                "InternetService": form["InternetService"],
                "OnlineSecurity": form["OnlineSecurity"],
                "OnlineBackup": form["OnlineBackup"],
                "DeviceProtection": form["DeviceProtection"],
                "TechSupport": form["TechSupport"],
                "StreamingTV": form["StreamingTV"],
                "StreamingMovies": form["StreamingMovies"],
                "Contract": form["Contract"],
                "PaperlessBilling": form["PaperlessBilling"],
                "PaymentMethod": form["PaymentMethod"],
                "MonthlyCharges": float(form["MonthlyCharges"]),
                "TotalCharges": float(form["TotalCharges"])
            }], columns=FEATURES)

            prediction = int(model.predict(customer)[0])

            # Find the probability column corresponding to class 1,
            # rather than assuming that class 1 is always at index 1.
            probability = None
            if hasattr(model, "predict_proba"):
                classes = list(model.classes_)
                if 1 in classes:
                    probability = float(model.predict_proba(customer)[0][classes.index(1)])

            result = {
                "prediction": prediction,
                "probability": probability
            }

        except Exception as exc:
            error = (
                "Prediction could not be completed. Check that the model "
                "expects these 19 input features and that its preprocessing "
                f"matches the training pipeline. Details: {exc}"
            )

    return render_template("index.html", result=result, error=error, form=form)


if __name__ == "__main__": app.run(host="127.0.0.1", port=5000, debug=True)
