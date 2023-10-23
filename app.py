import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
CORS(app)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["30/minute"],
)

with open("heart_disease_classifier_model2", "rb") as f:
    model = pickle.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    user_data = {
        "age": [data["age"]],
        "sex": [data["sex"]],
        "cp": [data["cp"]],
        "trestbps": [data["trestbps"]],
        "chol": [data["chol"]],
        "fbs": [data["fbs"]],
        "restecg": [data["restecg"]],
        "thalach": [data["thalach"]],
        "exang": [data["exang"]],
        "oldpeak": [data["oldpeak"]],
        "slope": [data["slope"]],
        "ca": [data["ca"]],
        "thal": [data["thal"]],
    }

    user_data_df = pd.DataFrame.from_dict(user_data)

    uci_data = pd.read_csv("heart-disease (1).csv")
    uci_data = uci_data.drop("target", axis=1)

    scaler = StandardScaler().fit(uci_data)

    user_data_df_sc = scaler.transform(user_data_df)

    prediction = model.predict(user_data_df_sc)

    result = {
        "prediction": int(prediction[0]),
        "message": "The model predicts that you have heart disease."
        if prediction[0] == 1
        else "The model predicts that you do not have heart disease.",
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
