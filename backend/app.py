from fastapi import FastAPI
from rag.rag_agent import search_disease
import joblib
import pandas as pd

app = FastAPI()

# Load model and encoder
model = joblib.load("models/disease_model.pkl")
all_symptoms = joblib.load("models/symptoms.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# Load datasets
description_df = pd.read_csv("data/symptom_Description.csv")
precaution_df = pd.read_csv("data/symptom_precaution.csv")
severity_df = pd.read_csv("data/Symptom-severity.csv")


@app.get("/")
def home():
    return {
        "message": "Clinical Decision Support System Running"
    }


@app.get("/predict")
def predict(symptoms: str):

    # Convert user input into list
    user_symptoms = [
        symptom.strip()
        for symptom in symptoms.split(",")
    ]

    # Create feature vector
    input_vector = []

    for symptom in all_symptoms:
        if symptom in user_symptoms:
            input_vector.append(1)
        else:
            input_vector.append(0)

    try:
        # XGBoost prediction
        prediction_encoded = int(
            model.predict([input_vector])[0]
        )

        print("Prediction Encoded:", prediction_encoded)
        print("Type:", type(prediction_encoded))

        prediction = encoder.inverse_transform(
            [prediction_encoded]
        )[0]

    except Exception as e:
        return {
            "error": str(e)
        }

    # Calculate Severity Score
    severity_score = 0

    for symptom in user_symptoms:

        row = severity_df[
            severity_df["Symptom"] == symptom
        ]

        if not row.empty:
            severity_score += int(
                row["weight"].values[0]
            )

    # Risk Level
    if severity_score < 10:
        risk_level = "Low"
    elif severity_score < 20:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Disease Description
    description_row = description_df[
        description_df["Disease"] == prediction
    ]

    if not description_row.empty:
        description = description_row[
            "Description"
        ].values[0]
    else:
        description = "Description not available."

    # Precautions
    precaution_row = precaution_df[
        precaution_df["Disease"] == prediction
    ]

    if not precaution_row.empty:
        precautions = [
            precaution_row["Precaution_1"].values[0],
            precaution_row["Precaution_2"].values[0],
            precaution_row["Precaution_3"].values[0],
            precaution_row["Precaution_4"].values[0],
        ]
    else:
        precautions = []

    return {
        "disease": prediction,
        "severity_score": severity_score,
        "risk_level": risk_level,
        "description": description,
        "precautions": precautions
    }


@app.get("/ask_rag")
def ask_rag(question: str):

    answer = search_disease(question)

    return {
        "answer": answer
    }