from fastapi import FastAPI
import joblib
import pandas as pd
import os
from xgboost import XGBClassifier

app = FastAPI()

print("=" * 60)
print("NEW MODEL DEPLOYMENT LOADED")
print("Current Working Directory:", os.getcwd())
print("=" * 60)

# Global variables
model = None
all_symptoms = None
encoder = None

description_df = None
precaution_df = None
severity_df = None


def load_resources():
    global model
    global all_symptoms
    global encoder
    global description_df
    global precaution_df
    global severity_df

    if model is None:
        try:

            model_path = "models/disease_model.json"

            print("=" * 60)
            print("Loading Model From:", model_path)
            print("File Exists:", os.path.exists(model_path))
            print("=" * 60)

            model = XGBClassifier()
            model.load_model(model_path)

            print("Disease Model Loaded Successfully")
            print("Model Type:", type(model))

        except Exception as e:
            print("Model Loading Error:", str(e))
            raise

    if all_symptoms is None:
        print("Loading Symptoms...")
        all_symptoms = joblib.load(
            "models/symptoms.pkl"
        )

    if encoder is None:
        print("Loading Label Encoder...")
        encoder = joblib.load(
            "models/label_encoder.pkl"
        )

    if description_df is None:
        print("Loading Description Dataset...")
        description_df = pd.read_csv(
            "data/symptom_Description.csv"
        )

    if precaution_df is None:
        print("Loading Precaution Dataset...")
        precaution_df = pd.read_csv(
            "data/symptom_precaution.csv"
        )

    if severity_df is None:
        print("Loading Severity Dataset...")
        severity_df = pd.read_csv(
            "data/Symptom-severity.csv"
        )


@app.get("/")
def home():
    return {
        "message": "Clinical Decision Support System Running"
    }


@app.get("/predict")
def predict(symptoms: str):

    load_resources()

    user_symptoms = [
        symptom.strip()
        for symptom in symptoms.split(",")
    ]

    input_vector = [
        1 if symptom in user_symptoms else 0
        for symptom in all_symptoms
    ]

    try:

        prediction_encoded = int(
            model.predict([input_vector])[0]
        )

        prediction = encoder.inverse_transform(
            [prediction_encoded]
        )[0]

    except Exception as e:
        return {
            "error": str(e)
        }

    severity_score = 0

    for symptom in user_symptoms:

        row = severity_df[
            severity_df["Symptom"] == symptom
        ]

        if not row.empty:
            severity_score += int(
                row["weight"].values[0]
            )

    if severity_score < 10:
        risk_level = "Low"
    elif severity_score < 20:
        risk_level = "Medium"
    else:
        risk_level = "High"

    description_row = description_df[
        description_df["Disease"] == prediction
    ]

    if not description_row.empty:
        description = description_row[
            "Description"
        ].values[0]
    else:
        description = "Description not available."

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

    try:
        from rag.rag_agent import search_disease

        answer = search_disease(question)

        return {
            "answer": answer
        }

    except Exception as e:
        return {
            "error": str(e)
        }