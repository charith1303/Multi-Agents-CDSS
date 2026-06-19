from fastapi import FastAPI
import joblib
import pandas as pd
import os

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
            model_path = "models/disease_model.pkl"

            print("=" * 60)
            print("Loading Model From:", model_path)
            print("File Exists:", os.path.exists(model_path))
            print("=" * 60)

            model = joblib.load(model_path)

            print("✅ Disease Model Loaded Successfully")
            print("Model Type:", type(model))

        except Exception as e:
            print("❌ Model Loading Error:", str(e))
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