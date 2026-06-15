import pandas as pd

# Simulated output from Disease Prediction Agent
predicted_disease = "Malaria"

# User symptoms
user_symptoms = [
    "chills",
    "vomiting",
    "high_fever"
]

# Load datasets
description_df = pd.read_csv("data/symptom_Description.csv")
precaution_df = pd.read_csv("data/symptom_precaution.csv")
severity_df = pd.read_csv("data/Symptom-severity.csv")

# Get Description
description = description_df[
    description_df["Disease"] == predicted_disease
]["Description"].values[0]

# Get Precautions
precautions = precaution_df[
    precaution_df["Disease"] == predicted_disease
]

# Calculate Severity
severity_score = 0

for symptom in user_symptoms:

    row = severity_df[
        severity_df["Symptom"] == symptom
    ]

    if not row.empty:
        severity_score += row["weight"].values[0]

# Risk Level
if severity_score < 10:
    risk = "Low"
elif severity_score < 20:
    risk = "Medium"
else:
    risk = "High"

# Report
print("\n===== CLINICAL REPORT =====\n")

print("Predicted Disease:")
print(predicted_disease)

print("\nSymptoms:")
for symptom in user_symptoms:
    print("-", symptom)

print("\nDescription:")
print(description)

print("\nSeverity Score:", severity_score)
print("Risk Level:", risk)

print("\nPrecautions:")

for i in range(1, 5):
    print(
        f"{i}.",
        precautions[f"Precaution_{i}"].values[0]
    )