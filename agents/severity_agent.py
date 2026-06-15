import pandas as pd

severity_df = pd.read_csv("data/Symptom-severity.csv")

user_symptoms = [
    "chills",
    "vomiting",
    "high_fever"
]

severity_score = 0

for symptom in user_symptoms:

    symptom_row = severity_df[
        severity_df["Symptom"] == symptom
    ]

    if not symptom_row.empty:
        severity_score += symptom_row["weight"].values[0]

print("Severity Score:", severity_score)

if severity_score < 10:
    print("Risk Level: Low")

elif severity_score < 20:
    print("Risk Level: Medium")

else:
    print("Risk Level: High")