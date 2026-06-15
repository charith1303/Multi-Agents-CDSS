import pandas as pd

precaution_df = pd.read_csv("data/symptom_precaution.csv")

disease = "Malaria"

result = precaution_df[
    precaution_df["Disease"] == disease
]

print("Precautions:\n")

for i in range(1, 5):
    precaution = result[f"Precaution_{i}"].values[0]
    print(f"{i}. {precaution}")