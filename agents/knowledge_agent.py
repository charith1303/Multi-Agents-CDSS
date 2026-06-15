import pandas as pd

description_df = pd.read_csv("data/symptom_Description.csv")

disease = "Malaria"

result = description_df[
    description_df["Disease"] == disease
]

print(result["Description"].values[0])