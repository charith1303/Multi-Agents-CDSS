import pandas as pd

# Main Dataset
disease_df = pd.read_csv("data/dataset.csv")

# Disease Description
description_df = pd.read_csv("data/symptom_Description.csv")

# Precautions
precaution_df = pd.read_csv("data/symptom_precaution.csv")

# Severity
severity_df = pd.read_csv("data/Symptom-severity.csv")

print("All datasets loaded successfully!\n")

print("Disease Dataset:")
print(disease_df.shape)

print("\nDescription Dataset:")
print(description_df.shape)

print("\nPrecaution Dataset:")
print(precaution_df.shape)

print("\nSeverity Dataset:")
print(severity_df.shape)