import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Create symptom vocabulary
symptoms = set()

for col in df.columns[1:]:

    cleaned_symptoms = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    symptoms.update(cleaned_symptoms)

# IMPORTANT
symptoms = sorted(list(symptoms))

# Create X and y
X = []
y = []

for _, row in df.iterrows():

    row_symptoms = set()

    for col in df.columns[1:]:

        if pd.notna(row[col]):
            row_symptoms.add(
                str(row[col]).strip()
            )

    feature_vector = []

    for symptom in symptoms:
        feature_vector.append(
            1 if symptom in row_symptoms else 0
        )

    X.append(feature_vector)
    y.append(row["Disease"])

# Encode labels AFTER loop
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Train XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"Accuracy: {accuracy:.4f}")

# Save everything
joblib.dump(model, "models/disease_model.pkl")
joblib.dump(symptoms, "models/symptoms.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")

print("Model Saved Successfully!")