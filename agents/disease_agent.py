import joblib

# Load saved model
model = joblib.load("models/disease_model.pkl")

# Load symptom dictionary
symptoms = joblib.load("models/symptoms.pkl")

# User symptoms
user_symptoms = [
    "continuous_sneezing",
    "cough"
]

# Create input vector
input_vector = []

for symptom in symptoms:
    if symptom in user_symptoms:
        input_vector.append(1)
    else:
        input_vector.append(0)

# Predict disease
prediction = model.predict([input_vector])

print("Predicted Disease:", prediction[0])