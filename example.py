import joblib

model = joblib.load("models/disease_model.pkl")
symptoms = joblib.load("models/symptoms.pkl")
encoder = joblib.load("models/label_encoder.pkl")

user_symptoms = [
    "itching",
    "skin_rash",
    "nodal_skin_eruptions"
]

input_vector = []

for symptom in symptoms:
    if symptom in user_symptoms:
        input_vector.append(1)
    else:
        input_vector.append(0)

prediction_encoded = int(
    model.predict([input_vector])[0]
)

prediction = encoder.inverse_transform(
    [prediction_encoded]
)[0]

print("Encoded:", prediction_encoded)
print("Disease:", prediction)