import pickle
import os
import numpy as np

# Get the absolute path of the current file (model.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

# -----------------------------
# Load model and encoders safely
# -----------------------------
def load_pickle(file_name):
    path = os.path.join(MODEL_DIR, file_name)
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_pickle("fertilizer_model.pkl")
soil_encoder = load_pickle("soil_encoder.pkl")
crop_encoder = load_pickle("crop_encoder.pkl")
fertilizer_encoder = load_pickle("fertilizer_encoder.pkl")

# -----------------------------
# Prediction function
# -----------------------------
def predict_fertilizer(temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous):
    try:
        if soil_type not in soil_encoder.classes_:
            raise ValueError(f"Unknown soil type: {soil_type}. Available: {list(soil_encoder.classes_)}")

        if crop_type not in crop_encoder.classes_:
            raise ValueError(f"Unknown crop type: {crop_type}. Available: {list(crop_encoder.classes_)}")

        soil_type_encoded = soil_encoder.transform([soil_type])[0]
        crop_type_encoded = crop_encoder.transform([crop_type])[0]

        features = np.array([[temperature, humidity, moisture,
                              soil_type_encoded, crop_type_encoded,
                              nitrogen, potassium, phosphorous]])

        prediction = model.predict(features)
        fertilizer = fertilizer_encoder.inverse_transform(prediction)[0]

        return fertilizer

    except Exception as e:
        return f"Error in prediction: {e}"
