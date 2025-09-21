import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
import os
from utils import encode_features  # make sure utils.py exists with encode_features()

# -----------------------------
# Load dataset
# -----------------------------
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fertilizer_dataset.csv")
df = pd.read_csv(DATA_PATH)

# -----------------------------
# Encode categorical variables
# -----------------------------
df, soil_encoder, crop_encoder, fertilizer_encoder = encode_features(df)

# -----------------------------
# Features (X) and Target (y)
# -----------------------------
X = df.drop(['Fertilizer Name'], axis=1)
y = df['Fertilizer Name']

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Hyperparameter tuning with GridSearchCV
# -----------------------------
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train, y_train)

# -----------------------------
# Best model
# -----------------------------
best_clf = grid_search.best_estimator_

# -----------------------------
# Evaluate model
# -----------------------------
y_pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"[INFO] Model accuracy after tuning: {accuracy:.2f}")

# -----------------------------
# Save model & encoders
# -----------------------------
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
os.makedirs(SAVE_DIR, exist_ok=True)

joblib.dump(best_clf, os.path.join(SAVE_DIR, "fertilizer_model.pkl"))
joblib.dump(soil_encoder, os.path.join(SAVE_DIR, "soil_encoder.pkl"))
joblib.dump(crop_encoder, os.path.join(SAVE_DIR, "crop_encoder.pkl"))
joblib.dump(fertilizer_encoder, os.path.join(SAVE_DIR, "fertilizer_encoder.pkl"))

print(f"[INFO] Model and encoders saved successfully at: {SAVE_DIR}")
