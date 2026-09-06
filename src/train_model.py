import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH        = os.path.join(BASE_DIR, "data", "tyre_historical_data.csv")
HEALTH_MODEL_PATH = os.path.join(BASE_DIR, "models", "tyre_health_model.pkl")
RUL_MODEL_PATH   = os.path.join(BASE_DIR, "models", "tyre_rul_model.pkl")

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

df = pd.read_csv(DATA_PATH)

features = [
    "tyre_age_days",
    "distance_km",
    "pressure_psi",
    "temperature_c",
    "vibration",
    "speed_kmh",
    "pressure_change",
    "temperature_change",
    "vehicle_load_kg",
    "braking_intensity",
    "wear_ratio",
]

X = df[features]

# ========================================
# MODEL 1: TYRE HEALTH CLASSIFICATION
# ========================================

print("\n" + "=" * 60)
print("TRAINING TYRE HEALTH CLASSIFICATION MODEL")
print("=" * 60)

label_encoder = LabelEncoder()
y_health = label_encoder.fit_transform(df["tyre_health_status"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_health, test_size=0.20, random_state=42, stratify=y_health
)

health_model = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=5,
    random_state=42, class_weight="balanced"
)
health_model.fit(X_train, y_train)

health_pred = health_model.predict(X_test)
print(f"\nHealth Model Accuracy: {accuracy_score(y_test, health_pred):.4f}\n")
print(classification_report(y_test, health_pred, target_names=label_encoder.classes_))

joblib.dump({"model": health_model, "label_encoder": label_encoder, "features": features}, HEALTH_MODEL_PATH)
print(f"Health model saved to: {HEALTH_MODEL_PATH}")

# ========================================
# MODEL 2: REMAINING USEFUL LIFE
# ========================================

print("\n" + "=" * 60)
print("TRAINING TYRE REMAINING USEFUL LIFE MODEL")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, df["remaining_life_km"], test_size=0.20, random_state=42
)

rul_model = RandomForestRegressor(
    n_estimators=200, max_depth=20, min_samples_split=5,
    random_state=42, n_jobs=-1
)
rul_model.fit(X_train, y_train)
rul_pred = rul_model.predict(X_test)

print(f"\nRUL Model MAE:  {mean_absolute_error(y_test, rul_pred):.2f} km")
print(f"RUL Model RMSE: {root_mean_squared_error(y_test, rul_pred):.2f} km")
print(f"RUL Model R²:   {r2_score(y_test, rul_pred):.4f}")

joblib.dump({"model": rul_model, "features": features}, RUL_MODEL_PATH)
print(f"RUL model saved to: {RUL_MODEL_PATH}")

# ========================================
# FEATURE IMPORTANCE
# ========================================

importance_df = pd.DataFrame({
    "feature":          features,
    "health_importance": health_model.feature_importances_,
    "rul_importance":    rul_model.feature_importances_,
})

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)
print(importance_df.sort_values("rul_importance", ascending=False))
